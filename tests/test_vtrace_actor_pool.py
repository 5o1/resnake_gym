from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from resnake_gym.gamepad_vtrace_contract import VTraceConfig  # noqa: E402
from resnake_gym.training import vtrace_actor_pool as actor_pool_module  # noqa: E402


class FakeEvent:
    def __init__(self):
        self.was_set = False

    def set(self):
        self.was_set = True


class FakeQueue:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.closed = False
        self.joined = False

    def close(self):
        self.closed = True

    def join_thread(self):
        self.joined = True


class FakeProcess:
    def __init__(self, *, target, args, name, fail_start=False):
        self.target = target
        self.args = args
        self.name = name
        self.fail_start = fail_start
        self.start_calls = 0
        self.join_calls = 0
        self.terminate_calls = 0
        self.alive = False
        self.exitcode = None

    def start(self):
        self.start_calls += 1
        if self.fail_start:
            raise RuntimeError("synthetic actor start failure")
        self.alive = True

    def join(self, timeout):
        assert timeout == 5
        self.join_calls += 1

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminate_calls += 1
        self.alive = False
        self.exitcode = -15


class FakeContext:
    def __init__(self, *, fail_actor=None):
        self.fail_actor = fail_actor
        self.event = FakeEvent()
        self.queue = None
        self.processes = []

    @staticmethod
    def Value(kind, value):
        assert kind == "q"
        return SimpleNamespace(value=value)

    @staticmethod
    def Lock():
        return object()

    def Event(self):
        return self.event

    def Queue(self, *, maxsize):
        self.queue = FakeQueue(maxsize)
        return self.queue

    def Process(self, *, target, args, name):
        process = FakeProcess(
            target=target,
            args=args,
            name=name,
            fail_start=len(self.processes) == self.fail_actor,
        )
        self.processes.append(process)
        return process


def _config():
    return VTraceConfig(
        dim=8,
        actor_processes=2,
        envs_per_actor=1,
        unroll_length=1,
        recurrent_burn_in=1,
        bptt_window=1,
        credit_trace_max_transitions=1,
        batch_min_transitions=1,
        batch_max_transitions=1,
        batch_food_target=0,
        queue_capacity=3,
    )


def test_actor_pool_owns_publication_collection_and_idempotent_shutdown(monkeypatch):
    context = FakeContext()
    monkeypatch.setattr(
        torch.multiprocessing,
        "get_context",
        lambda method: context if method == "spawn" else None,
    )
    published = []

    def fake_publish(model, shared, version, lock, policy_version):
        published.append(policy_version)
        version.value = policy_version
        return policy_version

    monkeypatch.setattr(actor_pool_module, "publish_parameters", fake_publish)
    collected = ([{"trace": 1}], {"fresh_logic_ticks": 1})

    def fake_collect(queue, event, config, assembler, *, actor_processes):
        assert queue is context.queue
        assert event is context.event
        assert config is pool.config
        assert assembler is sentinel_assembler
        assert actor_processes == context.processes
        return collected

    monkeypatch.setattr(
        actor_pool_module,
        "collect_fresh_credit_traces",
        fake_collect,
    )
    pool = actor_pool_module.VTraceActorPool(
        _config(),
        torch.nn.Linear(2, 1),
        run_generation=4,
    )
    sentinel_assembler = object()

    with pytest.raises(RuntimeError, match="not running"):
        pool.collect(sentinel_assembler)
    assert pool.start(policy_version=0) == 0
    assert pool.publish(policy_version=3) == 3
    assert pool.collect(sentinel_assembler) == collected
    with pytest.raises(RuntimeError, match="already started"):
        pool.start(policy_version=3)

    pool.close()
    pool.close()

    assert published == [0, 3]
    assert context.event.was_set
    assert context.queue.maxsize == 3
    assert context.queue.closed and context.queue.joined
    assert [process.start_calls for process in context.processes] == [1, 1]
    assert [process.terminate_calls for process in context.processes] == [1, 1]
    assert [process.join_calls for process in context.processes] == [2, 2]
    assert context.processes[0].args[0:2] == (0, 4)
    assert context.processes[1].args[0:2] == (1, 4)


def test_actor_pool_cleans_up_started_processes_when_later_start_fails(monkeypatch):
    context = FakeContext(fail_actor=1)
    monkeypatch.setattr(torch.multiprocessing, "get_context", lambda method: context)
    monkeypatch.setattr(
        actor_pool_module,
        "publish_parameters",
        lambda model, shared, version, lock, policy_version: policy_version,
    )
    pool = actor_pool_module.VTraceActorPool(
        _config(),
        torch.nn.Linear(2, 1),
        run_generation=0,
    )

    with pytest.raises(RuntimeError, match="synthetic actor start failure"):
        pool.start(policy_version=0)

    assert pool.closed
    assert context.event.was_set
    assert context.processes[0].terminate_calls == 1
    assert context.processes[0].join_calls == 2
    assert context.processes[1].join_calls == 0
    assert context.queue.closed and context.queue.joined
