# 代码结构

这份说明只描述当前训练路径。历史脚本仍保留用于复现实验，但不作为新增功能的落点。

## 分层

`src/resnake_gym/envs/`保存游戏规则和可观测状态；`wrappers/`处理当前手柄动作块与
延迟。旧相对动作wrapper的实现位于`resnake_gym_legacy`，原wrapper模块只保留惰性
兼容导入。环境层和当前wrapper都不依赖PPO或V-trace。

`gamepad_runtime.py`负责模型、环境和批量观测的公共构造。PPO和V-trace都依赖它，
V-trace不再通过PPO模块取得这些函数。动作概率、报告解码和网络结构仍在
`models.py`。推理与评估所需的两类checkpoint统一由`policy_checkpoint.py`校验和加载；
训练恢复仍由各训练器恢复其优化器和运行态。评估与预览共用
`evaluation_protocol.py`中的环境字段，避免各自维护一份参数表。

PPO的版本与配置在`gamepad_ppo_contract.py`，训练器在`gamepad_ppo.py`。一次更新明确
分为采集、GAE与循环窗口准备、PPO优化、SIL优化、隐状态刷新和指标组装；纯数据结构
和窗口计算放在`training/ppo_update.py`。`training/ppo_run.py`负责恢复校验、实验元数据、
训练循环和checkpoint产物，命令行脚本只负责参数映射。

V-trace按职责拆成以下模块：

- `gamepad_vtrace_contract.py`：配置和版本契约；
- `gamepad_vtrace_fragments.py`：actor传输包及校验；
- `gamepad_vtrace_actor.py`：actor进程和参数发布；
- `gamepad_vtrace_credit.py`：credit trace拼接、接收侧审计和批次收集；
- `gamepad_vtrace_learner.py`：V-trace目标、窗口反传和优化；
- `training/vtrace_learner_audit.py`：动作分布漂移与信用轨迹指标；
- `gamepad_vtrace.py`：兼容导入入口，不放训练实现。

训练进程生命周期、指标和checkpoint分别在`training/vtrace_actor_pool.py`、
`training/vtrace_reporting.py`、`training/vtrace_metrics.py`和
`training/vtrace_checkpoint.py`；`training/vtrace_run.py`负责配置映射、恢复、单次更新和
完整训练事务。正式checkpoint的sidecar名为`runtime.pt`，只保存恢复接收守恒所需的
assembler状态。孤立fragment replay不属于当前算法，旧实现放在
`experimental/`；通用的progress replay存储实现也在该目录，顶层同名模块只保留
旧导入路径兼容。

checkpoint与runtime sidecar通过原子rename和previous sidecar抵抗进程在发布边界
中断；该路径没有对文件和目录执行fsync，因此不宣称能抵抗主机崩溃或断电。

独立checkpoint评估由`evaluation_run.py`执行；预览与结果分析的业务代码分别在
`preview/`和`analysis/`。W&B事件转换、同步循环及训练中checkpoint watcher也属于
`analysis/`。对应的`scripts/`文件只处理命令行、进程入口和旧调用方兼容。

`artifact_io.py`保存文件内容身份计算，`provenance.py`生成训练源码清单，
`process_status.py`提供后台工具共用的只读进程命令匹配；这些能力不属于某一种算法。

墙钟实时执行的IPC、tick调度、帧传输和原始trace写入在`realtime.py`；策略加载、
决策循环及timing audit产物由`realtime_runner.py`编排。`run_realtime_gamepad.py`只保留
命令行参数到runner配置的映射，不另维护一套时间戳或命令提交逻辑。

历史相对动作实验与当前手柄训练物理隔离：独立包`resnake_gym_legacy/v0/`保存
`ReSnake-v0`的PPO、Hamiltonian BC、oracle、环境／策略基准和视频复核实现，
`resnake_gym_legacy/relative_chunked/`保存早期相对动作块PPO及其评测。原有九个
`scripts/`路径只保留兼容入口：

- `train_ppo.py`、`train_bc.py`、`evaluate_policy.py`；
- `train_chunked_ppo.py`、`evaluate_chunked_policy.py`；
- `evaluate_oracle.py`、`benchmark_env.py`、`benchmark_policy_latency.py`、
  `record_video.py`。

已有命令和通过`runpy`读取辅助符号的调用方不需要修改。新增训练功能不得继续写入
这些历史实现。`relative_chunked/chunked_control.py`和`v0/distance_reward.py`同样属于
历史实现；`resnake_gym.wrappers`仅在旧调用方显式请求时加载它们。

## 依赖规则

允许的依赖方向是：

```text
scripts
  -> preview / analysis / training / algorithm modules
  -> runtime / models / envs / wrappers

algorithm modules
  -> runtime / models / envs / wrappers

envs / wrappers
  -X-> PPO / V-trace / scripts
```

公共环境状态通过`SnakeState`、`GamepadSnakeEnv.kernel`和显式kernel方法读取。脚本不应
再访问`SnakeEnv`的私有字段。旧v0环境仍以`LegacySnakeEnv`命名导出，避免和当前
gamepad环境混淆。

## 修改时必须守住的契约

1. 环境只接收完整的`[L, 20]`手柄报告；训练器不能直接提交蛇的方向动作。
2. requested、executed和policy latent是三种不同数据，不能互相替代概率或时间戳。
3. V-trace的received、trained、pending和ready tick/transition必须守恒。
4. food fragment可以结束传输包，但不能重置游戏或循环状态。
5. 预览是固定协议下的独立checkpoint评估，不是训练轨迹重放。
6. `experimental/`中的代码不得由正式训练入口隐式启用。
7. 冻结评估协议必须记录实际环境的`initial_snake_length`；构造环境时不得依赖一个
   未进入协议哈希的默认值。

## 本地质量门

```bash
conda run -n resnake-gym ruff check .
conda run -n resnake-gym ruff format --check .
PYTHONDONTWRITEBYTECODE=1 conda run -n resnake-gym pytest -q -p no:cacheprovider \
  --cov=resnake_gym --cov-report=term --cov-fail-under=80
conda run -n resnake-gym python -m build --no-isolation
```

这些检查验证接口和实现回归，不验证31×20棋盘上的学习效果。收敛、吞吐和实时性能
必须由单独实验给出曲线与原始产物。
