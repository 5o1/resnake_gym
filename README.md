# ReSnake Gym

## 当前开发接口：v1（2026-10-06）

新入口为 `resnake_gym/ReSnake-v1`，只接受标准控件语义的20维归一化手柄报告，
不接受蛇的相对转向。按钮、摇杆和扳机布局见 `src/resnake_gym/gamepad.py`。
可微模型接口见 `src/resnake_gym/models.py`；真实设备读取适配见
`src/resnake_gym/sdl_gamepad.py`。

[设计依据与未完成项](docs/gamepad-interface.html) 明确区分官方标准、论文方法和
本项目适配。当前已验证手柄接口、代理梯度、异步事件队列和循环PPO短流程，
不代表已有游戏能力或VLA联合训练结果。
[代码结构](docs/architecture.md)说明当前模块边界、依赖方向和修改时必须保持的契约。
当前v3/raw与v4/dpad5模型共用训练与评估入口 `scripts/train_gamepad_ppo.py` 和
`scripts/evaluate_gamepad_ppo.py`。循环V-trace另用
`scripts/train_gamepad_vtrace.py`，评估入口可以直接识别两类checkpoint。请在GPU
节点运行，输出目录必须新建：

当前源码会在提交前运行全量回归、Ruff检查和构建检查；具体用例数以CI输出为准，
不再把容易过期的测试数量写进接口说明。测试通过只验证实现，不是学习成绩。

```bash
python scripts/train_gamepad_ppo.py --output /data/lyy/resnake_gym/artifacts/my-v3-run --updates 20
python scripts/train_gamepad_ppo.py --action-head dpad5 --output /data/lyy/resnake_gym/artifacts/my-v4-run --updates 20
python scripts/train_gamepad_vtrace.py --batch-food-target 4 --bptt-window 128 --credit-trace-max-transitions 2048 --max-fresh-logic-ticks 1000000 --output /data/lyy/resnake_gym/artifacts/my-vtrace-credit2048 --updates 10000
python scripts/train_gamepad_vtrace.py --batch-food-target 4 --bptt-window 128 --credit-trace-max-transitions 128 --max-fresh-logic-ticks 1000000 --output /data/lyy/resnake_gym/artifacts/my-vtrace-credit128 --updates 10000
python scripts/train_gamepad_vtrace.py --action-head held_dpad5 --batch-food-target 4 --bptt-window 128 --credit-trace-max-transitions 2048 --max-fresh-logic-ticks 1000000 --output /data/lyy/resnake_gym/artifacts/my-vtrace-h1-credit2048 --updates 10000
python scripts/evaluate_gamepad_ppo.py /data/lyy/resnake_gym/artifacts/my-v3-run/checkpoint.pt --output /data/lyy/resnake_gym/artifacts/my-v3-run/evaluation.json
```

默认31×20棋盘、随机2—5帧决策间隔；这是延迟事件仿真，不是墙钟实时能力测试。
随机障碍和可选旋转已加入独立的 `resnake_gym/ReSnakePerturbed-v1`（14通道），
但默认训练和预览已关闭随机旋转；显式旋转接口只保留给单独对照实验。
旧v1保持9通道；当前训练入口已切到14通道、有效棋盘mask和policy-time-v2。
raw checkpoint版本为gamepad-ppo-v3，dpad5为gamepad-ppo-v4；两者拒绝互载及隐式载入旧权重。默认parallel动作头，
`--decoder gru`是待比较的循环解码器，不意味着随机采样已具备跨帧相关性。
时间输入字段与实时调用方式见[API第7节](docs/gamepad-api.html)。

首次v2短跑（pre-v3历史记录）：31×20、20次更新、8,650游戏tick；三个尺寸共
15局独立测试均0分。
GPU全量测试216项通过。实时样本仅16tick，时间审计通过但无游戏能力；
以上不是验收成绩。产物保留在GPU节点
`/data/lyy/resnake_gym/model-v2-JDXE6Aez/pilot`，不复制权重回本地。
旧训练脚本、旧延迟包装器及下文命令仍针对v0，不能直接用于v1。历史BC和规则教师
仅保留复现记录，不进入新的纯强化学习实验。

## 任务1—3与跨线程接口

训练入口现默认事件驱动采样：每环境最低128、最高512次决策；最低长度之后，
全体并行环境累计4次取食才提前更新。死亡和时间截断都不能满足配额；若到512仍
不足4次取食，则按硬上限更新，并在指标中记录缺口。`--collection-mode fixed`
保留固定长度对照；`--minimum-steps`、`--event-target`、`--rollout-steps`可配置。
取食片段不截断GAE或清空记忆，PPO仍使用这一版策略产生的全部失败与未完成数据。
循环更新保存采样时隐状态，默认用64步无梯度burn-in重建状态，再以不超过128步的
窗口反传；这不会延长GAE越过采样批次。未终止回合一旦取食，会立即保留从本局开头
到最新取食决策的SIL成功快照，并用该决策后的value作bootstrap；每个环境最多保留
一份，后续取食原子替换旧快照。完成回合仍进入自主SIL回放。成功序列与失败回合两层
都非空时，默认抽样目标各占一半。完整checkpoint会保存活动成功快照；resume时将其
提升为不绑定新环境槽位的独立成功fragment，但不恢复普通pending或原游戏。这不是
人工示范，也不把旧样本放进PPO概率比。
具体定义、依据和风险见[成功事件训练设计](docs/ppo-success-training-design.md)，新奖励与
环境限制见[事件采样与奖励修订](docs/gamepad-interface.html)。`--shaping-scale 0`关闭势函数，
`--gamma`与`--gae-lambda`用于单独研究信用分配跨度。示例中的20次更新是流程预算，
不是足量学习的承诺；当前尚无v3学习效果报告。

raw六因子手柄头与候选D-pad五分类头的语义、PPO概率及必做消融见
[结构化手柄动作设计](docs/structured-gamepad-action.md)。D-pad5尚无学习效果，不能用
接口测试代替31×20完整通关验收。

PPO/SIL短时域问题、循环IMPALA/V-trace、64步burn-in、五分类类别—请求报告—执行
报告三层审计，以及跨fragment的事件credit trace，见
[V-trace长程信用设计](docs/vtrace-progress-replay-design.md)。当前fragment只是最长128步的
传输包；learner按流拼到取食、terminal、truncation或独立安全界，默认最多2048步，
先在完整trace上算V-trace target，再以不超过128步的BPTT窗口累计梯度，最后只做一次
optimizer step。`gamma=.9999`、事件配额和2048安全界仍是实验假设。
训练入口的`--max-fresh-logic-ticks`按fragment第一次被接收的游戏tick计主预算，
`--updates`只是安全上限；日志分别核对received、trained、pending和ready守恒。
孤立fragment replay会重新截断信用，因此正式训练入口不再提供replay参数，也不会
创建replay对象。旧实现移入`resnake_gym.experimental`，只用于历史复现和单独对照。
第一轮跨度对照只改`--credit-trace-max-transitions 128/2048`，保持received tick预算、
`--bptt-window 128`、棋盘、seed和其余设置一致。

`--action-head held_dpad5` 是单次采样后保持整个手柄chunk的H1诊断项。它仍向环境
输出完整`[L,20]`报告，但V-trace只计算一个五分类变量的概率。它与默认J8动作头的
checkpoint及fragment语义互拒；其文献边界、现有槽执行率证据和表达力／熵尺度混杂
见[H1动作头说明](docs/held-dpad5-diagnostic.md)。

[虚拟手柄API](docs/gamepad-api.html) 是字段顺序、动作块时间戳、场景通道和
策略适配器的交接文档。[设计说明](docs/task123-design.html) 记录扰动的可解性
保证范围、文献依据、实时调度和闭环对照的限制。

障碍生成已解除相邻双格限制；[可解性验证说明](docs/solvability.html) 介绍必要
条件、一般覆盖环搜索及小规模完整状态验证。返回 `solvable / unsolvable / unknown`，
未判定不算无解。真实时钟运行已改为异步求解：预告时后台搜索，目标帧快速复核，
未就绪就跳过。裸环境仍默认同步，便于快速事件仿真；两种模式的时间预算和候选
筛选不同。自动采样会在生成几何前排除当前证书路径按自由格奇偶必然不能接受的
批量大小，但不会按位置反复采样到验证成功；正式实验仍需分别检查接受率，不能把
两种模式当成同一扰动分布。

训练线程提供 `module:factory`，返回 `reset(seed=...)` / `act(obs)->[L,20]`
适配器后，可运行：

```bash
python scripts/evaluate_closed_loop.py --policy my_adapter:load_policy --training-seeds 2026 --output /data/lyy/resnake_gym/artifacts/closed-loop.json
python scripts/run_realtime_gamepad.py --policy my_adapter:load_policy --output /data/lyy/resnake_gym/artifacts/realtime-run
```

`--fixture-smoke` 可替代 `--policy` 检查工具，但不构成模型成绩。实时入口独立
进程推进游戏；日志区分观测采集、推理、提交、接收和实际执行时间。当前只实现
同主机时钟域，跨主机时间校准与真实HID采样回传仍需接入。

## 历史版本：v0

`ReSnake Gym` 是一个面向强化学习实验的贪吃蛇环境，接口遵循
[Gymnasium](https://gymnasium.farama.org/)。环境直接输出完整棋盘张量，并把逻辑
帧率、动作重复、episode 上限和初始局面做成显式参数。默认棋盘为 31 × 20，和
早期的 [Snake_qt5](https://github.com/5o1/Snake_qt5) 一致。

环境 ID：`resnake_gym/ReSnake-v0`

仓库目前包括：

- Gymnasium 环境、终端 / RGB / pygame 三种渲染方式；
- Hamiltonian cycle 确定性 oracle；
- 环境吞吐测试和严格通关评测；
- 从零训练 PPO、Hamiltonian 行为克隆（BC）以及 BC 后继续 PPO；
- 可复查轨迹哈希的 MP4 录制脚本。

这里的“通关”只有一个含义：蛇真正填满棋盘。高分、存活到时间上限或接近填满都
不算通关。

## 安装

仅使用环境和无窗口渲染：

```bash
python -m pip install -e .
```

人工试玩：

```bash
python -m pip install -e ".[render]"
```

运行训练、策略评测和视频录制：

```bash
python -m pip install -e ".[rl,render]"
```

项目要求 Python 3.10 及以上。当前正式实验环境使用 Python 3.11.16、PyTorch
2.14.1、Gymnasium 1.3.0、Stable-Baselines3 2.9.0；GPU 环境为 PyTorch
2.14.1 + CUDA 12.6。建议先按 PyTorch 官方索引安装对应 CUDA wheel，再安装本
项目：

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu126 \
  "torch==2.14.1+cu126"
python -m pip install -e ".[rl,render]"
```

本项目不使用 `torchvision`。不要把依赖旧版 Torch 的 `torchvision` 混进实验环境；
若节点上已有这种包，应使用独立虚拟环境。小棋盘的 MLP 实验往往受环境推进和
CPU 推理限制，因此下文的已复现实验使用 `torch==2.14.1+cpu`，并不影响同一套
代码在正式 CUDA 12.6 环境运行。

## 最小示例

```python
import gymnasium as gym
import resnake_gym  # 导入时注册环境

env = gym.make(
    "resnake_gym/ReSnake-v0",
    action_mode="absolute",
    logic_fps=10.0,
    frame_skip=1,
)

observation, info = env.reset(seed=2026)
terminated = truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

Gymnasium 区分两种结束方式：撞墙、撞到身体或填满棋盘时
`terminated=True`；达到 tick 数或模拟时长上限时 `truncated=True`。调用方必须
同时检查两者。

## 观测

观测是 `np.float32` 数组，采用 CHW 排列，形状为
`(9, height, width)`。默认棋盘对应 `(9, 20, 31)`。各通道都在 `[0, 1]` 内：

| 通道 | 内容 |
| ---: | --- |
| 0 | 蛇所占的格子，蛇头和身体均为 1 |
| 1 | 身体顺序；蛇头为 1，沿身体递减，蛇尾为 `1 / length` |
| 2 | 蛇头位置 |
| 3 | 蛇尾位置 |
| 4 | 食物位置 |
| 5 | 蛇头朝上时，在蛇头位置为 1 |
| 6 | 蛇头朝右时，在蛇头位置为 1 |
| 7 | 蛇头朝下时，在蛇头位置为 1 |
| 8 | 蛇头朝左时，在蛇头位置为 1 |

Gymnasium 返回 NumPy 数组，不把环境绑定到某个深度学习框架。使用 PyTorch 时
可直接转换：

```python
import torch

state = torch.from_numpy(observation)
batch = state.unsqueeze(0).to("cuda")  # (1, 9, H, W)
```

## 动作

默认的 `action_mode="absolute"` 使用四个绝对方向：

| 动作 | 方向 |
| ---: | --- |
| 0 | 上 |
| 1 | 右 |
| 2 | 下 |
| 3 | 左 |

与当前方向正好相反的指令会被忽略，蛇继续直行。这与 `Snake_qt5` 的规则一致。

`action_mode="relative"` 使用三个相对动作，只供legacy v0及其旧训练脚本使用：

| 动作 | 含义 |
| ---: | --- |
| 0 | 直行 |
| 1 | 右转 |
| 2 | 左转 |

## 逻辑帧率、frame skip 与 episode 上限

一次逻辑 tick 移动一格。`logic_fps` 定义模拟时间中每秒包含多少个 tick，
`frame_skip` 定义一次 `env.step(action)` 最多重复多少个 tick：

```python
env = gym.make(
    "resnake_gym/ReSnake-v0",
    logic_fps=20.0,
    frame_skip=4,
)
```

若中途没有结束，这次 `step` 会移动 4 格，模拟时间前进 `4 / 20 = 0.2` 秒。
动作只在 `step` 开始时解释一次，然后保持得到的方向；相对动作“右转”不会在一次
调用里连续转四次。一旦碰撞、获胜或达到上限，执行会提前停止，实际 tick 数见
`info["ticks_advanced"]`。

`logic_fps` 是模拟时钟，不限制训练的墙钟速度。`render_mode=None`、`"ansi"` 和
`"rgb_array"` 都不会主动 `sleep`；只有 `"human"` 按逻辑帧率显示。

主要构造参数如下：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `width`, `height` | `31`, `20` | 棋盘宽和高，均不得小于 3 |
| `grid_size` | `None` | `(width, height)` 的可选简写 |
| `initial_length` | `3` | 默认初始蛇长 |
| `logic_fps` | `10.0` | 每个模拟秒的逻辑 tick 数 |
| `frame_skip` | `1` | 每次 agent 动作重复的 tick 数 |
| `action_mode` | `"absolute"` | `"absolute"` 或 `"relative"` |
| `max_logic_steps` | `None` | 按内部逻辑 tick 截断 episode |
| `max_episode_seconds` | `None` | 按模拟秒数截断 episode |
| `cell_size` | `20` | RGB / pygame 中每格的像素边长 |
| `living_reward` | `0.0` | 每个已执行 tick 的基础奖励 |
| `food_reward` | `1.0` | 吃到食物时增加的奖励 |
| `death_reward` | `-1.0` | 撞墙或身体时增加的奖励 |
| `win_reward` | `1.0` | 填满棋盘时额外增加的奖励 |
| `timeout_reward` | `0.0` | 时间上限截断时额外增加的奖励 |

一次 `step` 执行多个 tick 时，返回这些 tick 的奖励之和。`timeout_reward` 只在
内部时间上限触发时加一次。`gym.make(..., max_episode_steps=N)` 则会在环境外套
Gymnasium 的 `TimeLimit`，按调用 `step` 的次数计数；它和 `max_logic_steps` 不是
同一个量。直接构造 `SnakeEnv` 时还保留了 `max_episode_steps` 作为
`max_logic_steps` 的兼容别名，不要同时传入二者。

每次 `reset` 和 `step` 返回的 `info` 包括：

- `length`、`score`；
- `logic_fps`、`logic_dt`、`frame_skip`、`decision_fps`；
- `logic_steps`、`elapsed_seconds`、`ticks_advanced`；
- `termination_reason` 和 `won`。

`termination_reason` 可能为 `wall_collision`、`self_collision`、
`board_filled`、`time_limit` 或尚未结束时的 `None`。
直接使用环境对象时，还可以只读访问 `snake`（头到尾的坐标）、`head`、
`direction` 和 `food`；经 `gym.make` 创建后从 `env.unwrapped` 访问这些属性。

## 固定种子和指定局面

`reset(seed=...)` 控制初始食物及后续食物生成；默认蛇身位置是固定的。若策略从
动作空间随机采样，还应单独设置动作空间的种子：

```python
seed = 2026
env.action_space.seed(seed)
observation, info = env.reset(seed=seed)
```

`reset(options=...)` 可以设置 `snake`、`direction`、`food` 和本局的
`logic_fps`。坐标使用 `(x, y)`，原点在左上角；`snake` 按蛇头到蛇尾排列：

```python
observation, info = env.reset(
    seed=2026,
    options={
        "snake": [(5, 4), (4, 4), (3, 4)],
        "direction": "right",
        "food": (10, 4),
        "logic_fps": 20.0,
    },
)
```

`direction` 可用 `up`、`right`、`down`、`left`（不区分大小写）或对应的绝对
动作整数。显式局面会检查边界、重复格、身体连续性、朝向和食物冲突。

## 可选的距离奖励

`DistanceRewardWrapper` 给有效移动增加 Manhattan 距离进度：

```python
import gymnasium as gym

from resnake_gym.wrappers import DistanceRewardWrapper

base_env = gym.make(
    "resnake_gym/ReSnake-v0",
    action_mode="relative",
    timeout_reward=-1.0,
)
env = DistanceRewardWrapper(base_env, scale=0.1)
```

shaping 项为 `scale * (distance_before - distance_after)`。吃到食物后目标发生变化，
以及发生碰撞的终止步，都不加入这一项；完成了有效移动的时间截断步仍会计算。
这是普通奖励塑形，不是势函数不变性证明，比较算法时必须报告 `scale`。

## 渲染和试玩

创建环境时可选：

- `None`：不渲染，适合训练；
- `"ansi"`：`env.render()` 返回终端字符串；
- `"rgb_array"`：返回 `uint8`、HWC 排列的 RGB 图像；
- `"human"`：打开 pygame 窗口并按逻辑帧率播放。

```python
env = gym.make("resnake_gym/ReSnake-v0", render_mode="ansi")
env.reset(seed=0)
print(env.render())
env.close()
```

仓库自带随机策略和人工试玩入口：

```bash
python examples/random_agent.py --episodes 10 --seed 2026
python examples/random_agent.py --episodes 1 --ansi
python examples/human_play.py --logic-fps 8
```

人工试玩使用方向键或 WASD 转向，`R` 重新开始，`Esc` 退出。

## Oracle 和环境吞吐基准

`HamiltonianOracle` 沿一条固定 Hamiltonian cycle 行动。当矩形的至少一边为偶数
时，它可作为确定性正确性基线；默认的 31 × 20 棋盘满足条件。它不寻找捷径，
也不代表学得了通用贪吃蛇策略。

本节工具和后文的历史视频工具实现均隔离在`resnake_gym_legacy.v0`；保留原有
`scripts/*.py`兼容入口，因此以下命令及产物格式没有改变。

严格评测 oracle：

```bash
python scripts/evaluate_oracle.py \
  --width 31 --height 20 \
  --episodes 10 --start-seed 0 \
  --max-logic-steps 200000 \
  --require-perfect \
  --output-json oracle_31x20.json
```

环境吞吐基准分别在蛇长 3、25%、50% 和 90% 时测量完整的
`SnakeEnv.step`，包含动作处理、状态推进、观测、奖励和 `info`，不包含 reset、
策略推理和一次预热：

```bash
python scripts/benchmark_env.py \
  --width 31 --height 20 --duration 1 \
  --output-json reference_31x20.json
```

当前节点的一次参考测量为 46,572–49,311 logic steps/s。该数字来自 Python
3.11.16、Gymnasium 1.3.0、NumPy 2.4.6 的单机运行，只用于发现环境实现回退，
不应当当作跨机器性能承诺。

## legacy v0训练（不用于当前v3）

本节的`train_ppo.py`、`train_bc.py`等legacy v0脚本使用相对动作；它们不是当前
`train_gamepad_ppo.py`的20维手柄训练入口。若没有显式传入 `--max-logic-steps`，上限为
`capacity * (capacity + 1) / 2`。Stable-Baselines3 会按完整 rollout 收集数据，
所以实际 timesteps 通常略高于请求值。

为避免历史算法与当前代码混在一起，这些命令的实现位于`resnake_gym_legacy`；下述
`scripts/*.py`路径是兼容入口，命令行用法与历史产物格式保持不变。

`train_ppo.py` 会写入 `config.json`、`train_summary.json`、`final_model.zip`、
周期 checkpoint、Monitor CSV 和 TensorBoard 日志；`train_bc.py` 对应写入
`config.json`、`bc_summary.json`、`bc_model.zip` 和 TensorBoard 日志。

以下命令假定重资产根目录在 GPU 节点：

```bash
export RESNAKE_DATA_ROOT=/data/lyy/resnake_gym
```

### legacy v0从零训练PPO

下面是已复现的 4 × 4 shaping 配置：

```bash
python scripts/train_ppo.py \
  --width 4 --height 4 --seed 2026 \
  --total-timesteps 1000000 \
  --n-envs 64 --n-steps 128 --batch-size 512 --n-epochs 4 \
  --learning-rate 0.0003 --gamma 0.99 --gae-lambda 0.95 \
  --ent-coef 0.01 \
  --food-reward 1 --death-reward -1 --living-reward -0.001 \
  --win-reward 10 --timeout-reward -1 \
  --distance-reward-scale 0.1 \
  --device cpu --torch-threads 16 --checkpoint-every 250000 \
  --output-dir \
    "$RESNAKE_DATA_ROOT/artifacts/runs/ppo_scratch_shaped_4x4_seed2026_v1"
```

这是从随机初始化开始的纯 PPO，没有示范数据。距离奖励属于人工 shaping，因此结果
应写作 “scratch PPO + distance shaping”，不能简写成无先验 PPO。

### legacy v0 Hamiltonian行为克隆

`train_bc.py` 在线生成 oracle 状态，不在磁盘上保存示范数据。它对 PPO 的 actor
做交叉熵训练，并默认平衡直行、左转和右转三类动作：

```bash
python scripts/train_bc.py \
  --width 31 --height 20 --seed 2026 \
  --updates 2000 --batch-size 1024 --learning-rate 0.0003 \
  --eval-every 100 --device cpu --torch-threads 32 \
  --output-dir \
    "$RESNAKE_DATA_ROOT/artifacts/runs/bc_hamiltonian_31x20_seed2026_v1"
```

这个模型学习的是固定 Hamiltonian oracle。即使严格通关率为 100%，也只能报告为
“Hamiltonian BC”，不能报告成纯 RL 或一般性的自主探索结果。

### legacy v0 BC后继续PPO

`train_ppo.py --resume` 从现有 PPO archive 继续训练。已复现的 8 × 8 配置为：

```bash
python scripts/train_ppo.py \
  --width 8 --height 8 --seed 2026 \
  --resume \
    "$RESNAKE_DATA_ROOT/artifacts/runs/bc_hamiltonian_8x8_seed2026_v1/bc_model.zip" \
  --total-timesteps 500000 \
  --n-envs 64 --n-steps 256 --batch-size 1024 --n-epochs 4 \
  --learning-rate 0.00001 --gamma 0.999 --gae-lambda 0.95 \
  --ent-coef 0 \
  --food-reward 1 --death-reward -1 --living-reward -0.0001 \
  --win-reward 10 \
  --device cpu --torch-threads 16 --checkpoint-every 250000 \
  --output-dir \
    "$RESNAKE_DATA_ROOT/artifacts/runs/ppo_ft_bc_8x8_seed2026_v1"
```

`--resume` 不会抹掉示范初始化。该结果必须标为 “BC+PPO”，不能和从随机初始化的
PPO 放在同一类别里。

## 严格策略评测

严格通关同时要求：

```text
terminated is True
info["won"] is True
info["termination_reason"] == "board_filled"
```

碰撞和 `time_limit` 截断一律计为失败。`evaluate_policy.py` 使用确定性动作，按连续
seed 启动相互独立的环境，并支持批量推理；`--require-win-rate` 不达标时返回状态码
2：

```bash
python scripts/evaluate_policy.py \
  "$RESNAKE_DATA_ROOT/artifacts/runs/ppo_scratch_shaped_4x4_seed2026_v1/final_model.zip" \
  --algorithm ppo --width 4 --height 4 \
  --episodes 1000 --start-seed 500000 \
  --batch-envs 256 --torch-threads 16 --device cpu \
  --require-win-rate 0.99 \
  --output-json \
    "$RESNAKE_DATA_ROOT/artifacts/benchmarks/ppo_scratch_shaped_4x4_seed2026_eval1000.json"
```

### legacy v0已复现结果

| 方法 | 棋盘 | 训练量 | 测试 seeds | 严格通关 | 说明 |
| --- | ---: | ---: | --- | ---: | --- |
| PPO scratch，无距离 shaping | 4 × 4 | 请求 100k，实际 106,496 steps | 100000–100999 | 0 / 1000 | 早期失败对照；策略学会拖延，不能通关 |
| PPO scratch + distance shaping | 4 × 4 | 请求 1M，实际 1,007,616 steps | 500000–500999 | 997 / 1000 | 纯 PPO，但使用显式距离和超时奖励 |
| Hamiltonian BC | 8 × 8 | 2.048M 在线样本 | 200000–200999 | 1000 / 1000 | 固定 oracle 示范 |
| Hamiltonian BC + PPO | 8 × 8 | 2.048M BC 样本 + 请求 500k PPO steps | 300000–300999 | 1000 / 1000 | 不是纯 RL |
| Hamiltonian BC | 31 × 20 | 2.048M 在线样本 | 410000–410099 | 100 / 100 | 固定 oracle 示范 |

4 × 4 shaping PPO 的训练耗时为 90.52 s，1000 局批量评测耗时 1.54 s，平均
45.218 ticks；31 × 20 BC 的 100 局平均为 95,934.02 ticks。耗时依赖机器和线程
设置，通关数与 seed 范围才是主要结果。

最早的无 shaping smoke test 运行在迁移前的 PyTorch 2.10 环境，只作为失败案例
保留。其后的 shaping PPO、BC 和 BC+PPO 结果均使用 PyTorch 2.14.1；后续正式
实验入口固定为 PyTorch 2.14.1 / CUDA 12.6。

## 录制视频

`record_video.py` 支持 oracle、PPO 和 QR-DQN。长 episode 不逐帧全部保留，而是
完整保留开头和结尾，并从中段均匀采样。脚本先跑一遍确定轨迹与帧计划，再跑第二
遍流式写入 ffmpeg；两遍轨迹哈希不一致时不会发布视频。MP4 旁会写同名
`.mp4.json`，其中包括严格通关结果、模型 SHA-256、完整轨迹 SHA-256、采样 tick
和 ffmpeg 信息。

```bash
python scripts/record_video.py \
  --controller ppo \
  --model \
    "$RESNAKE_DATA_ROOT/artifacts/runs/ppo_ft_bc_8x8_seed2026_v1/final_model.zip" \
  --width 8 --height 8 --seed 300000 \
  --max-logic-steps 2080 --cell-size 48 --fps 30 \
  --output \
    "$RESNAKE_DATA_ROOT/artifacts/videos/ppo_ft_bc_8x8_seed300000.mp4"
```

需要 oracle 视频时省略 `--model` 并使用 `--controller oracle`。系统 ffmpeg 优先；
找不到时使用 `imageio-ffmpeg` 提供的二进制。

## 代码与重资产分开

本地工作区和 Git 仓库只保存源码、测试和文档。训练 checkpoint、TensorBoard
日志、完整评测 JSON、视频、wheel cache 和其他大文件放在计算节点。当前目录约定
为：

```text
/data/lyy/resnake_gym/
├── repo/                  # 仓库工作树
├── env/                   # Python / CUDA 环境
├── cache/                 # pip、conda 和 wheel 缓存
└── artifacts/
    ├── runs/              # config、monitor、TensorBoard、checkpoint
    ├── checkpoints/       # 需要单独整理的模型
    ├── benchmarks/        # 评测与吞吐 JSON
    └── videos/            # MP4 及 JSON sidecar
```

不要为了在 README 中展示结果而把 checkpoint、日志或 MP4 复制回 WSL。需要共享时
应发布独立 artifact，并保留相应的 JSON 元数据和哈希。

对于节点上被透明代理或 fake-IP DNS 干扰的 PyPI 下载，可用仓库中的辅助脚本按
确切版本下载兼容 wheel。它会清除继承的代理变量，通过指定网卡和 DoH 直连，并
核对 PyPI 提供的 SHA-256：

```bash
python scripts/download_pypi_wheels_direct.py \
  --output-dir "$RESNAKE_DATA_ROOT/cache/wheels" \
  "some-package==1.2.3"
```

若节点网络策略不同，不要照搬脚本默认的 `eno2`；通过 `--interface`、
`--doh-host` 和 `--doh-ip` 明确设置。直连确实失败时，再显式使用节点约定的
`127.0.0.1:17891` 代理，不使用系统默认代理。

## pre-v3纯RL历史实验的W&B曲线

run04（pre-v3）的历史指标与当时新增指标同步至
[resnake-gym / run04](https://wandb.ai/assanekowww/resnake-gym/runs/snake-e01e0feef1d846fc)。
`scripts/sync_wandb.py` 是独立日志读取进程，不修改训练、不加载 checkpoint，
也不上传轨迹、权重或源码。复用 GPU 节点已有 SDK 和登录凭据。

训练指标使用 `train/update` 横轴；确定性与采样评估各用自己的评估轮次横轴，
避免把迟到的评估误记为当前训练轮次。网页记录时间是同步时间，不是原始采集时间。
主指标为每局食物数、通关率和存活游戏 tick；loss 仅供诊断。
已使用的验证种子不当作最终未见测试集。

```bash
python scripts/watch_learning.py /absolute/gpu/run_directory \
  --pid TRAINING_PID --every 50 --episodes 30 --device cuda:3 &
RESNAKE_EVALUATION_PID=$!
python scripts/sync_wandb.py /absolute/gpu/run_directory \
  --entity assanekowww --project resnake-gym --training-pid TRAINING_PID \
  --evaluation-pid "$RESNAKE_EVALUATION_PID"
```

不传 `--training-pid` 时只补传一次。相同目录使用稳定 run ID，重启时从服务端
读取事件 ID 去重。网络中断时 SDK 可能暂存未发送记录，不能把本地入队计数当成
服务端已收到；可用 W&B Public API 回读验证。同步默认仅使用指定的 17891 代理。
训练、watcher与同步器都应传绝对run路径。正式运行把watcher PID交给同步器；同步器会
等训练和最终冻结评估都退出，再做两次无新增文件的扫描。只靠固定的墙钟等待会漏掉
200,000 tick长局在训练结束后才写出的验证结果。

## 有界的实时策略预览

`scripts/live_preview.py` 监视训练本来就会保存的轻量 `policy-*.pt`，在独立环境中
生成受控评估。页面只回答一个定性问题：在固定评估协议与同一初始随机流
（seed）下，最近 N 个 checkpoint 的确定性闭环完整单局行为如何演化。这不是训练轨迹，
也不是多 seed 泛化评估。同一 seed 也不意味后续场景逐格相同：策略会改变蛇身和
空格集合，因而后续食物映射与障碍候选仍可以分叉。

每个 checkpoint 只 reset 一次，直到撞击、`board_filled` 或环境正式 `time_limit`。
`time_limit` 在页面上标为“达到评估上限（删失）”，不算胜利。worker从环境内部在每个
游戏逻辑 tick 捕获状态，所以回放不是决策帧；手柄的一次决策跨过 2–5 tick 时，
这些游戏帧仍逐帧保留。事件列表随播放显示游戏开始、吃到食物、障碍预告、生效／
拒绝与本局结束，每条都带逻辑 tick、仿真时间、决策号和 policy version。
页面默认按整局约90秒自适应播放；可切换1×、10×、100×或1000×，也可直接跳到
本局结尾或下一checkpoint。加速只改变浏览器抽取显示帧的速度，归档仍保存每个游戏tick。

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nice -n 19 ionice -c 3 python scripts/live_preview.py RUN_DIRECTORY \
  --site RUN_DIRECTORY/live-preview --host 127.0.0.1 --port 8780 \
  --device cuda:0 --retention 6 --seed 730000 \
  --chunk-frames 256 --max-storage-mib 128
```

轨迹以增量 JSON 分块存储，每块都有独立关键帧；浏览器按需加载，且最多保留两个已解码分块。
服务端同时限制 checkpoint 数和轨迹总字节数；新建完整局若单独就超过上限，会拒绝发布，
不会把半局冒充成完整局。缓存键包含 checkpoint 内容 SHA-256、seed 与评估协议指纹；任一项
变化都会使旧回放过期。HTML、manifest 和分块都返回 `Cache-Control: no-store`。worker 每轮最多
补一个 checkpoint，可以放在低调度优先级与独立 GPU 上。服务只绑定远端 loopback，通过 SSH
`-L 127.0.0.1:8780:127.0.0.1:8780`访问，不直接暴露实验节点端口。

“不在训练热路径”不等于可以不测共享主机影响。以每10轮轻量策略文件的mtime
比较启用前后耗时；如果后续窗口持续超过关闭预览时的自然波动，应停止worker或
降低片段频率。当前页面中的训练轮次来自最新稳定策略，并不一定等于训练进程的
瞬时轮次。

## 开发检查

```bash
python -m pip install -e ".[dev,render]"
python -m ruff check .
python -m ruff format --check .
pytest
python -m build
```

测试同时覆盖当前手柄训练代码和legacy v0复现代码。后者包括Gymnasium checker、
规则边界、可复现seed、frame skip、时间截断、奖励塑形、Hamiltonian cycle、BC批
生成器、PPO resume和批量策略评测；这些legacy测试不表示当前v3使用BC、示范或
相对动作。
