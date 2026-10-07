# ReSnake Gym

ReSnake Gym 是一个让强化学习策略通过标准化虚拟游戏手柄玩贪吃蛇的工程项目。
当前目标是在完整的 `31 × 20` 棋盘上通关；测试通过、短程存活或偶尔吃到食物
都不算完成这一目标。

策略不直接输出蛇的转向，而是输出 `float32[L, 20]` 手柄报告。环境执行时逐帧
接收一条 20 维报告，并显式传递观测、推理、提交和执行时间戳。这一边界以后可以
替换为真实手柄、机械臂和灵巧手，而不改变游戏策略所使用的控制语义。

## 当前实现

- 两个 Gymnasium 环境：基础环境 `resnake_gym/ReSnake-v1`，以及带可解性约束
  障碍物的 `resnake_gym/ReSnakePerturbed-v1`。
- 两条受支持的训练路线：循环 PPO 加成功经验 SIL 回放，以及分布式 actor 加
  V-trace 事件信用轨迹。前者落实“发现成功经验后保留并继续训练”的明确需求；
  后者处理异步 actor/learner 的策略滞后和跨 fragment 信用，不替代 SIL。
- 可变棋盘输入：策略使用蛇头局部编码和全局注意力，不要求为每种棋盘尺寸重建
  固定尺寸全连接输入。
- 标准化手柄 API、延迟与丢帧模拟、实时运行入口、checkpoint 恢复、W&B 同步
  和低开销网页预览。
- 当前尚无通过完整 `31 × 20` 棋盘通关验收的模型。

仓库只维护上述当前路径。旧版方向动作环境、规则教师、BC、固定路径 oracle，
以及多算法或多 seed 对比 harness 已从当前工程入口删除。PPO/SIL 和 V-trace 都是
当前受支持实现，不把其中任何一条写成唯一训练路线。

## 安装

项目要求 Python 3.10 或更高版本。建议使用独立 Conda 环境：

```bash
conda create -n resnake-gym python=3.11 -y
conda activate resnake-gym
python -m pip install -e ".[rl,render]"
```

只使用环境且不需要 PyTorch 或窗口渲染时：

```bash
python -m pip install -e .
```

大规模训练、checkpoint 和预览归档应保存在 GPU 节点的数据盘，不要复制到本地
工作区。

## 环境接口

- `ReSnake-v1` 的观测为 9 通道 `float32[C,H,W]` 场景张量。
- `ReSnakePerturbed-v1` 的观测为带有效棋盘 mask、障碍物和待执行扰动信息的
  14 通道方形填充张量。
- 环境动作始终是一条 `float32[20]` 标准化手柄报告；策略一次可生成
  `float32[L,20]` 动作块，再由运行时按目标时间逐条提交。
- 数字按钮范围为 `[0,1]`，摇杆范围为 `[-1,1]`，扳机范围为 `[0,1]`。
- 默认棋盘是 `31 × 20`，随机旋转默认关闭。

20 个字段的顺序、时间戳语义和动作块协议见
[虚拟手柄 API](docs/gamepad-api.html)。

最小环境示例：

```python
import gymnasium as gym
import numpy as np
import resnake_gym  # 注册环境

env = gym.make(
    "resnake_gym/ReSnakePerturbed-v1",
    width=31,
    height=20,
    random_rotation=False,
)
observation, info = env.reset(seed=2026)
report = np.zeros(20, dtype=np.float32)

terminated = truncated = False
while not (terminated or truncated):
    observation, reward, terminated, truncated, info = env.step(report)

env.close()
```

这段代码只展示接口，不是可学习的策略，也不是成绩。

## 训练

PPO/SIL 路线用新采集的 on-policy 数据更新 PPO，并把自主产生的成功前缀保存在
按序列数限制的 replay 中供 SIL 重复学习。默认使用适合 Snake 的 `dpad5` 动作头；`raw` 动作头
仍受支持，但只供显式选择，不是实体控制的默认动作头。默认事件采样在每环境至少
1 次决策且全局观察到 1 次取食后停止，每环境最多采 512 次决策：

```bash
python scripts/train_gamepad_ppo.py \
  --output /data/lyy/resnake_gym/runs/ppo-sil \
  --device cuda \
  --updates 10000 \
  --save-every 10
```

完整 episode 不会逐条无界落盘；可恢复 checkpoint 只保存按序列数限制容量的 SIL
replay 状态和当前成功快照。这个上限不是字节或 transition 上限。

V-trace 路线使用分布式 actor 和跨 fragment 的事件信用轨迹：

```bash
python scripts/train_gamepad_vtrace.py \
  --output /data/lyy/resnake_gym/runs/main \
  --device cuda \
  --max-fresh-logic-ticks 1000000 \
  --updates 10000 \
  --save-every 10
```

`--max-fresh-logic-ticks` 是主采样预算，`--updates` 是安全上限。actor 将固定长度
fragment 作为传输单元；learner 会按同一环境流拼接到取食、死亡、截断或信用轨迹
安全上限，再计算 V-trace target。fragment 长度不是一局游戏，也不是人为规定模型
只能关注多少步。

两条路线都要求每次运行使用新的输出目录。目录中主要包含：

- `checkpoint.pt`：模型、优化器和恢复训练所需的状态；
- `metrics.jsonl`：逐次更新的训练指标；
- `config.json`：配置、版本和源码身份；V-trace另有与checkpoint配对的
  `runtime.pt` assembler状态。

从 V-trace checkpoint 恢复时仍写入一个新的目录：

```bash
python scripts/train_gamepad_vtrace.py \
  --resume /data/lyy/resnake_gym/runs/main/checkpoint.pt \
  --output /data/lyy/resnake_gym/runs/main-resumed \
  --device cuda \
  --max-fresh-logic-ticks 1000000 \
  --updates 10000
```

PPO/SIL 同样使用新输出目录恢复；`--updates`表示本次调用继续执行的更新数，产物中的
`update`、`policy_version`和`policy-NNNNNN.pt`沿 checkpoint 的累计轴继续编号：

```bash
python scripts/train_gamepad_ppo.py \
  --resume /data/lyy/resnake_gym/runs/ppo-sil/checkpoint.pt \
  --output /data/lyy/resnake_gym/runs/ppo-sil-resumed \
  --device cuda \
  --updates 1000
```

## 验收

训练是否成功只由模型在完整 `31 × 20` 棋盘上的独立完整游戏判断。默认工程验收
不要求跑 seed 矩阵，也不以多个算法的横向对比替代“模型是否会玩”这个问题。

两类 checkpoint 共用同一个评估入口：

```bash
python scripts/evaluate_gamepad_policy.py \
  /data/lyy/resnake_gym/runs/main/checkpoint.pt \
  --sizes 31x20 \
  --episodes 1 \
  --output /data/lyy/resnake_gym/runs/main/evaluation.json
```

## 可选运行工具

以下工具服务于训练观察和真实时间控制，不是额外实验 harness：

```bash
# 自动播放最近的完整策略回放；旧预览按数量与空间上限过期
python scripts/live_preview.py \
  /data/lyy/resnake_gym/runs/main \
  --site /data/lyy/resnake_gym/runs/main-preview \
  --retention 6 \
  --max-storage-mib 128

# 将本地 JSONL 指标增量同步到 W&B
python scripts/sync_wandb.py \
  /data/lyy/resnake_gym/runs/main \
  --entity YOUR_ENTITY \
  --project resnake-gym

# 用 module:factory 策略适配器运行墙钟调度与时间戳审计
RESNAKE_CHECKPOINT=/data/lyy/resnake_gym/runs/main/checkpoint.pt \
RESNAKE_DEVICE=cuda \
python scripts/run_realtime_gamepad.py \
  --policy resnake_gym.policy_adapter:from_environment \
  --output /data/lyy/resnake_gym/runs/realtime
```

内置adapter会声明checkpoint的动作块长度和timing-v2要求，实时runner自动采用这两项；
显式传入不一致的`--horizon`或`--no-timing-v2`会在创建输出目录前报错。

预览从 checkpoint 生成独立完整游戏，用于回答“最近保存的策略实际如何行动”；
它不回灌训练数据，也不阻塞 actor/learner。实时入口与快速 Gymnasium 采样是两个
不同执行模式，快速训练结果本身不能证明满足墙钟延迟要求。

## 开发检查

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -m build
```

自动测试验证接口、因果边界、恢复与数值实现，不等价于模型已经学会游戏。

## 进一步说明

- [代码结构](docs/architecture.md)
- [虚拟手柄 API](docs/gamepad-api.html)
- [扰动与实时任务设计](docs/task123-design.html)
- [障碍物可解性](docs/solvability.html)
- [PPO/SIL 成功经验设计](docs/ppo-success-training-design.md)
- [V-trace 长程信用设计](docs/vtrace-progress-replay-design.md)

历史设计文档保留在 `docs/` 中用于追溯讨论，不代表其中每条实验路线仍是当前
可执行入口。当前命令和维护范围以本 README 与 `scripts/` 实际内容为准。
