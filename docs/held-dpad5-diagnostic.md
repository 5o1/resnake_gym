# H1：单次采样、整段保持的手柄动作头

## 为什么加这条对照

当前 J8 动作头在一次决策里独立采样 `L` 个
`neutral/up/down/left/right` 类别。环境收到的仍是 `[L,20]` 标准手柄报告，
但一次 V-trace 动作的联合概率是这 `L` 个类别概率的乘积。H1 用来检查一个很窄的
问题：联合动作过宽，是否让因果信用和重要性比率变差。

现有 J8 执行审计给出了更直接的理由。2381 个 transition 中，chunk 第 0—7 槽在
同一 transition 内被实际执行的比例依次为
`32.85 / 65.86 / 72.86 / 47.85 / 23.38 / 0 / 0 / 0%`；按整局追踪，第 7 槽仍是
`0/2381`。约 69.96% 的 joint negative log-likelihood 来自当次 transition 没有
执行的槽，约 57.85% 来自该局最终也没有执行的槽；约 3.75% 的 chunk 八个槽全未
执行。这些数字是当前项目运行记录的诊断统计，不是外部论文结论；正式报告还要把
它们绑定到确定的 run、checkpoint 和审计脚本版本。它们说明 J8 的 score 中确实有
大量未形成当次物理动作的随机变量，但不能单独证明 H1 会学得更好。

H1 每次决策只采样一个五分类变量，再把该类别对应的 20 维手柄报告原样重复 `L`
次。环境 API、观测延迟、命令延迟、丢包、decision tick 数和奖励均不改。它不读取
棋盘来屏蔽动作，也不修复反向输入或碰撞。

| 项目 | J8 `dpad5` | H1 `held_dpad5` |
|---|---:|---:|
| 每次决策的随机变量数 | `L` | 1 |
| 策略 latent | `[B,L]` | `[B]` |
| 完整 behavior 分布 | `[B,L,5]` | `[B,1,5]` |
| 对环境输出 | `[B,L,20]` | `[B,L,20]` |
| joint log-prob | `sum(j=1..L)` | 单个类别的 log-prob |

H1 的可微 `forward` 也只做一次 softmax 和一次 straight-through 离散化，然后沿
chunk 维复制。复制出来的槽不是新的随机变量，V-trace、entropy 和 KL 都只能计一
次。checkpoint 中的 `action_head`、`training_objective` 和 `action_encoding` 会把 H1
与 J8 隔开，不能互相恢复。

## 与已有做法的关系

### DQN 的 action repeat

Mnih 等人的 DQN Atari 设置在若干模拟器帧内重复所选动作，用较低的决策频率训练；
论文与补充材料见 [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236)。
这是常见的时间抽象。H1 借用了“一个选择保持若干帧”这个操作，不照搬 Atari 的
四帧协议：本项目仍提交带时间槽的完整手柄 chunk，实际到达和执行由既有异步手柄
链路决定，且一次决策推进的游戏 tick 数仍在配置区间内随机变化。

### ALE 的评测协议

Machado 等人提出 sticky actions：每个模拟器帧以给定概率继续执行上一个动作，而
不是执行代理刚请求的动作。官方 ALE 文档把默认概率写为 0.25，并要求明确报告该
设置，见 [ALE Environment Specifications](https://ale.farama.org/env-spec/#action-repeat-stochasticity)
和 [Revisiting the Arcade Learning Environment](https://jair.org/index.php/jair/article/view/11182)。
sticky action 会随机改变真正执行的动作；H1 不增加这种随机性。已有命令延迟、丢
包和执行审计继续原样工作。

### 延迟与动作状态增广

固定动作延迟可通过把待执行动作队列并入状态转成标准 MDP；Chen 等人给出了对应的
DA-MDP 构造和等价性证明，见
[Delay-Aware Model-Based Reinforcement Learning](https://arxiv.org/abs/2005.05440)。
随机观测/动作延迟的状态增广分析可见
[Revisiting State Augmentation Methods for Reinforcement Learning with Stochastic Delays](https://arxiv.org/abs/2108.07555)。
当前模型已经接收实际报告、按时间排序的执行历史、上一命令 chunk、各自年龄和时间
特征；H1 不删这些输入，也不声称单靠保持动作解决了延迟 POMDP。

## 能回答和不能回答的问题

同等游戏 tick 预算、seed 和其余配置下，如果 H1 的食物事件率、score 分布、ESS
或 held-out 评估明显好于 J8，可以支持“J8 的 chunk 联合动作是当前训练瓶颈之一”。
它不能证明保持动作是最终手柄模型，也不能证明真实双手控制只需低频动作。H1 会失
去 chunk 内改变方向的表达能力；真实机械臂阶段仍需测量控制频率、时延和接触状态。
因此它只作为因果信用诊断和消融项保留。

H1 与 J8 不是只差“无效槽是否计概率”，至少还有三个混杂因素：

1. H1 不能在一个 chunk 内预先安排转向，表达力低于 J8；在急转弯场景中这可能直接
   变差。
2. J8 最大类别熵为 `L log 5`，H1 为 `log 5`。沿用同一 entropy coefficient 会让
   H1 的最大熵奖励缩小 L 倍；把 H1 系数乘 L 又会改变优化尺度。至少应同时报告同
   系数和最大熵奖励匹配两种设置，并记录 raw entropy 与每随机变量 entropy。
3. H1 把探索噪声在时间上完全相关，J8 则独立。重要性比率、NLL 尺度和动作持续时
   间都随之改变，不能把全部差异只归因于“信用分配”。

因此首轮比较应使用相同 fresh logic tick 预算、配对 seed 和冻结评估集，至少包含
J8、H1 同熵系数、H1 熵奖励匹配三组。除分数外还要比较每槽执行率、整段未执行率、
NLL 中未执行槽占比、ESS、rho 截断率、动作切换率及急转弯死亡。H1 若改善，只能作
为缩小后续动作研究范围的证据，不能直接定为最终动作头。
