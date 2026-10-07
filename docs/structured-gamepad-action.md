# 结构化手柄动作头：raw 与 D-pad 5 类

## 1. 要改的不是环境接口

`ReSnake-v1`的环境接口不变，仍只接收`[L,20]`标准化手柄报告。模型面向下游的
张量接口也仍是`[B,L,20]`。当前PPO/SIL同时保留`raw`和`dpad5`两个内部动作头，
默认使用`dpad5`：每个chunk位置先从五个有效方向中选一个，再展开成完整手柄报告。
`raw`必须由调用者显式选择，它保留完整控制器头，但不是实体控制的默认动作头。

五个类别按以下顺序固定：

| 类别 | 含义 | 20维报告中的非零项 |
| ---: | --- | --- |
| 0 | neutral | 无 |
| 1 | up | `dpad_up=1` |
| 2 | down | `dpad_down=1` |
| 3 | left | `dpad_left=1` |
| 4 | right | `dpad_right=1` |

记这五行报告为模板矩阵`T ∈ R^(5×20)`。策略对每个batch和chunk位置输出五个
logit，采样

```text
z[b,l] ~ Categorical(logits[b,l,:])
report[b,l,:] = T[z[b,l],:]
```

模板只做固定索引，不读取棋盘、蛇头、墙、食物或当前方向。它不会把危险动作改成
安全动作，也不会替策略规划下一步。

这个表示与
[Gymnasium的MultiDiscrete手柄示例](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiDiscrete)
一致：方向键本来就可以写成`NOOP/UP/RIGHT/DOWN/LEFT`五选一，而不必解释为四个
可以任意并发的独立按钮。外部接口仍保留20维，是为了不把当前游戏的内部训练表示
误写成通用手柄标准。

## 2. 为什么原始六因子与Snake语义不合

当前raw动作头生成完整20维报告，Snake训练目标只计入四个D-pad Bernoulli因子和
`lx, ly`两个Gaussian因子。这样已经去掉了ABXY、右摇杆和扳机等无关维度的
score-function噪声，但六个因子之间仍是独立的。它和游戏绑定存在以下错位：

- `up+down`、`left+right`可以同时采到，相反方向先抵消；
- 两个正交D-pad形成平局时不产生方向请求；
- 只要任一D-pad为1，左摇杆就被整段忽略；
- 大量不同的轴幅值只对应同一个主轴方向；
- 中立、冲突和被游戏规则忽略的反向请求都可能得到相同的状态转移。

因此，raw头在不同报告之间分配概率和熵，并不等于在五种有效方向之间分配概率和熵。
[Baram等对动作冗余的分析](https://proceedings.mlr.press/v161/baram21a.html)
指出，多动作映射到同一转移时，动作熵与转移熵并不一致。该论文不是D-pad5算法的
来源，但它准确描述了这里的冗余问题。D-pad5的目的只是让策略分布的一个类别对应
一个合法、可区分的方向请求。

这也解释了为什么不能只在采样以后“清理”raw报告。如果先按旧分布采样，再把冲突
按钮修成单一方向，环境执行的动作已经不是PPO所记录概率的原样本。D-pad5直接定义
新的分布，并对该分布本身计算概率。

## 3. PPO使用精确Categorical概率

设第`l`个chunk位置的类别概率为`p_l`，实际采样类别为`z_l`。一段动作块的概率和
熵是

```text
log pi(z | h) = sum_l log p_l[z_l]
H(pi(. | h)) = sum_l -sum_k p_l[k] log p_l[k]
```

PPO概率比仍按原论文计算：

```text
ratio = exp(log pi_new(z | h) - log pi_old(z | h))
```

这里没有20个控件概率相乘，也没有把模板报告反推成六个独立动作。训练时保存和重算
的随机变量都是整数类别`z`。这一写法对应
[PPO论文](https://arxiv.org/abs/1707.06347)的离散策略目标；
[CleanRL PPO参考代码](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
也用`Categorical.log_prob()`和`Categorical.entropy()`形成离散PPO目标。
[OpenAI Baselines的分布实现](https://github.com/openai/baselines/blob/master/baselines/common/distributions.py)
则分别实现了Categorical、MultiCategorical、Bernoulli和Gaussian，说明“控制器由
多个字段组成”并不要求所有字段都使用独立Bernoulli。

当前chunk各位置仍是条件独立的Categorical样本，联合log probability是逐位置求和。
改成D-pad5并没有解决动作块内部的随机相关性、跨chunk信用分配或循环状态陈旧问题。

实现把raw权重继续标为`gamepad-ppo-v3 / snake-causal-marginal-v1`，D-pad5另标为
`gamepad-ppo-v4 / snake-dpad5-categorical-v1 / xinput-dpad5-categorical-v1`。
两种checkpoint不能静默互载。这是为了防止同一个actor参数被解释成20维raw logit
或5维类别logit，也让后续曲线能按动作语义分开审计。

## 4. PPO路径与可微下游路径分开

离散类别的PPO更新使用上节的score-function概率，不经过straight-through（ST）
代理。另一个接口用于以后把策略输出接到可微VLA或执行模型。令

```text
p = softmax(logits)
h = one_hot(argmax(logits))
y_st = h - stop_gradient(p) + p
report_st = y_st @ T
```

前向的`report_st`仍是五个模板之一，反向则沿softmax传递有偏代理梯度。这个构造与
[Jang等的Gumbel-Softmax](https://arxiv.org/abs/1611.01144)所讨论的ST离散代理属于
同一类做法，但当前确定性路径没有声称在采样Gumbel噪声。它更不表示游戏状态转移、
USB HID或机械接触可微。执行到NumPy、真实手柄或硬阈值以后，梯度链就结束。

必须保留两条路径的区分：

- PPO：采样整数类别，使用精确Categorical `log_prob`和`entropy`；
- 联合模型接口：硬模板前向，softmax ST反向，不参与PPO概率比。

把ST代理概率塞进PPO ratio，或把硬模板称为真实物理导数，都会改变这里的定义。

## 5. 为什么这不是手工技能

[ALE官方文档](https://ale.farama.org/env-spec/#minimal-action-set)允许为每个游戏使用
完成游戏所需的minimal action set，同时明确提醒这项设置会显著影响性能，实验必须
报告。D-pad5采用同一类约束：它删除控制器层面的冲突组合，不提供任何局面相关建议。
网络仍须从观测中学会何时上、下、左、右或保持，取食路线、避障和长远规划都没有被
写入模板。

这属于[action-space shaping](https://arxiv.org/abs/2004.00980)，不是规则教师，
也不是人工示范。Action Space Shaping的实验表明，删除无意义组合和离散化连续动作
可能显著改变学习难度；因此checkpoint必须记录动作编码，不能把D-pad5成绩误写成
raw手柄策略或实体控制结果。

[Huang与Ontañón](https://arxiv.org/abs/2006.14171)研究的是依赖状态的invalid-action
mask。D-pad5不是这种mask：五个类别在所有状态都保留，策略在墙边仍可选择撞墙，
也可请求与当前方向相反的方向；环境继续按公开规则处理。D-pad5既不查询合法移动，
也不根据蛇头位置动态删除动作。

## 6. 限制

1. D-pad5只覆盖当前Snake所需的五种有效请求。它不能代表模拟量摇杆、扳机、ABXY
   或双手同时操作，不能作为机器人阶段的完整动作空间。
2. 所有非零D-pad模板都是满按。它没有学习按压力、持续时间或D-pad与左摇杆之间的
   物理实现选择。
3. 这一等价关系依赖当前游戏绑定。若游戏以后读取多个同时按键、摇杆幅度或其他控件，
   不能继续沿用D-pad5目标。
4. ST梯度是有偏代理。它只服务于可微模型之间的接口，不是D-pad5在真实设备上的
   梯度证明。
5. 缩小动作支持可能提高样本效率，也可能隐藏原始手柄控制的困难。即便D-pad5通关，
   也只能说明策略在这个公开动作集上学会Snake，不能据此宣称raw手柄策略或灵巧手
   已经学会。
6. D-pad5没有处理延迟、丢帧、异步动作覆盖、稀疏成功、长时信用分配和棋盘尺寸
   泛化；这些条件仍按各自实验验收。

目前尚无D-pad5学习结果。单元测试、有限loss或偶然取食都不是有效性证据。正式验收
仍是在完整31×20棋盘的冻结策略评估中真正填满棋盘；训练seed上的分数、存活时间和
小棋盘通关不能替代该标准。

## 7. 当前工程选择

- `dpad5`是Snake训练的默认动作头，也是当前V-trace支持的动作头；它始终展开为完整
  20维报告。
- `raw`仍由PPO/SIL支持，供明确需要完整控制器分布的调用者选择；它不会因保留而成为
  实体控制默认，也不能与`dpad5`checkpoint静默互载。
- `stick5`和曾用于因果诊断的H1 held-chunk头不是当前实现。相关旧文档只用于解释
  历史判断，不能写进当前checkpoint契约。
- 工程不维护raw/dpad5/stick5的批量消融或多seed调度。无论选择哪个受支持动作头，
  最终都由通用评估入口检查冻结策略能否在31×20棋盘完成整局。

## 8. 依据与采用范围

- Gymnasium `MultiDiscrete`：
  <https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiDiscrete>。
  采用方向键五分类的控制器表示示例；环境公开接口仍是本项目的20维`Box`。
- ALE minimal action set：<https://ale.farama.org/env-spec/#minimal-action-set>。采用“游戏
  可以声明较小动作集，但必须报告”的实验规范；没有复用Atari动作或评测成绩。
- Schulman等，PPO：<https://arxiv.org/abs/1707.06347>；CleanRL参考实现：
  <https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py>；OpenAI Baselines分布：
  <https://github.com/openai/baselines/blob/master/baselines/common/distributions.py>。
  用于核对Categorical采样、log probability、entropy与PPO ratio。
- Baram等，Action Redundancy：<https://proceedings.mlr.press/v161/baram21a.html>。
  用于说明动作熵与状态转移熵可能错位；当前没有实现其transition-entropy算法。
- Jang等，Gumbel-Softmax：<https://arxiv.org/abs/1611.01144>。只采用离散硬前向、
  连续代理反向的思想；PPO本身不用该代理。
- Huang与Ontañón，Invalid Action Masking：<https://arxiv.org/abs/2006.14171>。
  用于区分状态相关mask和本项目的固定支持集，不把D-pad5称为动态合法动作mask。
- Kanervisto等，Action Space Shaping：<https://arxiv.org/abs/2004.00980>。用于说明动作
  删除和离散化会改变视频游戏RL难度，因此必须公开checkpoint所用动作编码；当前
  工程不据此附带raw/dpad5/stick5对比harness。
