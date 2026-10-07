# 成功事件采样与循环 PPO/SIL 设计

## 1. 这次修改解决什么

当前目标仍是让策略从零通过强化学习控制标准手柄，在31×20棋盘上完成贪吃蛇，
不是只验证训练脚本能运行。此前实现有三个会妨碍这个目标的问题：

1. Snake不读取的手柄控件也进入PPO联合log probability，给策略梯度加入无关噪声；
2. 想让一批采样覆盖取食事件时，死亡也能满足事件配额，批次仍可能主要由短失败组成；
3. 增大rollout后若直接对整段循环网络反传，计算和显存随采样长度一起增长；稀少的
   成功经验也需要在发现时保留，不能只等一个可能很长的episode结束。

当前实现分别采用Snake因果边缘概率、只计取食的可变长度采样、存储状态加短窗口
BPTT，以及带取食快照的成功分层SIL。它们是四项可拆开的项目适配，不能作为一个已经验证有效的
新算法来宣传。目前只有接口和训练语义，尚无v3学习曲线或31×20通关结果。

## 2. 不变的外部接口

环境始终只接受`[L,20]`手柄报告；模型面向下游的接口是
`torch.Tensor[B,L,20]`。14个按钮、4个摇杆轴和2个扳机轴全部保留，字段顺序遵循
`xinput-normalized-v1`。不能把Snake方向、三分类相对转向或六维内部训练子集暴露成
新的动作API。

模型的确定性`forward`仍为20维报告保留PyTorch计算图。按钮采用硬前向、软反向的
代理梯度；它只说明模型之间可以传递代理梯度，不说明离散游戏、物理手柄或机器人
接触可微。执行边界转换为NumPy或HID后也不再承诺梯度。

## 3. Snake因果边缘概率

### 3.1 因果子集

当前Snake绑定只读取：

- 按钮0—3：`dpad_up, dpad_down, dpad_left, dpad_right`；
- 轴14—15：`lx, ly`。

其余14个手柄控件仍被模型采样和输出，但不会影响本游戏的状态转移或奖励。记完整
动作块为`a=(a_C,a_N)`，其中`a_C`包含所有时间槽上的上述六个控件，`a_N`是其余
控件。当前按钮与轴分布在控件维上独立；可选`ChunkNormal`只让同一控件沿chunk时间
相关。因此

```text
pi(a_C | h) = sum_or_integral pi(a_C, a_N | h) da_N
```

等于直接保留六个控件因子的概率。PPO新旧策略比、SIL的`log pi`和熵都只对这个
边缘分布求和。完整20维样本不被裁掉。

仅删掉当前step的无关log probability还不够。如果`a_N`通过实际报告、执行历史或
上次动作块进入GRU，它仍能改变以后`a_C`的分布，边缘化就会有偏。当前策略因此在
三条反馈路径上使用同一个Snake内部遮罩：`report`、`history`和`previous_chunk`
进入循环记忆前只保留因果六维。公开观测和日志仍保存全部20维。

这个选择受到
[Factored Policy Gradients](https://proceedings.neurips.cc/paper_files/paper/2021/hash/2ba8698b79439589fdd2b0f7218d8b07-Abstract.html)
对动作—目标影响结构和梯度方差的分析启发，但当前实现不是FPG：没有其factor
baseline或影响网络，只利用了本环境可以直接检查的绑定关系。论文同时指出错误结构
会造成偏差；因此以下任一变化都必须重新审计，而不能沿用当前遮罩：

- 游戏开始读取ABXY、扳机、右摇杆等控件；
- 策略分布在不同控件之间建立相关性；
- 机械执行误差让一个名义无关控件影响有效控件；
- 非因果控件重新进入后续策略状态。

## 4. 只计取食的可变长度采样

训练CLI当前默认值以`scripts/train_gamepad_ppo.py`为准：

| 参数 | 默认值 | 精确含义 |
| --- | ---: | --- |
| `collection_mode` | `events` | 使用下述取食配额；程序类`PPOConfig`本身仍默认`fixed` |
| `num_envs` | 4 | 同步完成一次并行环境轮次后才检查停止条件 |
| `minimum_steps` | 128 | 每环境至少采128次策略决策 |
| `rollout_steps` | 512 | 每环境的硬上限，不是游戏tick数 |
| `event_target` | 4 | 全体环境合计的取食事件数，不是完成回合数 |

达到128次决策以后，只要全局累计取食数达到4，本批次以`food_target`结束。死亡和
时间截断会继续记录，但绝不满足配额。到512次仍没有足够取食时以`hard_limit`结束，
训练不会等待一个可能永远不出现的成功；日志同时记录目标是否达到和缺少几次取食。

一批始终由同一版策略采集。取食只是审计边界，不被改成`done`，不会清空GRU，也不
截断GAE。PPO使用整批的失败、普通移动和未完成片段；“尽可能包含成功经验”不等于
丢弃失败样本，或把成功片段当成人工标签。

这不是
[PPO](https://arxiv.org/abs/1707.06347)
论文的采样停止规则，而是本项目为稀疏事件做的适配。停止时间依赖策略产生的取食数，
可能改变有限批次的组成。实验必须保留`collection_mode=fixed`作为对照，并按相同环境
交互量而不是相同update数比较。food quota提高每批观察到取食的机会，不保证每批达到
目标，更不证明策略在改善。

## 5. 长rollout与短BPTT分开

奖励、value、终止标记和`gamma ** K_i`折扣先按完整采样批次计算GAE。GAE按时间
顺序跨越普通取食事件，但不跨episode重置，也不跨本次批次边界。增大rollout因此能
增加一批内的优势估计跨度，却没有消除折扣衰减或批次末端bootstrap误差。

循环网络的反向传播另行切窗。当前CLI默认：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `recurrent_burn_in` | 64 | learn窗口前最多重放64次决策，不保留梯度 |
| `recurrent_unroll` | 128 | 单个learn窗口最多反传128次决策 |

采样时保存每个转移之前的GRU状态。训练窗口按环境和episode边界划分，从存储状态开始
重放burn-in，再detach并对learn区间反传。每个on-policy样本每个epoch恰好属于一个
learn区间，窗口不能跨reset；各窗口损失按本批总样本数归一化并累积后，每个epoch
执行一次优化器更新。

存储循环状态和burn-in的动机来自
[R2D2](https://openreview.net/forum?id=r1lyTjAqYX)
对recurrent state staleness的研究。这里没有采用R2D2的DQN目标、分布式actor、固定
replay序列或优先TD误差；把这项实现称为R2D2会误导。它只是on-policy PPO中的循环
状态重建方式。

## 6. 成功分层SIL

[Self-Imitation Learning](https://proceedings.mlr.press/v80/oh18b.html)
用智能体自己的历史好经验训练正优势动作。当前实现沿用其核心形式：只在
`R-V(s)>0`时加入策略项`-log pi(a|s) * stop_gradient(R-V)`和价值残差项。
SIL是PPO之外的off-policy更新；旧回放样本从不进入PPO的新旧策略概率比。作者的
[参考实现](https://github.com/junhyukoh/self-imitation-learning)
用于核对方法语义，但本项目没有引入演示、BC、DAgger或规则教师。

### 6.1 取食边界快照与完整episode

各环境的活动episode可以跨多个PPO update留在独立`pending`中，不会在批次边界被
拼到别的环境或下一局。未终止回合一旦在某次决策中取食，回放器立即深拷贝从本局
开头到该决策的完整前缀，并用该决策后的value作尾部bootstrap，再计算各步return。
这份前缀当即属于成功层，可以参加本次采样结束后的SIL update。

每个活动环境最多保留一份成功快照。后续再次取食时，新增长前缀通过临时文件替换
旧快照，而不是保存多个高度重叠的前缀。episode真正终止或时间截断后，完整episode
按实际terminal或截断bootstrap重新计算return并进入完成池；对应活动快照随即删除。
完成episode只要出现过取食、`score>0`或严格通关就属于成功层，否则属于失败层。

checkpoint保存这些replay-only成功快照，但不保存普通`pending`、活动环境状态或随机
数流。resume会重开游戏，并把旧活动快照提升为受保护的独立成功fragment：它不再
绑定原来的环境槽位，仍可采样，也绝不与新游戏的pending拼接；新回合结束不会删除
它。这保住了最近成功前缀，但不能逐bit继续原局，快照尾部value也仍有估计误差。

### 6.2 保存和抽样

相关训练CLI默认值如下：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `sil_updates` | 2 | 每个PPO update之后执行的SIL update数 |
| `sil_batch_size` | 1 | 每次SIL抽取的可采样序列数；可以是完成episode、活动快照或恢复后的fragment |
| `replay_capacity` | 256 | 完成episode及恢复后独立fragment共用的内存容量；活动快照另计 |
| `sil_priority_alpha` | 0.0 | 层内优先级指数；0表示层内不按优先级偏置 |
| `replay_success_fraction` | 0.5 | 成功、失败两层都存在时分给成功层的采样质量 |

默认主池容量为256个序列，容纳完成episode及resume时由快照提升的独立成功fragment。
溢出时先淘汰最早失败，失败不能挤掉已保存成功；只有池内全是成功才淘汰最早成功。
正在运行的活动快照不占主池容量，但每环境最多一份，因此可采样序列数最多是
`replay_capacity + num_envs`。完成回合的GPU归档不随内存淘汰删除；活动快照使用
每环境固定文件，只保存最新前缀，并在正常完成episode归档时删除。resume则将其另存
为不绑定环境的retained success fragment。

默认`replay_success_fraction=0.5`。成功与失败两层都存在时，序列采样的总质量
各占0.5；成功层包含完成成功episode、活动取食快照和恢复后保留的成功fragment，
失败层只含完成失败episode。
只有一层时，质量全部退回现有层。层内以序列长度形成均匀转移目标，
实际采样为：

```text
0.1 * length_mass + 0.9 * length_times_priority_power_alpha_mass
```

损失使用层内“目标概率/实际概率”修正，但不校正回全局均匀转移分布，否则会抵消
明确设置的成功比例。`alpha=0`只关闭层内优先级作用，并不关闭成功分层。成功保护、
0.5目标比例、序列优先级和层内校正都是项目适配，不是SIL论文原样实现；它们必须与
未分层回放做等预算消融。

SIL当前仍从被抽中序列的起点前向重建GRU，每128步detach一次，但会扫描到序列末尾。
因此计算量仍为O(采样序列总长度)，没有使用PPO的存储状态+burn-in窗口。很长的完成
回合或很晚才取食的快照可能成为吞吐瓶颈；在实测之前不把它写成已经解决的长序列方案。

## 7. checkpoint与可审计指标

新checkpoint格式为`gamepad-ppo-v3`，同时写入并校验四个语义版本：

```text
training_objective   = snake-causal-marginal-v1
collection_semantics = food-quota-v1
recurrent_update     = stored-state-burnin-v1
replay_objective     = success-snapshot-sil-v2
```

旧v2权重的前向反馈和训练概率语义不同，不能静默resume或拿新适配器预览。需要复现
pre-v3结果时，应保留对应源码和加载器。

每次update至少核对以下指标，而不是只看总reward：

- `food_count`、`collection_food_target_met`、`collection_food_shortfall`；
- `collection_stop`、`rollout_decisions_per_env`、`rollout_transitions`；
- `terminal_events`、`truncation_events`、`completed_food_episodes`；
- `recurrent_window_count`、最长learn/burn-in步数、跨reset窗口数；
- replay中的完成成功/失败episode数、活动成功快照数、恢复后成功fragment数、各类
  实际抽样数及成功比例；
- PPO loss、KL、entropy和梯度范数，另行配合独立冻结策略评估。

这些指标只能说明实现按预定语义运行。学习结论必须来自冻结策略、未见seed和多随机
种子的31×20评估；小棋盘、smoke test、训练episode分数或单个成功片段都不能代替。

## 8. 需要做的对照

在启动长训练前，至少固定环境交互tick、评估seed和模型容量，比较：

1. `events`与`fixed`采样，判断food quota是否只是改变批次组成；
2. v3因果边缘目标与独立保留的pre-v3全20维目标，判断梯度方差和学习速度；
3. PPO-only、未分层SIL、无即时快照的成功分层SIL、当前SIL，判断快照、成功保护和
   0.5配额各自的贡献；
4. burn-in/unroll组合，报告吞吐、显存、取食率和通关率，而不只报告loss；
5. 默认异步时序与延迟压力条件，确认改进不是固定轨迹或无扰动场景造成的。

当前方法仍可能不够。尤其是活动快照依赖value bootstrap、GAE不能跨rollout，且成功
分层会改变回放目标分布。如果对照显示无收益，应分别修改这些机制，不能仅继续增加
迭代次数后把失败解释为“训练量不够”。

## 9. 一手依据与采用范围

- Schulman等，PPO：<https://arxiv.org/abs/1707.06347>。采用裁剪概率比和多epoch
  on-policy更新；food quota、因果边缘和循环窗口均为项目适配。
- Schulman等，GAE：<https://arxiv.org/abs/1506.02438>。采用优势递推；可变决策间隔
  的`gamma ** K_i`、事件边界处理和批次长度为项目实现。
- Oh等，SIL论文：<https://proceedings.mlr.press/v80/oh18b.html>；作者代码：
  <https://github.com/junhyukoh/self-imitation-learning>。采用自主经验的正优势目标；
  取食快照、成功判定、保护性淘汰、分层比例与循环序列实现不同于原作。
- Kapturowski等，R2D2：<https://openreview.net/forum?id=r1lyTjAqYX>。只采用存储状态
  与burn-in的设计动机，不采用其value-learning和分布式replay算法。
- Spooner等，FPG：
  <https://proceedings.neurips.cc/paper_files/paper/2021/hash/2ba8698b79439589fdd2b0f7218d8b07-Abstract.html>。
  只把其动作影响结构分析作为因果审计依据，不声称实现FPG。
