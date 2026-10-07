# V-trace长程信用与事件轨迹设计

## 这次改动修正什么

前一版把actor传输长度、V-trace回报长度和循环网络反传长度都压在约128次决策内。
actor即使继续玩同一局，第一段末尾也只能用当时的value作bootstrap；后来吃到食物，
奖励不会回写到前一段。延长采样总量不能补上这条断点。

当前实现把四个概念分开：

1. `fragment`是actor送到learner的有界传输包，通常不超过
   `unroll_length=128`；参数同步、长度上限和取食都可能封包；
2. `credit trace`是learner真正计算V-trace目标的序列。相邻fragment先按流和
   episode拼接，到取食、死亡、time-limit truncation或独立安全上限才结束；
3. `bptt_window`只限制一次保留的自动求导图，默认且最大为128；
4. collection batch由若干条已经闭合的credit trace组成，用于一次optimizer step。

所以，`128`不再同时表示回报边界和反传边界。默认
`credit_trace_max_transitions=2048`，一条取食轨迹可以跨过多个128步fragment；
V-trace递推看完整条轨迹，GRU梯度仍按不超过128步的窗口截断。

旧配置每个游戏tick使用`gamma=0.99`，每次策略决策跨2—5个tick，同时GAE按决策再乘
`lambda=0.95`。按平均3.5个tick估算，128次决策后的传递系数约为
`(0.99^3.5 * 0.95)^128 = 1.6e-5`。当前V-trace实验仍以
`gamma=0.9999`作为待验证的长时域假设，并对每次决策使用实际的`gamma ** K`。
这能减少人为截断，不保证31×20任务必然学会。

## 文献依据和本项目的改法

V-trace递推来自[IMPALA论文](https://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf)。
行为策略概率、目标策略概率、截断重要性比及actor/learner分离沿用论文定义。
[DeepMind参考实现](https://github.com/google-deepmind/scalable_agent/blob/master/vtrace.py)
和[RLax实现](https://github.com/google-deepmind/rlax/blob/main/rlax/_src/vtrace.py)
都把递推得到的target停止梯度；因此“目标递推跨完整trace，网络求导分窗口进行”不要求
把整条trace的计算图留在显存中。两遍计算和窗口间隐藏状态的具体安排是本项目适配，
不是IMPALA论文原样提供的训练循环。

循环状态的stored state与burn-in思路参考
[R2D2](https://openreview.net/pdf/387fb2fcee8f74c53cf707a9856f40c458f33933.pdf)。
这里只采用无梯度burn-in重建隐藏状态的办法，不采用DQN、value rescaling、优先级定义
或R2D2 replay目标。

[LASER](https://proceedings.mlr.press/v119/schmitt20a.html)说明大规模异步回放需要处理
陈旧数据和稳定性问题。当前版本没有实现LASER的trust-region机制，也不把孤立
fragment作为replay样本训练。正式训练CLI不再暴露replay参数，checkpoint sidecar只
保存assembler运行状态。旧实现移入`resnake_gym.experimental`，仅供历史复现和单独
对照；在能以完整credit trace定义回放单位、恢复行为概率及验证陈旧度以前，不做
progress replay实验。

## 动作和时间仍按手柄记录

J8策略在一个chunk的每个槽分别采样五类动作：

```text
neutral / up / down / left / right
```

固定模板把类别展开成`[L,20]`标准手柄报告。H1诊断头只采样一个五分类变量，再把
同一报告保持`L`个槽。模板不读取蛇头、食物、墙或当前方向，不删除危险动作，也不把
反向请求改成安全方向。环境仍然只接收完整手柄报告。

训练数据分开保存：

1. `policy_actions`：J8为`[L]`个五分类变量，H1为一个五分类变量；
2. `requested_reports`：策略提交的完整20维报告；
3. `executed_reports`：经过延迟、覆盖、过期和neutral fallback后，每个游戏tick真正
   执行的20维报告。

V-trace只对实际随机变量计算概率。H1保持的重复槽不能重复乘概率；执行后的报告也
不能反推行为策略概率。每步还保留controller/capture/origin/arrival/execution tick、
`submitted_sequence`和simulation timebase。一次环境step跨`K`个游戏tick，reward和
discount都按这些实际tick汇总。

## fragment怎样拼成credit trace

每个actor连续维护游戏、环境随机流和GRU状态。封存fragment不会reset游戏。取食后
当前决策立即封包，下一次决策从同一episode、同一环境状态和已经推进的hidden继续；
只有死亡和time-limit truncation才reset并开始新的episode。

fragment v5增加连续编号：

- `(run_generation, actor_id, env_id, episode_id)`标识来源；
- `fragment_sequence`标识同一episode内的传输顺序；
- `[decision_start, decision_end)`标识该fragment覆盖的策略决策区间；
- `policy_versions`、joint behavior log probability及完整五类behavior分布按transition
  保存，不再假定一条credit trace只来自一个behavior版本；
- burn-in观测、`burn_h0`、learn观测和`bootstrap_observation`保留原始循环状态语义；
- reward、实际`K`、`gamma ** K`、终止/截断、分数、事件和全部手柄时间戳照常保存。

learner端assembler按`(run_generation, actor_id, env_id)`维护流。它拒绝缺失、重复或
乱序的fragment，并检查：episode、决策区间和分数连续；前一段
`bootstrap_observation`等于后一段首观测；执行tick、下一命令origin tick和提交序号
连续；取食或episode边界以后不能继续拼同一条trace。校验失败会停止训练，不跳过坏包。

credit trace只在以下边界闭合：

- `food`：最后一个fragment包含真实取食事件，游戏继续；
- `terminated`：真实终止，末步discount为0；
- `truncated`：time limit，保留`gamma ** K`，以reset前最终观测bootstrap；
- `safety_cap`：累计transition达到内存安全上限；
- `resume`：恢复checkpoint时，将旧run generation里未闭合的前缀显式封口。

`food`、`truncated`、`safety_cap`和`resume`通常都是可bootstrap边界；是否bootstrap以
最后一步实际discount为准。“完整trace”指边界和末观测已经明确，不等于一条完整游戏。
默认2048只是host内存与异常episode的安全界，不是对游戏最优路径长度的估计。设置
`--credit-trace-max-transitions 128`做对照时仍经过同一个assembler，不另走旧fragment
训练路径。

## 一次learner更新怎样计算

learner在optimizer step前固定一个target policy版本。第一遍使用当前参数、无梯度地
遍历每条完整credit trace：先从第一段的burn-in重建hidden，再得到整条trace每一步的
target log probability、value以及窗口起点hidden。随后把各fragment的reward、
`gamma ** K`和逐transition behavior log probability连起来，做一次覆盖完整trace的
V-trace反向递推。

一条trace可以跨越多次actor参数同步，因此importance ratio按每个transition各自保存的
behavior概率计算。`policy_versions`用于审计版本跨度，不能拿trace首段或末段的版本
代替整条序列。当前target参数在这一批的两遍计算和全部窗口中保持不变，直到最后才
执行optimizer step。IMPALA对固定行为策略给出理论结果；这里把多次同步产生的不同
behavior版本拼入有限轨迹，是工程适配，不把原论文结论直接外推到任意陈旧度。

第二遍从第一遍保存的窗口起点hidden重新前向。每个窗口最长
`bptt_window<=128`，窗口边界detach hidden；policy、value和entropy loss都使用第一遍
算好的完整trace目标。各窗口依次`backward()`并累积梯度，全部窗口完成后只裁剪一次
梯度、只调用一次`optimizer.step()`。

这一区别需要保留：跨窗口的未来奖励能进入较早窗口的V-trace target，但较晚窗口的
GRU计算图不会穿过detach边界反传到较早hidden。前者修复了fragment处的人为回报截断；
后者仍是截断BPTT的近似。

## fresh采集、预算和恢复

采集循环只把已经闭合的credit trace交给learner。达到
`batch_min_transitions`以后，若`batch_food_target`仍未满足，可以继续收完整trace，
直到配额满足或达到`batch_max_transitions`。日志中的
`fresh_food_quota_required_after_minimum`和`fresh_food_quota_binding`都表示“达到最小量
时配额仍未满足”；另用wait trace、transition和logic tick记录后来多收了多少。

主预算按fragment第一次从actor队列被接收时累计的
`cumulative_received_logic_ticks`计算，不按已经完成训练的trace计算。否则较长安全上限
会把更多未闭合交互藏在assembler里，2048组就会比128组多用环境数据。每次更新后应有：

```text
cumulative received
= cumulative trained + pending assembler + ready queue
```

transition和logic tick各自检查一次，误差字段必须为0。接收数据只可能处在三处：本次
或过去已训练、assembler内等待事件的pending前缀、已经闭合但留待下一批的ready trace。
不允许静默丢弃。`fresh_trained_*`表示本次实际训练的完整trace，
`fresh_received_*`表示本次从队列第一次消费的新fragment，二者不要求相等。旧字段
`fresh_food_count`仅作为trained口径兼容别名保留，不得除以received tick。正式
`food_per_10k_logic_ticks`使用`fresh_received_food_count / fresh_received_logic_ticks`。
每个食物另记录一基于队列接收顺序的一维
`fresh_received_food_tick_positions`；这使25万tick分界落在一个采集批次内部时仍能精确
分箱。终局记录也带`receipt_tick_position`，ready trace下一轮才训练时不会重复计数。

checkpoint同时保存learner和credit assembler状态，包括pending、ready及累计received
计数。恢复时先校验assembler格式和安全上限，再把旧generation的pending前缀封成
`resume` trace，放入ready队列；新actor使用新的`run_generation`，不会与旧前缀拼接。
这种恢复保住已经接收的数据及其bootstrap边界，但不会恢复actor进程中的游戏、命令队列
或随机数流，因此仍不声称bitwise续跑。

## 版本契约

当前格式为：

```text
checkpoint                 gamepad-vtrace-v3
J8 objective               snake-dpad5-event-trace-vtrace-v2
H1 objective               snake-held-dpad5-event-trace-vtrace-h1-v2
collection                 fifo-stitched-credit-trace-v3
recurrent state            stored-state-burnin-window-tbptt-v5
fragment                   gamepad-vtrace-fragment-v5
credit trace               gamepad-vtrace-credit-trace-v1
credit assembler           gamepad-vtrace-credit-assembler-v2
```

v2 checkpoint和v4 fragment没有完整的拼接编号、pending状态与received预算语义，直接
拒绝加载，不做猜测迁移。J8与H1仍校验动作变量数、模型形状和目标版本，不能互载。

## 必须看的曲线

除分数、通关率、死亡原因、rho、ESS、KL和actor lag外，长程信用实验至少报告：

- `credit_trace_len_mean/p50/p90/p99/max`、`credit_trace_over_128_count/rate`；
- `credit_trace_cap_hit_count/rate`、`credit_artificial_boundary_count/rate`；
- `segments_per_trace_mean/max`与`bptt_windows_per_update`；
- `unique_behavior_versions_mean/max`（同时保留`mixed_policy_versions_*`别名）；
- `transport_log_weight_mean/min`，观察折扣与截断importance weight沿trace的实际衰减；
- `credit_bootstrap_boundary_count/rate`；
- received、trained、pending、ready的transition/tick计数及两项守恒误差；
- received与trained各自的food、episode和吞吐；主曲线只用received口径；
- episode score P50/P90/P99、终止原因、lifetime，以及
  `entropy_joint_nats`、`entropy_per_action_variable_nats`、
  `entropy_normalized_fraction`和`entropy_beta_h`；
- `pending_credit_bytes`、队列回压和logic ticks/s。

第一项正式信用跨度对照应只改变
`credit_trace_max_transitions=128/2048`，其余棋盘、扰动、J8/H1、seed、received tick预算
和`bptt_window=128`保持一致。fragment replay在这轮固定关闭。H1是因果诊断，不应与
cap变化合并成一个无法归因的实验。

## 仍然没有解决的部分

- 2048是安全上限，不覆盖一局可能需要的全部十万级游戏tick；`safety_cap`仍会引入
  bootstrap边界。
- 完整trace target跨窗口，GRU参数梯度仍只在128步窗口内传播；两者不能写成“完整局
  端到端反传”。
- 不同behavior版本拼接后的稳定范围要靠lag、rho、KL和transport指标验证；当前实现
  不提供超出IMPALA假设的新收敛证明。
- replay训练暂时禁用。要恢复时，样本单位必须是可校验的完整credit trace，而不是从
  中间截出的旧fragment，并需另做陈旧度与采样偏差对照。
- `gamma=.9999`、事件配额、2048安全界和H1/J8仍是待消融设置。
- 当前结果只涉及游戏策略和虚拟手柄，不证明灵巧手、真实HID或联合训练已经成功。
