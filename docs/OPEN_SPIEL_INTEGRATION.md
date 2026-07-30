# OpenSpiel 集成说明

本文说明 Fugitive 如何作为原生 C++ game 接入 OpenSpiel，以及算法实验需要遵守的
接口和信息边界。桌游规则本身以
[`rules/CANONICAL_RULES.md`](rules/CANONICAL_RULES.md) 为依据。

## 1. 版本与源码布局

集成固定在以下上游版本：

```text
OpenSpiel version: v2.0.1
OpenSpiel commit:  112b77704631fc2ce7ad8e4581f6ca09798ce15a
Repository:        https://github.com/google-deepmind/open_spiel.git
```

固定值记录在 [`../integration/OPEN_SPIEL_VERSION`](../integration/OPEN_SPIEL_VERSION)。
仓库内的职责分工是：

```text
cpp/fugitive/                         项目维护的权威 C++ 源码
third_party/open_spiel/               setup 生成的上游工作树，不提交
third_party/open_spiel/open_spiel/
  games/fugitive                      指向 ../../../../cpp/fugitive 的相对软链接
integration/open_spiel-v2.0.1.patch   将 game 和测试加入上游 CMake 的最小补丁
src/fugitive/web/                     网页协议适配，不实现第二套规则
```

旧 Python 引擎、Agent、PSRO 和实验代码已从当前工作树删除，需要时从 Git 历史
查阅，不再保留仓库内归档副本。

软链接解决的是源码所有权问题：修改发生在 `cpp/fugitive`，不会埋在可删除的第三方
checkout 中。软链接本身不足以完成集成，CMake 还必须编译相应 `.cc`，最终的
`pyspiel` 扩展也必须链接包含 `REGISTER_SPIEL_GAME` 静态注册代码的目标。

因此，安装预编译的官方 pip wheel 后再放入几个 Python 文件，并不能让 wheel 自动
获得这个 C++ game 的静态注册。OpenSpiel 也支持把游戏作为外部 library 注册并链接，
但需要自行管理链接、注册入口和 Python 扩展装载，复杂度更高；这不是本项目选择的
方案。我们采用官方 developer guide 中直接、可测试的路径：把 game 放进
`open_spiel/games` 的源码视图、加入构建清单并重编 `pyspiel`。OpenSpiel 的纯
Python game 注册机制同样存在，但不符合本项目使用 C++ 重新实现引擎的目标。

## 2. 环境、构建与运行

一次性准备环境和上游源码：

```bash
bash scripts/setup_openspiel.sh
```

脚本会：

1. 按 [`../environment.yml`](../environment.yml) 创建或复用 Conda 环境
   `openspiel`；
2. 安装 [`../requirements.txt`](../requirements.txt) 中的 Python 依赖；
3. checkout 固定的 OpenSpiel commit 和构建所需的上游依赖；
4. 创建 Fugitive 源码软链接；
5. 幂等地应用 CMake 补丁。

编译 C++ game、`pyspiel` 并运行 C++/Python 测试：

```bash
bash scripts/build_openspiel.sh
```

启动网页：

```bash
bash scripts/run_web.sh --host 127.0.0.1 --port 8000
```

脚本显式设置 `PYTHONPATH`，使 Python 同时找到当前 `src` 包、OpenSpiel Python
源码以及本地构建出的 `pyspiel` 扩展。

## 3. OpenSpiel game 契约

注册名为 `fugitive`，默认可这样加载：

```python
import pyspiel

game = pyspiel.load_game("fugitive")
state = game.new_initial_state()
```

也可以显式传入研究 horizon：

```python
game = pyspiel.load_game("fugitive", {"max_rounds": 50})
# 等价形式：pyspiel.load_game("fugitive(max_rounds=50)")
```

注册的 `GameType` 为：

| 字段 | 值 |
| --- | --- |
| players | 2；player 0 是 Fugitive，player 1 是 Marshal |
| dynamics | sequential |
| chance mode | explicit stochastic |
| information | imperfect information |
| utility | zero sum；Fugitive 胜为 `[1,-1]` |
| reward model | terminal |
| information state | string |
| observation | string |
| tensors | 当前不提供 |

核心尺寸接口当前返回：`NumDistinctActions() = 47`、`NumPlayers() = 2`、
`MaxChanceOutcomes() = 42`（保守上界）、`MaxChanceNodesInHistory() = 38`（五次
setup 加其余 33 张可抽牌），以及保守的
`MaxGameLength() = 100 * max_rounds + 100`。效用范围是 `[-1,1]`，
`UtilitySum() = 0`。

`InformationStateString(player)` 和 `ObservationString(player)` 返回结构化 JSON
字符串，并按玩家视角隐藏对手手牌、未揭示 Hideout 和 Sprint 身份。算法只能消费
该玩家的信息状态或 observation；`ToString()` 是调试用全状态，不得当作策略输入。

### 牌与初始状态

牌面是整数 `1..42`，共 42 张；此外用 `0` 表示公开路线起点：

| 卡牌 | 初始位置 |
| --- | --- |
| `0` | 公开、已揭示的路线起点；不进入手牌，也不参与抽牌 |
| `1,2,3,42` | Fugitive 的固定初始手牌 |
| `4..14` | pile 0，共 11 张；setup 随机抽 3 张给 Fugitive |
| `15..28` | pile 1，共 14 张；setup 随机抽 2 张给 Fugitive |
| `29..41` | pile 2，共 13 张；setup 不抽 |

因此可抽牌区是 `4..41` 共 38 张，五次 setup chance outcome 后还剩 33 张；
Fugitive 开局时有 9 张手牌，Marshal 空手。奇数卡 Sprint value 为 1，偶数卡为
2；固定起点 0 不能作为 Sprint。

### Phase 状态机

C++ 状态暴露以下内部 phase。括号中是 `CurrentPlayer()`：

| Phase | 行为与下一步 |
| --- | --- |
| `setup_chance` (chance) | 依次完成 pile 0 的三次和 pile 1 的两次 setup 抽牌，然后进入 `fugitive_hideout` |
| `fugitive_draw_choice` (Fugitive) | 普通回合选择非空牌堆，然后进入 `fugitive_draw_chance`；全空则跳到 `fugitive_hideout` |
| `fugitive_draw_chance` (chance) | 从所选牌堆均匀发一张给 Fugitive，然后进入 `fugitive_hideout` |
| `fugitive_hideout` (Fugitive) | 开局必须选 Hideout；普通回合可 `Pass`，或选 Hideout 后进入 `fugitive_sprint` |
| `fugitive_sprint` (Fugitive) | 逐张选择 Sprint 并 `Commit`；开局完成第一段后回到 `fugitive_hideout`，完成第二段后开始 Marshal 回合 |
| `marshal_draw_choice` (Marshal) | 选择非空牌堆，然后进入 `marshal_draw_chance`；全空则跳到 `marshal_guess` |
| `marshal_draw_chance` (chance) | 均匀发牌；round 1 的 Marshal 要依次选堆并抽两次，其余 round 一次，然后进入 `marshal_guess` |
| `marshal_guess` (Marshal) | 逐个选择普通猜测并 `Commit`；结算胜负，否则完成本 round |
| `manhunt_guess` (Marshal) | 42 触发 Manhunt 时逐次执行一个单猜；猜中继续，第一次猜错结束 |
| `terminal` (terminal) | 不再有合法动作，返回终局收益 |

一个 **round** 是一对双方 turn：Fugitive 先行动，Marshal 随后行动。Round 1 的
Fugitive 不抽牌并建立两个 Hideout，随后 Marshal 抽两张再猜；round 2 起双方各抽
一张再行动。所有牌堆为空时只跳过抽牌节点，不跳过 Play/Pass 或 Guess。

若 Fugitive 提交 42，流程会从 `fugitive_sprint` 直接进入 `terminal` 或
`manhunt_guess`，不会先执行常规 Marshal draw/guess。

### 玩家视角与隐私

信息字符串和 observation 都包含 JSON schema、角色、round、公开 phase、牌堆剩余
张数、公开历史以及当前玩家自己的私有信息。关键字段的可见性如下：

| 信息 | Fugitive 视角 | Marshal 视角 |
| --- | --- | --- |
| 自己手牌 | 完整 | 完整 |
| 对手手牌 | 不提供 | 不提供 |
| 自己抽到的卡 | `draw_history.card` 可见 | `draw_history.card` 可见 |
| 对手抽到的卡 | `null`；抽牌人、牌堆和 round 仍公开 | `null`；抽牌人、牌堆和 round 仍公开 |
| 未揭示 Hideout | 完整可见 | `null`，但 42 按规则公开 |
| 每段 Sprint 张数 | 可见 | 可见 |
| 未揭示 Sprint 身份 | 完整可见 | `null` |
| 已揭示 Hideout/Sprint | 可见 | 可见；42 下的 Sprint 仍不公开 |
| 猜测数字与结果 | 可见 | 可见 |
| 正在组装的复合动作 | 仅 Fugitive 自己的 Hideout/Sprint | 仅 Marshal 自己的 Guess |

复合动作尚未提交时，两方都能看到 `pending_action_count`，但只有行动方能看到
`pending_action` 中的具体牌面或猜测数字。这个计数让每个原子动作后的 observation
与 information state 满足 OpenSpiel 对所有玩家的 action-observation-history 一致性；
非行动方在提交前没有决策节点，提交后 Sprint 张数或猜测集合本来就是公开信息。

`InformationStateString` 使用 `fugitive.information_state` schema，并包含完整可见
历史以满足 perfect recall；`ObservationString` 使用 `fugitive.observation` schema。
当前两者都保留各自视角可见的历史，主要契约差异是语义和 schema，不应依赖调试
`ToString()` 或原始 C++ 成员绕过隐私。全知视角只允许网页在展示层显式生成，不能
作为学习策略的输入。

## 4. 显式随机节点

牌堆不保存一份预先洗好的私有排列，也不调用游戏内 RNG。setup 和每次摸牌都进入
OpenSpiel chance node，`ChanceOutcomes()` 对当前所选牌堆中的剩余卡牌给出均匀
概率：

```text
setup: pile 0 抽 3 张 -> pile 1 抽 2 张
normal draw: player 选择 pile -> chance node 决定具体卡牌
```

这样 rollout、CFR 变体和测试工具能够统一遍历或采样随机事件；seed 属于调用算法或
网页适配器，而不是 C++ 状态内部的第二套随机系统。

## 5. 动作微步骤协议

OpenSpiel 的每个节点接受一个整数 action。Fugitive 的 Sprint payment 和 Marshal
的多数字猜测本来都是组合动作；若为每个组合分配一个 action，动作数会随手牌或候选
集合指数增长。实现改用有限的微步骤协议，总动作数固定为 47：

| Action ID | 含义 |
| --- | --- |
| `0` | Fugitive `Pass` |
| `1..42` | 当前 phase 下的卡牌、Hideout、Sprint 或猜测数字 |
| `43..45` | 选择牌堆 `0..2` |
| `46` | `Commit` 当前复合选择 |

具体决策序列如下：

```text
Fugitive normal turn:
  choose pile -> chance card ->
  Pass
  or choose Hideout -> choose zero or more Sprint cards -> Commit

Fugitive opening:
  choose Hideout -> choose zero or more Sprint cards -> Commit
  choose Hideout -> choose zero or more Sprint cards -> Commit

Marshal turn:
  choose pile -> chance card
  [first Marshal turn repeats the draw once]
  choose one or more guess numbers -> Commit

Manhunt:
  choose one guess number -> Commit -> resolve before the next guess
```

Sprint 卡和普通猜测数字必须按数值严格递增选择。这个规范化约束只去除同一集合的排列
重复，不改变可表达的集合。`LegalActions()` 还会过滤无法完成当前付款、或会让开局
第二个 Hideout 无法完成的选择，避免算法进入无合法动作的中间状态。

网页仍接收原来的宏动作，适配器负责在克隆状态上验证后原子地展开成 OpenSpiel
actions。典型映射为：

```text
{"type":"draw", "pile":1}                              -> [44] + chance outcome
{"type":"pass"}                                        -> [0]
{"type":"fugitive_action", "hideout":10,
 "sprint_cards":[2,5]}                                  -> [10,2,5,46]
{"type":"guess", "numbers":[7,10]}                   -> [7,10,46]
```

适配器对 Sprint/guess 数字排序并逐步检查 `LegalActions()`；任一步失败都丢弃克隆，
不会把半个宏动作写入会话。网页不会在 chance 或复合动作的内部 phase 暂停给用户，
因此保留了原界面的一个 Play/Guess 操作体验，同时网页和研究算法共享同一个 C++
规则状态机。

## 6. 有限研究 variant

原规则没有 horizon，且牌堆耗尽后允许继续 Pass 或失败猜测，因此存在无限对局路径。
OpenSpiel game 必须提供真实的有限 `MaxGameLength()`，许多遍历和学习算法也依赖
有限终止保证。

当前实现引入参数 `max_rounds`，默认值为 50。它主要防止牌堆耗尽后 Fugitive 持续
Pass、Marshal 持续猜错所形成的无限循环，不代表预期对局长度。每个 round 包含一个
Fugitive 回合和随后的 Marshal 回合；完成第 `max_rounds` 个 Marshal 回合仍无赢家
时，状态以 `reason = "max_rounds"` 结束并返回 `[0,0]`。正常逃脱、Manhunt 和全部
Hideout 揭示的胜负优先于 horizon。

参数范围是 `1..21474835`；上限确保 `MaxGameLength()` 的保守动作数上界不会溢出
OpenSpiel 使用的 32 位整数。实际研究应使用远小于该技术上限的值。

这是 **OpenSpiel 研究 variant** 的工程规则，不是对原桌游规则的重新解释。报告实验
结果时必须记录 `max_rounds`；旧 Python 归档中的“无平局”结果不能与该 variant
直接混用。

终局原因是稳定的机器可读字符串：

| `reason` | winner | returns |
| --- | --- | --- |
| `escape_no_manhunt` | Fugitive | `[1,-1]` |
| `manhunt_failed` | Fugitive | `[1,-1]` |
| `marshal_uncovered_route` | Marshal | `[-1,1]` |
| `manhunt_success` | Marshal | `[-1,1]` |
| `max_rounds` | Draw | `[0,0]` |

## 7. 状态序列化

该 game 使用 explicit chance nodes，因此可直接使用 OpenSpiel 基类按完整 action
history 提供的 `State::Serialize()` / `Game::DeserializeState()`。chance outcome
也在 action history 中，反序列化会确定性重放同一个隐藏世界：

```python
payload = pyspiel.serialize_game_and_state(game, state)
game_copy, state_copy = pyspiel.deserialize_game_and_state(payload)
```

网页 trace/export 应记录 OpenSpiel game string 和序列化状态或动作历史，不再维护
旧 Python 引擎的 replay manifest 格式。

## 8. 算法选择

迁移到 OpenSpiel 的目的，是使用其经过维护的 game API、policy 表示和算法实现，而
不是在本仓库复制 chance sampling、状态遍历、CFR、best response 或 PSRO。

当前优先使用 **outcome-sampling MCCFR**，并优先调用 pybind 暴露的 C++
`pyspiel.OutcomeSamplingMCCFRSolver`。它每次为双方各采样一条终局轨迹，不要求
information-state tensor，也不会预先展开全树。它仍是 tabular 方法：每个访问过的
information state 都会保存 regret 和平均策略，所以内存会随新信息集持续增长。

本次容器中的受限冒烟数据如下，只用于说明数量级，不是跨机器 benchmark：

| 配置 | 结果 |
| --- | --- |
| C++ outcome sampling，`max_rounds=50`，100 iterations + 20 次采样评估 | 完成；约 10.1 秒，进程峰值 RSS 约 538 MiB |
| C++ outcome sampling，`max_rounds=50`，1000 iterations，不评估 | 完成；约 86.8 秒，进程峰值 RSS 约 4.70 GiB |
| C++ external sampling，`max_rounds=1`，1 iteration | 30 秒超时，未完成 |
| C++ CFR/CFR+ 构造，`max_rounds=1` | 各自 15 秒超时，未完成 |
| uniform policy NashConv，`max_rounds=1` | 15 秒超时，未完成 |

这些数据说明当前主要瓶颈不是单条对局长度，而是大量 perfect-recall 信息集与复合动作
微步骤产生的树宽。扩大 outcome-sampling 预算前必须监控 RSS，并考虑更紧凑的信息
状态键、动作抽象或函数逼近，而不能机械增加迭代数。

其他算法的适用边界：

- external-sampling MCCFR 在接口上兼容，但会在更新玩家的节点展开全部动作。Fugitive
  的 Hideout/Sprint 微动作使这个分支树很大；本机实测即使 `max_rounds=1`，C++
  solver 的一次迭代在 30 秒内也未完成，因此当前不作为训练入口；
- CFR、CFR+、Discounted CFR、sequence-form LP、exact best response 和
  exploitability/NashConv 需要完整遍历或物化博弈树。本机 `max_rounds=1` 下，C++
  CFR/CFR+ solver 构造和 uniform policy 的 NashConv 都在 15 秒内未完成。它们只适合
  未来额外提供的小牌组、固定发牌或动作受限测试 variant；
- PSRO 是人口训练框架，不是无需配置即可运行的单一 solver。OpenSpiel 的 exact
  best-response oracle 同样遍历全树；RL oracle 依赖可训练策略和 tensor 环境。在有
  可承受的近似 response oracle 前，不应宣称 PSRO 已可用于完整 Fugitive；
- Online Outcome Sampling (OOS) 的思路适合按当前信息集限时搜索，但 v2.0.1 的实现
  是未暴露到 `pyspiel` 的 C++ API；接入它需要额外构建入口和独立验证，可作为后续
  方向；
- 普通 MCTS 使用真实 C++ `State`，会把对手私有牌和隐藏路线泄露给搜索，只能作为
  全知调试基线，不能报告为合法的玩家策略。

仓库中的最小训练入口直接调用 OpenSpiel 的
`pyspiel.OutcomeSamplingMCCFRSolver`：

```bash
PYTHONPATH=src:third_party/open_spiel:third_party/open_spiel/build/python \
  conda run -n openspiel python examples/train_outcome_sampling.py \
  --iterations 100 --evaluation-games 20 --max-rounds 50 --seed 0
```

这里的 solver 和平均 policy 来自 OpenSpiel；本地代码只负责参数、训练循环、按该
policy 与显式 chance 概率采样，以及 JSON 汇总。`mean_returns` 是有限场采样自博弈
的经验均值，不是 exploitability，也不能证明收敛到完整游戏的均衡。

需要额外工作才能使用的方向：

- IS-MCTS 需要正确实现 `ResampleFromInfostate`。Marshal 侧已有按 Completion 质量的
  可重放采样器，但 Fugitive 侧尚未实现，因此 C++ game 仍不覆盖该双方接口；当前调用
  `ISMCTSBot.step()` 仍会得到 `ResampleFromInfostate() not implemented`；
- OpenSpiel AlphaZero 示例面向完全信息设置，并依赖 tensor observation。当前 game
  只提供字符串 observation，且隐藏信息下还需要明确定义合法的训练信息边界，因此
  不能宣称可直接公平应用；
- Deep CFR、NFSP 等神经算法通常需要稳定 tensor 编码。应先补齐并测试 observation /
  information-state tensors，再接入训练代码。当前用 OpenSpiel `rl_environment`
  加载 Fugitive 会直接得到 `observation_tensor not supported`。

当前已直接支持的 OpenSpiel 核心接口包括 `Clone`、`LegalActions`、explicit
`ChanceOutcomes`、终局 `Returns`、information-state/observation strings，以及基于
action history 的序列化。当前未实现 `ResampleFromInfostate`、observation tensor、
information-state tensor 或专用神经网络编码；依赖这些接口的算法必须先补实现，
不能静默改用全知状态。

无论选择哪种算法，实验都应固定 OpenSpiel commit、`max_rounds`、算法 seed 和主要
预算参数，并通过玩家视角的 information state 检查无私有信息泄漏。

## 9. Marshal Route / Completion DP

`cpp/fugitive/belief.{h,cc}` 实现了两层 Marshal belief 计数。生产入口
`BuildMarshalBeliefInput` 只接受 `InformationStateString(kMarshalPlayer)`；JSON 只在
边界解析一次，内部只使用小数组、mask 和紧凑 memo key，不能访问全知
`FugitiveState` 的隐藏 getter。

Route 层枚举满足以下约束的具体 Hideout 路线：

- Hideout 严格递增，且满足公开 Sprint 张数或已揭示 Sprint value；
- 排除 Marshal 当前手牌与所有已揭示 Sprint；
- 失败单猜和多猜只约束猜测发生时已经存在的路线前缀；
- Hideout 与已揭示 Sprint 必须能占用其打出 round 之前的 Fugitive 抽牌槽；
- 42 只允许作为 Manhunt 状态中的最后一个 Hideout，永不作为 Sprint。

它输出 `route_support_upper_bound` 和未揭示 Hideout 的 route-support 计数。后者是在
Route 支持集均匀计数下的精确频率，但没有按 Completion 历史质量加权，只用于结构
诊断，不能作为 Marshal 的信念概率。

Completion 层对每条 Route 做进一步计数。候选 Sprint 分成 8 个桶：固定牌 1/3、
固定牌 2，以及三个牌堆各自的奇数/偶数牌。对路线位置 `i`：

```text
min_even = max(0, gap - 3 - sprint_count)
identity_ways = product C(remaining_bucket, selected_bucket)
slot_ways = product P(eligible_slots - used_slots, required_cards)
```

所有 Route 牌、Marshal 手牌和已揭示 Sprint 会先从桶中移除，避免同一张牌重复使用。
隐藏 Sprint 必须满足 `min_even`，来自牌堆的 Hideout/Sprint 必须被分配到打出 round
之前的有序 F 抽牌槽；到路线末尾，再用 `P(remaining_cards, unused_slots)` 补齐仍留在
F 手牌中的隐藏抽牌。固定牌 1/2/3 不占抽牌槽，42 不进入 Sprint 桶。

一次计数单位包括一条具体 Hideout Route、每个隐藏 Sprint 的具体集合，以及所有
Fugitive 隐藏抽牌事件的具体牌与顺序。不重复计算手牌顺序、未抽牌堆排列或公开的
牌堆选择。给定同一个 information state，这些具体 F 抽牌序列的 chance 分母相同，
所以组合质量与 chance 质量成比例；但它没有乘 Fugitive 的动作策略概率，因此明确
命名为 `uniform_consistent_history_mass`，不是 posterior。运行时用 `long double`
累计，范围足够当前完整牌组实验，但大质量不是逐位精确整数。

每条具体 Route 得到 Completion mass 时，同一次遍历还会把该质量累加到路线上的每张
隐藏 Hideout，得到 `UniformConsistentHiddenHideoutProbability(card)`。这才是当前
uniform-consistent 模型下的 history-weighted 单牌边缘；它仍然不是加入 Fugitive
策略后的 posterior，也不能把多个单牌概率相乘来代替联合猜测概率。

Completion memo 的状态只有：

```text
(route_position, remaining[8], used_draw_slots[3])
```

8 个 `remaining` 和 3 个 `used_draw_slots` 各占 4 bit；构造根状态时显式检查每个字段
小于 16，`fugitive_draw_rounds` 也受同一限制，避免未来改牌组后发生静默 key 碰撞。
这些小计数无损打包到一个 `uint64_t`。不同具体 Route 若每个位置的牌桶及
`min_even` 相同，就共享同一个 Completion 结果；同一桶内牌身份对组合计数是对称的，
但总质量仍按每条具体 Route 分别累加。

Sampler 先在 Route 枚举中按每条路线的 Completion mass 做加权蓄水池抽样，再沿选中
路线的 Completion memo 回溯 Sprint 桶数；桶内具体身份、deadline 前的有序抽牌槽和
剩余隐藏抽牌分别做无拒绝抽样。`ReplayMarshalHistory` 随后只使用公开时间线和这份样本
从初始状态重建对局。每个动作都必须属于 engine 的 `LegalActions()`，最终 Marshal
information string 必须逐字节等于目标字符串；非法样本会立即失败，不会丢弃重采。
当前也支持 Marshal 正在逐个组装猜测数字的原子化中间状态。

正式测试只保留手工可穷举计数、replay 和采样统计三类集中保护。作为一次性独立审计，
另用不共享 DP 代码的临时暴力枚举器对拍 4,144 个小状态，mass 和可补全路线数均为
0 失配；真实对局 oracle 另检查 1,320 个状态的真实 Route/Sprint 可补全性，0 失败。
该临时枚举器会复制约 445 行规则逻辑，因此不作为仓库内常驻的第二套实现。

固定回归使用完整 42 张牌和 `max_rounds=50`，在 Marshal 猜测宏动作边界收集
Early/Middle/Late 各 32 个去重信息状态。覆盖器只负责产生状态，不是策略模型。
2026-07-30 的固定 seed 0 复跑如下：

| Completion 时间 | Early | Middle | Late |
| --- | ---: | ---: | ---: |
| p50 | 10 us | 3.65 ms | 0.94 ms |
| p95 | 103 us | 44.52 ms | 147.78 ms |
| max | 114 us | 68.72 ms | 175.75 ms |

Route 层仍为微秒级。实验扫描 36 个 seed、完成 96/96 样本，包含 sampler/replay 后的
wall time 为 1.64 s，峰值 RSS 11,276 KiB。最大 Route 支持为 9,369；47/96 个状态有
Route 被 Completion 排除，单个状态最多排除 1,614 条。96/96 个状态各抽取一个完整
隐藏历史并成功重放。Sampler 的 Late p95/max 为 147.64/174.74 ms；engine replay
全局最大 0.46 ms。复现命令：

```bash
third_party/open_spiel/build/games/fugitive_belief_experiment \
  --samples_per_bucket 32 --seed_start 0 --max_seeds 1000
```

固定 Early/Middle/Late 标签不能证明深局覆盖，因此另有全 Marshal 边界 sweep：

```bash
third_party/open_spiel/build/games/fugitive_belief_experiment \
  --mode sweep --seed_start 0 --max_seeds 40
```

两种模式默认都使用 `--replay_samples 1`；只测 Route/Completion 性能而不做构造性验证
时可设为 0。

该次 sweep 覆盖 644 个去重状态，其中普通 Guess 且 Route support 至少 1,000 的状态
89 个、隐藏 Sprint 至少 10 张的状态 149 个、两者交集 55 个。普通 Guess 的
`route_length >= 10` 有 11 个，最深为 12；该子集 Completion 最大 15.31 ms。全体
Completion p99/max 为 103.66/176.21 ms，Sampler p99/max 为 104.01/175.38 ms，Replay
最大 0.62 ms，峰值 RSS 13,048 KiB。644/644 个样本全部合法重放。

因此当前结论限定为：在已观测的高 support、高隐藏 Sprint 以及深至 12 的状态上，
Route、Completion 和 Marshal sampler 都是 **go**。这不是对所有策略分布或路线
长度 13-14 的保证。Replay 证明每个已采样历史具有构造性合法性，但不构成所有正质量
分支的形式化完备证明。

当前 sampler 只解决 Marshal 视角。OpenSpiel `ResampleFromInfostate` 还要求 Fugitive
视角保持其私有历史并重采 Marshal 隐藏抽牌；在该独立采样器完成前，game 不覆盖此
接口，也不接 IS-MCTS。

## 10. 更新 OpenSpiel 版本

上游升级不是简单改 tag。至少要按以下顺序处理：

1. 更新 `integration/OPEN_SPIEL_VERSION` 中的 tag 和完整 commit；
2. 为新版本重新生成并审查 CMake 补丁；
3. 核对上游 mandatory dependency 版本；
4. 从干净 `third_party/open_spiel` 运行 setup 和 build；
5. 运行 C++ game test、OpenSpiel random simulation/serialization 测试和网页测试；
6. 检查 `GameType`、State 虚函数和 Python binding 是否有兼容性变化。

不要在未记录 commit 的 OpenSpiel `main` 上发布实验结果。
