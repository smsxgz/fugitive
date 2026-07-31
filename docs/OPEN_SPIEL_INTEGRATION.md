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

### 9.1 Manhunt evaluator

Manhunt 猜中后会同时公开路线位置和该位置的具体 Sprint 集合，所以成功观察定义为：

```text
o = (route_position, exact_sprint_cards)
```

精确参考递归为：

```text
V(B) = max_j sum_o mass(B, j, o) / mass(B) * V(B conditioned on j, o)
```

猜错分支价值为 0；没有隐藏 Hideout 时价值为 1。每个 `j` 的成功 outcome 必须满足
`sum_o mass(B,j,o) == hidden_card_history_mass[j]`。memo key 包含全部已知 Hideout、
具体 Sprint、不可用牌和失败猜测。Completion 或 solver-state 预算一旦截断，结果只
返回保守上下界并令 `exact=false`；公开 reveal API 默认最多 10,000 次 Completion，
无限预算只用于手工小状态 reference。

完整对局不能只看 Route support。seed 1 的实际 Manhunt 状态只有 10 条 Route，却有
5 个隐藏位置和 10 张隐藏 Sprint。100,000 次条件 Completion 调用约用 244 ms，枚举
26,564 个正质量 outcome 后价值区间仍为 `[0.0240463, 1]`，因此完整历史精确递归不
适合作为在线策略。

完整对局版本改为独立采样 `uniform_consistent` 完整历史，再精确求解这批粒子定义的
有限 belief 树。观察后直接筛选相容粒子，重复历史仍以重复 particle index 保持经验
权重。`exact_for_empirical_belief=true` 只表示没有触发 solver-state 截断；结果仍有
Monte Carlo 误差，不是底层 belief 的置信区间。同一个 seed 1 状态的 profile 为：

| 粒子 | value | 首猜 | solver states | 时间 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.875000 | 3 | 305 | 40 ms |
| 128 | 0.796875 | 3 | 583 | 60 ms |
| 256 | 0.785156 | 3 | 1,001 | 125 ms |

当前 sampler 已改为一次枚举 Route/Completion 支持，再按具体 Route 复用
`FixedRouteCompletionCounter` 的 memo；公开 API 和抽样顺序不变。它只把重复工作移到
一次批量调用中，仍显式报告粒子数、求解状态和时间。

#### 9.1.1 粒子数收敛检查

`manhunt_convergence` 模式从现有随机合法 coverage rollout 收集每局第一个去重 Manhunt
入口。它不是 L1/L2 的策略加权状态分布，只用于在一组真实可达状态上检查 evaluator。
同一个 checkpoint 和 sampling seed 下，每个粒子档都重置相同 RNG，所以 N 个粒子是
最大档的严格采样前缀。最大档只称 reference，不称 ground truth。

离线检查把经验 belief solver 的 state budget 设为无限；主一致率和 value 差异只统计
两边都 `exact_for_empirical_belief=true` 的配对。程序逐次输出 bounds、首猜、实际
`rng_seed`、solver states 和时间；逐 checkpoint 汇总还保存完整 `information_state`，便于
批量 sampler 优化前后复用完全相同的 belief。逐 checkpoint 汇总 value 均值/标准差、modal 首猜，
全局只汇总与 reference 的配对差异及成本，不把不同状态的原始 value 混成一个均值。

复现命令：

```bash
third_party/open_spiel/build/games/fugitive_belief_experiment \
  --mode manhunt_convergence --manhunt_checkpoints 16 \
  --manhunt_sample_seeds 16 --manhunt_particle_counts 64,128,256,512 \
  --seed_start 0 --max_seeds 1000
```

seed 0 开始检查 21 局后收集到 16 个入口；隐藏位置为 3--8，隐藏 Sprint 为 8--16。
共 1,024 次经验 belief 求解全部 exact，结果为：

| 粒子 | 对 512 首猜一致率 | value 平均绝对差 | value p95 绝对差 | 时间 p50 | 时间 p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 85.5% | 0.0727 | 0.1797 | 2.5 ms | 24.9 ms |
| 128 | 89.8% | 0.0523 | 0.1406 | 4.3 ms | 27.4 ms |
| 256 | 93.0% | 0.0282 | 0.0781 | 6.8 ms | 31.2 ms |
| 512 reference | 100% | 0 | 0 | 12.0 ms | 36.9 ms |

批量版本总墙钟 8.1 s，峰值 RSS 13,076 KiB；优化前同一数据为 626 s，且每一项
首猜、value 和 solver states 都相同。16 个状态中有 15 个的跨 seed 平均 value 从
64 到 512 单调下降，剩余一个始终为 1；这与小样本上先对多个动作取最大值产生的乐观
效应一致，也说明 512 本身尚未给出稳定极限。

批量版本进一步将 reference 推到 16,384（同样 16 个入口、16 条 seed）：

| 比较 | 首猜一致率 | value 平均绝对差 | value p90 | 较大档 p95 时间 |
| --- | ---: | ---: | ---: | ---: |
| 8,192 vs 16,384 | 91.0% | 0.0202 | 0.0456 | 914.6 ms |

因此当前可把 16,384 称为离线 reference，把 8,192 视为可用的近似档；两者都不是
底层 uniform-consistent belief 的真值。64 已被收敛初筛否决，N14 可以在明确报告这项
evaluator 敏感性的前提下开始，但不能把 Rao-Blackwell 回报宣称为实际 L1 对局的无偏值。

### 9.2 L1/L2 配对实验

`fugitive_baseline_experiment` 是独立 C++ runner，不接网页，也不恢复旧 Agent 框架。
Fugitive-L1 固定抽最低号非空牌堆，普通回合只打距离不超过 3 的最大 Hideout、否则
Pass；只要 42 出现在 `LegalActions()` 中就优先选择它，随后用现有
`MinimumSprintAction` 支付最少张数的 Sprint。可达性、Sprint 总值和“42 不能作
Sprint”都由 game engine 保证，runner 不重复实现。opening 不允许 Pass，因此没有
距离 3 内的第二张牌时也使用同一个最少 Sprint 逻辑。

为检验死牌烟幕假说，runner 另有 `--dead_card_sprints K`，K 只能取 0、1、2，含义是
每次普通、非 42 的 Hideout 最多额外打 K 张死牌。死牌只按出牌前的
`card <= previous_hideout` 定义，不把本次前进后刚变死的牌纳入；候选按
`(SprintValue, card)` 排序，先消耗奇数牌以保留价值 2 的偶数燃料，选好集合后再按
数字升序执行原子动作。opening、Pass 和 42 都不使用这条分支。默认 K=0，确保之前的
实验命令和结果不被静默改变；K=1 是本轮最小非零变体，K=2 只作强度敏感性对照。

Marshal-L2 使用 Completion 的 history-weighted 边缘。guard 的完整判定为：

```text
G = 按当前 L2 策略在 1..41 构造的实际猜测集
U = 当前隐藏 Hideout 数

|G| == U              -> 提交 G；成功就会直接揭完全部 U 个位置
存在正概率的 1..29   -> 在 1..29 内重新构造猜测集
低牌全部为零概率      -> 按 --low_exhausted 选择 lift 或 wait
```

`lift` 提交 unrestricted `G`；`wait` 选择牌 1 后 Commit。进入 `wait` 的前提已经证明
1..29 全为零概率，所以这是明确的策略性伪过牌，不是 Marshal 的合法动作中新增 Pass。
输出中的 `guard_fallback` 分别报告 lift/wait 的回合数、涉及游戏数，以及成功 lift 导致
Manhunt 失效的游戏数。普通 argmax 只接受正概率牌，不能再因 `-1` 初值意外选中零概率牌。
`normal_belief.unrestricted_argmax_ge_30_turns` 直接报告阈值路径出现次数；
`guard_restriction` 只在第 2 条重建后的实际猜测集与 unrestricted 不同时计数。因此它
不会把“执行了 guard 判断但动作相同”冒充成策略差异。

默认 L2 会加入所有概率为 1 的 Hideout，再加入一个正概率不确定 argmax。seed 0--999
中 guard/noguard 都是 1,000 次 Marshal 胜、1 局到 42；F-L1 暴露的确定项仍让路线很快
被直接揭完，所以默认模式无法检验 30 阈值。

`--guess_mode argmax_only` 提供明确标记的 single-guess 诊断，不冒充默认 L2。同一批
seed 的结果为：

| 策略 | Marshal | Fugitive | timeout | 到 42 | Manhunt |
| --- | ---: | ---: | ---: | ---: | ---: |
| guard (`lift`) | 983 | 17 | 0 | 121 | 121（Marshal 胜 104） |
| noguard | 983 | 17 | 0 | 121 | 121（Marshal 胜 104） |

这一批对局共使用 1,484 张 Sprint，其中 529 张来自强制开局，新增的 955 张全部来自
L1 打 42。原来的 21 个 timeout 已消失。扩大到 10,000 seed 后，`lift` 只触发 1 回合
且猜错，guard/noguard 终局分布仍同为 `9,774 M / 226 F / 0 timeout`；改用 `wait` 也只
触发 1 回合，并使对应一局的 Manhunt 结果由 Marshal 胜变为 Fugitive 胜。两种配置都
有 1,265 局到 42。

这说明 `--dead_card_sprints 0` 的旧 L1 matchup 本身仍无法测 N11：它虽然能制造最终
Sprint 和更多 Manhunt，普通路线位置却没有 Sprint，最高边缘从不进入 30 以上。后续
实验必须先改变 Fugitive 侧的公开 Sprint 形状，不能只拆 Marshal 的提交规则。

上面是 `--dead_card_sprints 0` 的旧 L1 对照。加入普通回合死牌烟幕后，同一批
seed 0--999、single-guess L2、64 个 Manhunt 粒子的结果为：

| 每次最多死牌 K | guard M/F/TO | noguard M/F/TO | guard 高牌 argmax | guard 真正限制 | guard 到 42 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 983 / 17 / 0 | 983 / 17 / 0 | 0 | 0 | 121 |
| 1 | 971 / 29 / 0 | 973 / 27 / 0 | 104 | 25 | 34 |
| 2 | 972 / 28 / 0 | 971 / 29 / 0 | 112 | 27 | 34 |

K=1 的 guard 一侧实际打出 2,475 张普通死牌 Sprint；K=2 为 2,913 张。两者都立即让
最高边缘进入 30 以上，并让第 2 条 guard 真正改变动作，验证了死牌烟幕假说的机制。
代价也很明显：到 42 从 121 局降到约 34 局，死牌仍是
最终冲刺燃料，不能称为免费资源。

K=1、`low_exhausted=lift` 扩大到 10,000 seed 后，guard 为
`9743 M / 257 F / 0 TO`，noguard 为 `9736 M / 264 F / 0 TO`。guard 侧共有 953 个
高牌 argmax 节点，第 2 条真正改动 216 次；47 个配对 seed 改变胜者，其中 guard 独赢
27 局、noguard 独赢 20 局。这个净差只有 7 局，而且 1,000 seed 的方向相反，足以证明
N11 已进入决策路径，不足以证明稳定的胜率优势。

`low_exhausted=wait` 不适合作为主结果。K=1 在 `max_rounds=50/100/200` 时始终是同一
7 局达到 horizon，胜负分布也完全不变；只有 wait 次数从 297 增至 647、1,347。
这说明它们是低牌概率耗尽后的策略循环，不是墙钟性能超时，也不能作为 guard 收益。

### 9.3 Marshal 强制赌博消融

`--guess_mode` 把原来的布尔开关拆成三种完整策略。设 `C` 是候选范围内全部概率为 1
的牌，`A` 是全部正概率牌的 argmax，`A?` 是非确定正概率牌的 argmax：

```text
argmax_only          -> 提交 A
certain_only         -> C 非空时提交全部 C；否则提交 A
certain_plus_argmax  -> 先提交 C；若 |C| < U，再追加 A?
```

并列概率统一选择最小牌号。`certain_only` 的 fallback 是必要定义：Marshal 没有 Pass，
没有确定项时不能提交空集合。默认仍是 `certain_plus_argmax`。旧参数
`--add_certain_guesses 0|1` 分别映射到 `argmax_only / certain_plus_argmax`，只用于复现
旧命令；同一次调用不能再同时指定 `--guess_mode`。

每种模式先在 1..41 构造真实猜测集，再走同一套 guard。若实际 `|G| == U`，成功后就会
覆盖全部隐藏位置，因此直接绕过 30 限制；否则才在 1..29 重建或进入 lift/wait。
`guess_mode_diagnostics` 报告：

- `forced_gamble.turns/losses/certain_cards_lost`：实际提交至少一张确定牌并追加非确定
  argmax，以及失败时丢掉的确定揭示；
- `banked_certain.turns/cards`：certain-only 在尚未覆盖全部隐藏位置时只拿确定进展；
- `cover_all.attempts/wins`：实际猜测数等于隐藏位置数，以及其中成功终局的次数。

旧 K=0、300 seed 的复跑精确得到 review 的插桩数字：249 次强制赌博、141 次失败
（56.6%），并进一步量出失败时共丢掉 218 张确定揭示。这证明局部成本真实存在，但
不能单独推出整局策略较差。

正式消融固定 K=1、seed 0--9999、64 个 Manhunt 粒子和 `low_exhausted=lift`。三种模式
各自仍输出 guard/noguard 配对；强制赌博的主比较使用 noguard，避免把 30 阈值混入：

| guess mode | guard M/F | noguard M/F | noguard 平均回合 | 关键诊断（noguard） |
| --- | ---: | ---: | ---: | --- |
| `argmax_only` | 9743 / 257 | 9736 / 264 | 7.00 | cover-all 成功 9,656 |
| `certain_only` | 9927 / 73 | 9930 / 70 | 6.33 | bank 9,661 回合 / 10,853 张确定牌 |
| `certain_plus_argmax` | 9997 / 3 | 9998 / 2 | 5.01 | 赌博 8,872 次，失败 2,930 次，丢 3,504 张确定牌 |

排序在 1,000 和 10,000 seed 中一致。plus 虽然约三分之一的赌博失败，却更快制造
cover-all 终局：noguard 的 16,493 次 cover-all 尝试中 9,998 次成功，最终只输 2 局。
因此在当前确定性 L1-smoke 对手上，“强制赌博”有明确的净收益；它的风险应当被记录，
但没有理由从默认 L2 中删除。这个结论只属于该 matchup，不是最优性证明。

## 10. 更新 OpenSpiel 版本

上游升级不是简单改 tag。至少要按以下顺序处理：

1. 更新 `integration/OPEN_SPIEL_VERSION` 中的 tag 和完整 commit；
2. 为新版本重新生成并审查 CMake 补丁；
3. 核对上游 mandatory dependency 版本；
4. 从干净 `third_party/open_spiel` 运行 setup 和 build；
5. 运行 C++ game test、OpenSpiel random simulation/serialization 测试和网页测试；
6. 检查 `GameType`、State 虚函数和 Python binding 是否有兼容性变化。

不要在未记录 commit 的 OpenSpiel `main` 上发布实验结果。
