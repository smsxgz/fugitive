# Fugitive Strategy Lab

这是一个针对双人桌游 *Fugitive* 的完整游戏规则引擎、随机策略基线、隐藏信息推断实验框架和本地交互界面。

项目的重点不是把“随机”写成从扁平合法动作表中均匀抽样，而是建立一组可以逐层比较的教学算法：

```text
分层合法随机
    -> 路径支持与路径计数
    -> 在线 bootstrap particle filter
    -> observation-local constructive sampling
    -> self-normalized importance sampling
    -> exact Sprint reference
    -> SIR + independent Metropolis-Hastings
    -> continuation count 与完整终局 rollout
    -> 双侧 Policy-Space Response Oracles (PSRO)
```

所有 Agent 都运行完整游戏，没有 mini game，也没有规则层的最大回合数或平局。

> 当前实现包含 stochastic baselines、inference approximations、full-game rollout 和经验 PSRO。PSRO 的解只针对当前有限策略群体和采样收益矩阵；它不是完整 Fugitive 博弈的 Nash equilibrium，也不是经过证明的 optimal/near-optimal strategy。实验胜率只表示指定 Agent、参数和 seed 分布之间的经验结果，不等于游戏的理论价值。

## 快速开始

需要 Python 3.11 或更高版本。运行时代码只使用标准库；测试额外需要 `pytest`。

```powershell
python -m pip install -e ".[test]"
python -m fugitive --host 127.0.0.1 --port 8000
```

也可以使用安装后生成的等价入口：

```powershell
fugitive --host 127.0.0.1 --port 8000
fugitive-web --host 127.0.0.1 --port 8000
```

然后打开 <http://127.0.0.1:8000>。

交互式探索可使用少量稳定的快捷导入；实现代码仍应从概念所属模块显式导入：

```python
from fugitive import GameEngine, Role
from fugitive.agents import (
    FUGITIVE_AGENT_REGISTRY,
    HierarchicalRandomFugitiveAgent,
    HierarchicalRandomMarshalAgent,
)
```

`fugitive.agents` 只重导出 registry、HR 和 BIR-1 这些高频入口。进阶 Agent 仍从各自实现模块导入，内部代码也保持显式依赖。

界面支持：

- 人类扮演 Fugitive，对战 Marshal Agent；
- 人类扮演 Marshal，对战 Fugitive Agent；
- Agent 对 Agent 逐步或自动运行；
- `omniscient`、Fugitive、Marshal 和 public 四种观战视角；
- `full` 与 `quick` 两种执行 profile；
- belief 质量、采样工作量和 MCMC 接受率等教学诊断；
- 终局 trace、replay manifest 和完整研究数据导出。

浏览器首屏默认是“人类 Fugitive 对 BIR-1 Marshal”，使用 `full` profile。API 空请求的默认模式则是 BIR-1 对 BIR-1 的全知观战。

`full` 使用正式 registry 默认参数；`quick` 使用较小的交互参数。后者会真实改变粒子数和 AgentSpec，因此可能改变动作与胜率，不只是让动画更快。

本地服务器没有认证，会话也只保存在内存中。通常应保持绑定 `127.0.0.1`；只有在可信网络中才应绑定 `0.0.0.0`。

## 规则范围

权威规则文本见 [CANONICAL_RULES.md](docs/rules/CANONICAL_RULES.md)，实现开关与来源说明见 [RULESET.md](docs/rules/RULESET.md)。

当前规则版本固定为：

- 第一版基础规则；
- 不使用 Event 或 SHIFT 牌；
- 牌为 0-42，奇数 Sprint 值为 +1，偶数为 +2；
- Fugitive 第一回合必须建立两个 Hideout；
- 正常 Fugitive 回合先摸一张，然后 Play 或 Pass；
- Pass 不额外摸牌；
- Marshal 第一回合摸两张，之后每回合摸一张；
- 多猜必须全部正确才揭示，否则什么都不揭示；
- 42 的 Manhunt 边界为最高已揭示 Hideout 是否达到 30；
- 游戏只有 Fugitive 或 Marshal 胜利，没有平局。

规则没有定义三個牌堆全部耗尽时如何处理。引擎采用确定性边界约定：应摸牌但所有牌堆都为空时跳过该次摸牌，继续进入本回合动作。这不会制造平局或额外赢家。

## 信息边界

每个策略只能是自己信息集上的函数：

```text
pi_i(action | I_i)
```

引擎只把不可变的 [`Observation`](src/fugitive/game/model.py) 交给 Agent，不会传入真实牌堆顺序或对手私有状态。

| 信息 | Fugitive | Marshal |
| --- | --- | --- |
| 自己手牌 | 可见 | 可见 |
| Fugitive 隐藏 Hideout/Sprint 身份 | 可见 | 不可见 |
| 隐藏 Hideout 下的 Sprint 张数 | 可见 | 可见 |
| 对方抽到的具体牌 | 不可见 | 不可见 |
| 对方选择的牌堆 | 可见 | 可见 |
| 猜测及成功/失败 | 可见 | 可见 |
| 已揭示 Hideout 和 Sprint | 可见 | 可见 |
| 各牌堆剩余张数 | 可见 | 可见 |

Agent 只选择从哪个牌堆摸牌；具体抽到哪张牌由引擎的隐藏牌序决定。全知观战只存在于 Web 序列化层，不会成为 Agent 输入。

Constructive inference 从 [`CompiledMarshalConstraints`](src/fugitive/agents/marshal/inference/constructive/constraints.py) 开始只消费 Marshal Observation。测试还包含 observation non-interference、不同观战视角脱敏和私有状态替换检查。

这是一条数据/API 隔离边界，不是针对恶意 Python Agent 的进程安全沙箱。

## 为什么采用分层随机

直接从所有原子合法动作中等概率抽样会受到动作编码方式支配：

- 同一个 Hideout 可能对应大量 Sprint 子集；
- Marshal 的猜数组合数量随候选数指数增长；
- Pass 只有一个动作。

结果会是 Fugitive 频繁过量 Sprint、Marshal 一次猜很多数字、Pass 几乎消失。为避免这个问题，基础策略先随机选择规则层的宏动作，再在该分支内随机：

```text
Fugitive:
选择牌堆 -> Pass/Play -> Hideout -> Sprint payment

Marshal:
选择牌堆 -> 猜测规模 -> 具体数字集合
```

Fugitive 开局则直接考虑完整的两步计划，避免第一步随机消耗掉第二个 Hideout 所需的资源。

Sprint payment 使用以下资源成本排序：

```text
C(S) = |S| + 0.5 * overpay(S) + 0.75 * future_cards(S)
```

`future_cards` 指打出目标 Hideout 后仍可作为未来 Hideout、却被当前 Sprint 消耗的牌。Agent 默认不把 42 用作 Sprint；这是 baseline 的策略选择，不是引擎规则限制，引擎仍允许合法的 42 Sprint。

## Agent 总览

### Fugitive

| Registry ID | 名称 | 核心思想 | 实现 |
| --- | --- | --- | --- |
| `hierarchical-random` | HR-1 | 分层合法随机 | [hierarchical_random.py](src/fugitive/agents/fugitive/hierarchical_random.py) |
| `belief-informed-random` | BIR-1 Fugitive | 信息集启发式 + epsilon-softmax | [bir.py](src/fugitive/agents/fugitive/bir.py) |
| `continuation-count` | Continuation Count | 有限深度续着数 + 公开隐蔽性 | [continuation_count.py](src/fugitive/agents/fugitive/continuation_count.py) |
| `belief-rollout` | Belief Rollout | 采样 Marshal 手牌并模拟到终局 | [belief_rollout.py](src/fugitive/agents/fugitive/belief_rollout.py) |

### Marshal

| Registry ID | 名称 | Belief/更新方式 | 实现 |
| --- | --- | --- | --- |
| `hierarchical-random` | HR-1 | hard route support | [hierarchical_random.py](src/fugitive/agents/marshal/hierarchical_random.py) |
| `route-count-random` | HR-1.1 | exact compatible-route counts | [route_count.py](src/fugitive/agents/marshal/route_count.py) |
| `support-catalogue-random` | HR-1C | 共享候选集上的 Boolean support | [support_catalogue.py](src/fugitive/agents/marshal/support_catalogue.py) |
| `route-count-catalogue-random` | HR-1.1C | 同一候选集上的 route-count weight | [route_count_catalogue.py](src/fugitive/agents/marshal/route_count_catalogue.py) |
| `belief-informed-random` | BIR-1 | constructive bootstrap particle filter | [bootstrap.py](src/fugitive/agents/marshal/bir/bootstrap.py) |
| `unweighted-constructive-belief-informed-random` | BIR-2U | observation-local unweighted constructive samples | [unweighted.py](src/fugitive/agents/marshal/bir/unweighted.py) |
| `rollout-bir2u` | Rollout BIR-2U | BIR-2U 根 belief + 完整终局 rollout | [rollout_bir2u.py](src/fugitive/agents/marshal/rollout_bir2u.py) |
| `constructive-belief-informed-random` | BIR-2S | observation-local constructive SNIS | [snis.py](src/fugitive/agents/marshal/bir/snis.py) |
| `exact-sprint-belief-informed-random` | BIR-2E | exact Sprint DP reference + SNIS | [exact_sprint.py](src/fugitive/agents/marshal/bir/exact_sprint.py) |
| `mcmc-belief-informed-random` | BIR-3 | SIR + finite-step independent MH | [mcmc.py](src/fugitive/agents/marshal/bir/mcmc.py) |

`belief-informed-random` 在 Fugitive 和 Marshal registry 中分别指两个不同实现。前者是可见信息上的 Fugitive 启发式策略；后者是 Marshal constructive bootstrap filter。

## Fugitive 策略

### HR-1: Hierarchical Legal Random

HR-1 Fugitive 是规则和动作空间的主要下界：

- 从非空合法牌堆中均匀选择；
- 第一回合从完整合法两步计划中选择；
- 正常回合把 Pass 当独立宏动作；
- 先选择 Hideout，再从少量低成本 Sprint payment 中选择；
- 默认 5% 概率进入额外 overpay 分支；
- Pass 概率根据资源状态取约 3%、10% 或 25%；
- 如果可以在不触发 Manhunt 的条件下打出 42，则不 Pass。

它是“合理动作抽象下的随机”，而不是“所有合法原子动作均匀”。

### BIR-1 Fugitive: Information-Set Random

BIR Fugitive 复用 HR-1 的分层动作空间，但使用标准化后的 epsilon-softmax：

```text
P(a) = epsilon / |A|
     + (1 - epsilon) * exp(z(Q(a)) / tau)
       / sum_b exp(z(Q(b)) / tau)
```

默认 `epsilon=0.15`、`tau=0.7`。主要评分包括：

- 抽牌后预期最佳手牌价值；
- 向 42 的相对进度；
- Sprint 张数、overpay 和未来 Hideout 机会成本；
- 剩余手牌的可达 Hideout 数与最远距离；
- Pass 后下一次可能抽牌的价值和公开 catch-risk proxy；
- 打出 42 时的 Manhunt 生存概率。

Manhunt 生存值通过逐猜 rollout 估计：模拟 Marshal 猜中后揭示对应 Hideout/Sprint，再更新 shadow `PathBelief` 后选择下一猜。

这个 shadow Marshal 不知道真实 Marshal 手牌，也没有行为 likelihood；Pass 评分也没有完整模拟中间的 Marshal 回合。因此它是 information-set heuristic，不是 Bayesian planner。

### Continuation Count Fugitive

`continuation-count` 保留 BIR Fugitive 的分层候选，只替换动作评分。它递归统计有限深度内的 bounded Hideout/Sprint **动作树节点质量**；不同 Sprint payment 是不同分支，因此这个量不是 distinct future routes 的数量。每个决策先计算所有候选的 `log1p(count)`，再除以当前候选集最大值，而不是用固定常数截断。相同 `(sorted hand, previous hideout, depth)` 子问题使用 agent-local 精确 memo。

隐蔽性是公开 `PathBelief` 的 hidden-card marginal entropy，catch risk 是 epsilon-soft 单猜 response 在真实隐藏 Hideout 上的质量。Marshal shadow 保留已经发生的公开 Play/Pass 时间线，并追加当前假设动作；它不使用 Marshal 私有手牌。各项都有显式权重，可分别消融。

这个 Agent 的价值是把“保留未来机动性”和“当前隐藏点是否容易被猜”写成可检查的中间量；它仍然没有采样真实 Marshal 手牌，也不模拟完整对手策略。

### Belief-Rollout Fugitive

`belief-rollout` 先用 Continuation Count 产生有限候选，再从 Fugitive 信息集采样世界。Fugitive 已知自己的手牌、路线和具体抽牌，因此未知量只剩：Marshal 从每个公开牌堆抽走了哪些牌，以及未来牌序。采样器按公开抽牌次数均匀分配 Marshal 手牌，其余牌随机排序。

每个候选动作在相同 sampled worlds、未来牌序和 continuation-policy seeds 下模拟到规则终局，最终按 Fugitive 胜率做 epsilon-softmax。候选在查看世界前固定，continuation agent 只接收自己的正常 Observation。该策略是 root-sampled Monte Carlo response，不是完整 belief-state dynamic programming。

计算量由一个确定性的 `max_terminal_simulations` 总预算控制。预算在候选之间平均分成 paired scenarios，不足一个完整候选组的余数明确不用；每个已运行的 simulation 都到达规则终局，没有 horizon、wall-clock cutoff 或伪造赢家。若应用预算后只评估一个候选，策略在采样世界之前直接返回概率 1。

## Marshal 的路线模型

### PathBelief

[`PathBelief`](src/fugitive/agents/marshal/inference/path_belief.py) 用动态规划计数严格递增、与 Marshal 公开信息相容的 Hideout 数值路线。它处理：

- 已揭示 Hideout；
- Marshal 手牌和公开牌造成的排除；
- 每段公开 Sprint 张数带来的距离上界；
- 单猜失败；
- 多猜失败的正确联合含义；
- 猜测发生时的历史 route length；
- 牌堆来源与 Hideout 创建时的资源前缀限制。

在它所定义的 route-value path 模型中，计数和均匀 route sampling 是 exact 的。

但 `PathBelief` 不给隐藏 Sprint 分配具体身份。若某段有 `k` 张隐藏 Sprint，它只使用保守上界 `3 + 2k`。它也不计算手牌、Sprint、抽牌历史和剩余牌堆组成的完整世界数量，更不使用 Fugitive policy likelihood。

因此：

- route-count marginal 是“相容数值路线等权”下的 marginal；
- 它不是完整隐藏世界 posterior；
- `hard_route_support_count` 不是完整世界数量，也不是客观概率。

### HR-1 Marshal: Hard-Support Random

HR-1 Marshal 只询问一个数字或组合是否存在至少一条相容路线：

- 抽牌堆均匀；
- 支持集内的单猜均匀；
- 猜测规模使用偏向单猜的截断几何分布，默认 continuation 为 0.10；
- 多猜候选只保留有限个 information-consistent 组合；
- Manhunt 始终单猜，并在当前支持集内均匀。

它故意丢弃不同路线的完成数量，是 Boolean support baseline。

### HR-1.1 Marshal: Route-Count Random

HR-1.1 保持抽牌选择均匀，只改变猜测权重：

```text
P(n) proportional to count(n)^alpha
```

并混入 10% uniform epsilon。多猜使用联合 route count，而不是边际概率乘积；Manhunt 使用 `count(n)^2`，每次揭示后重新计算。

旧 `hierarchical-random` 与 `route-count-random` 的候选生成历史不同，因此它们适合展示两种完整 baseline，但不是严格单变量消融。为隔离 weighting 的影响，项目另外提供：

- `support-catalogue-random` (HR-1C)；
- `route-count-catalogue-random` (HR-1.1C)。

两者都调用 [`build_marshal_guess_catalogue`](src/fugitive/agents/marshal/candidates.py)，共享单猜、pair/triple 候选、候选上限、猜测规模先验和 observation-derived 随机域；唯一核心差异是支持集等权还是按 compatible-route count 加权。因此 HR-1C 与 HR-1.1C 才直接回答：

> 只知道“可能/不可能”是否足够，还是 route completion mass 能显著改善猜测？

HR-1.1 仍然只是 route-count model，不是 complete-world posterior。

## Marshal BIR 的共享动作策略

BIR-1、BIR-2U、BIR-2S、BIR-2E 和 BIR-3 共享 [`BeliefInformedMarshalActionPolicy`](src/fugitive/agents/marshal/action_policy.py)。不同 Agent 主要替换 belief backend，而不复制猜测策略。

共享策略包括：

- 抽牌：对每个可能抽到的具体牌做条件化，计算抽完后的最佳猜测价值，再求期望；
- 单猜与多猜：从 marginal 较高的数字和高质量 hidden-route hypotheses 生成有限候选；
- 联合成功率：直接在粒子内计算 `P(G subset of H)`，不把边际概率相乘；
- 猜测评分：

```text
Q(G) = |G| * P(success)
     + terminal_bonus * P(success)
     - escape_risk * P(failure)
```

- 动作选择：先对猜测规模做 epsilon-softmax，再在该规模内做 epsilon-softmax；
- Manhunt：按当前 particle marginal 的 `p(n)^alpha` 加 epsilon 选择单猜，揭示后重新推断。

共享策略让 BIR 系列更适合教学比较：胜率差异主要来自 belief representation、proposal weighting 和更新方式，而不是每个文件各自实现了一套猜测启发式。

### Rollout BIR-2U Marshal

`rollout-bir2u` 使用 BIR-2U 生成根节点 complete-world belief，并由共享 BIR 动作策略给出有限 shortlist。随后对粒子按当前权重重采样，为每个世界随机生成未来牌序，再把每个候选猜测或抽牌动作模拟到规则终局。

候选之间使用 common random numbers：相同世界、相同未来牌序、相同 Fugitive/Marshal continuation seeds。这样动作价值差异较少被无关 Monte Carlo 噪声淹没。默认对手是冻结的 Fugitive 策略混合。计算量与 Fugitive rollout 一样由 `max_terminal_simulations` 统一限制；所有实际运行的样本都模拟到规则终局。shortlist 在应用预算后只剩一个候选时，会跳过 root belief resampling、world generation 和 terminal rollout。

两侧 rollout 都提供独立 diagnostics side channel：候选胜场、paired outcome 四格表、动作差值与标准误、实际 terminal simulations 和 continuation decisions。它们不会进入动作 Observation，也不改变 replay。

## Complete-world constructive inference

一个 Marshal complete world 包含：

- Fugitive 当前手牌；
- Marshal 手牌；
- 完整 Hideout 路线；
- 每个 Sprint stack 的具体牌；
- Fugitive 的隐藏抽牌序列；
- 三个剩余牌堆。

[`ConstructiveWorldSampler`](src/fugitive/agents/marshal/inference/constructive/sampler.py) 按以下阶段构造世界：

1. 从 Marshal Observation 编译公开约束；
2. 用 `PathBelief` catalogue 抽取一条相容 Hideout 路线；
3. 给隐藏 Sprint stack 分配来源、Sprint 值和具体身份；
4. 匹配 Fugitive draw slots，并保证牌在使用前已经被抽到；
5. 生成剩余手牌和牌堆；
6. 用完整世界校验器检查牌唯一性、来源、时间和公开历史；
7. 保存完整 proposal probability `log_q`。

每个 constructive sample batch 还携带当前 Observation 的 canonical hash。共享 builder 在把 batch 转成 particle belief 前会重新计算并比对该 hash；因此 cache、并行实验或新 backend 误把 Observation A 的 batch 交给 Observation B 时会立即失败，而不会生成来源不一致的 belief。

参考 target `pi` 定义为：

```text
pi(x) = constant, if x is an observation-consistent complete world
        0,        otherwise
```

这是一个 policy-agnostic、uniform-over-consistent-worlds 的建模选择。它没有使用 Fugitive 行为 likelihood，所以不是“理性 Fugitive”条件下的行为 posterior。

默认正式 Agent 不设置 node/proposal cutoff，要求 sampler 产生完整、importance-valid 的 batch。复杂 observation 可能很慢；`search_nodes` 默认只是工作量计数。低层 `SamplingBudget` 主要供测试和算法实验注入，正式 backend 不会静默接受 degraded partial batch。

## BIR belief backends

### BIR-1: Constructive Bootstrap Particle Filter

BIR-1 第一次构造 belief 或 support 完全归零时，使用与 BIR-2U/BIR-2S 相同的 sequential constructive proposals。每个 accepted proposal draw 获得相同初始质量；重复抽到同一个 physical world 会保留 multiplicity。

后续 observation 不从头构造，而是增量推进：

- Marshal 自己的 draw 按观测到的牌做条件化；
- Fugitive 隐藏 draw 在每个 particle 的剩余牌上展开；
- guess success/failure 作为可行性约束过滤；
- hidden play 在与公开动作相容的 legal extensions 间传播父粒子质量；
- duplicate worlds 聚合；
- population 超过上限，或有信息事件后的 pre-resample ESS 不高于 `0.5N` 时 systematic resample；
- 只有 support 完全归零时才 observation-local fresh reset。

公开事件只被当作 feasibility constraints，不被当作 Fugitive 行为似然证据。

当 hidden-play 组合数不超过 4096 时会枚举；更大时每个 parent 最多保留 64 个 deterministic-RNG proposals。这是 BIR-1 算法定义中的 bounded approximation，不是仅用于测试的运行时 cutoff。

因此 BIR-1 是 heuristic constructive bootstrap filter：

- 会保留祖先连续性；
- 会积累 Monte Carlo error、resampling history 和 ancestry collapse；
- 增加粒子数可以降低部分方差，但不能消除 uniform-extension prior 或 bounded proposal 的系统偏差；
- 非空但高度集中的 belief 不会自动 fresh rebuild。

### BIR-2U: Observation-Local Constructive, Unweighted

BIR-2U 在每个新 Observation 上独立重建 complete-world batch，并对每次 accepted proposal draw 赋权 `1/N`。

若 proposal 输出 `A, A, B`，聚合质量为：

```text
P(A) = 2/3
P(B) = 1/3
```

它近似的是 proposal distribution `q`，不是 uniform complete-world target。保留它的目的正是测量 proposal bias。

### BIR-2S: Observation-Local Constructive SNIS

BIR-2S 与 BIR-2U 使用完全相同的：

- sequential constructive sampler；
- proposal seed domain `fugitive.marshal.sequential-constructive-belief.v1`；
- reference target；
- accepted world batch；
- Marshal action policy。

唯一核心差异是 BIR-2S 使用 self-normalized importance weights：

```text
w_i = [pi(x_i) / q(x_i)]
      / sum_j [pi(x_j) / q(x_j)]
```

因此 BIR-2U 与 BIR-2S 是项目中最干净的消融实验：同一批 proposals，只改变是否校正 proposal bias。

Sequential Sprint proposal 只按当前 identity multiplicity 和当前 draw feasibility 选择分支，不递归计算所有未来 descendants；它通过记录完整 `log_q` 让 SNIS 可以进行有限样本校正。

### BIR-2E: Exact Sprint DP Reference

BIR-2E 保留 exact Sprint assignment dynamic program 作为独立慢速 Agent。

给定一条 Hideout 路线后，Exact Sprint DP 会递归计数所有后续 Sprint category allocation、具体身份和 draw completion，再按 descendant count 采样。它适合：

- 与 sequential Sprint proposal 做小状态枚举对照；
- 比较速度、proposal variance、SNIS ESS 和最大权重；
- 作为教学中的 reference implementation。

“Exact”只修饰给定路线后的 Sprint/draw proposal。路线本身仍由 Monte Carlo 抽样，最终 belief 仍是有限样本 SNIS；BIR-2E 不是 exact posterior，更不是 exact game strategy。

### BIR-3: SIR + Independent MH

BIR-3 的每个 observation-local inference 包含：

1. 生成与 BIR-2S 相同类型的 constructive SNIS batch；
2. 用 systematic resampling 产生等权 chains；
3. 每条 chain 从同一个 complete-world proposal `q` 独立提出新世界；
4. 使用 independent Metropolis-Hastings 接受率：

```text
alpha(x, y) = min(1, pi(y) q(x) / [pi(x) q(y)])
```

默认每条 chain 只运行 1 个 MH step。这是 finite-step independent MH，不是 hand swap、Sprint swap 等 local/block MCMC，也不声称已经混合或收敛。

Acceptance rate 可能包含“接受了相同 world”；change rate 更接近实际链移动，但两者都不是收敛证明。

## 最有教学价值的比较

### 1. HR-1C Marshal vs HR-1.1C Marshal

共享完全相同的候选 catalogue、猜测规模先验和随机域，只比较 Boolean route support 与 route-count weighting。旧 HR-1/HR-1.1 仍可作为历史完整策略比较，但不能解释成严格单变量实验。

### 2. BIR-1 Marshal vs BIR-2U

两者 fresh construction 的 proposal 与等权语义相同。比较重点是：

- 在线增量维护与 observation-local rebuild；
- 祖先连续性与 ancestry collapse；
- resampling 和 support-extinction reset；
- 是否能重新解释全部未观测历史。

这是一组很有价值的系统比较，但不是严格单变量实验，因为 incremental transition kernel 与全局 constructive proposal 也不同。

### 3. BIR-2U vs BIR-2S

完全相同 proposals 下比较 uniform accepted-proposal weights 与 SNIS correction。这是最干净的 weighting ablation。

### 4. BIR-2S vs BIR-2E

比较快速 sequential Sprint proposal 与 exact descendant-count Sprint reference，观察时间、ESS 和方差。

### 5. BIR-2S vs BIR-3

比较有限样本 SNIS 与 SIR 后再做 finite independent-MH rejuvenation。需要同时报告 MH acceptance、实际 change rate 和最终 world/route diversity。

### 6. HR-1 Fugitive vs BIR Fugitive

两者共享分层动作抽象，主要比较纯随机宏动作与进度、资源、mobility、Pass、Manhunt 启发式评分。

### 7. 启发式 vs full-game rollout

Fugitive 可比较 BIR、Continuation Count 与 Belief Rollout；Marshal 可比较 BIR-2U 与 Rollout BIR-2U。报告时必须同时给出 shortlist 大小、`max_terminal_simulations`、实际使用/未使用的预算、wall-clock 时间和对手策略混合，否则胜率提升无法与额外计算预算分开解释。

## 诊断指标

Web 和 experiment runner 会记录 Marshal belief diagnostics。它们描述 Agent 实际使用的近似 belief，不是客观世界概率。

| 指标 | 含义 |
| --- | --- |
| `particle_entries` | 粒子条目数，clone 分开计数 |
| `unique_worlds` | 按完整 physical world identity 合并后的数量 |
| `entry_ess` | 按 entry weight 计算的 ESS；重采样后可能看起来很高 |
| `world_ess` | 合并相同 world 后的 ESS，更能暴露 clone concentration |
| `max_world_mass` | 最大 complete-world 质量 |
| `unique_hidden_routes` | 不同当前未揭示 Hideout 集合数量 |
| `hidden_route_ess` | 合并相同 hidden-route hypothesis 后的 ESS |
| `max_hidden_route_mass` | 最大 hidden-route hypothesis 质量 |
| `hard_route_support_count` | `PathBelief` 的 route-value path 数量 |
| `proposal_importance_ess` | constructive batch 的 reference SNIS 权重 ESS |
| `max_normalized_importance_weight` | 最大 reference importance weight |
| `dead_end_route_proposals` | 路线无法完成 Sprint/draw assignment 的 proposal 数 |
| `search_nodes` | 确定性 DP 工作量，不是 wall-clock 时间 |

解读时应同时看 `entry_ess`、`world_ess` 和 hidden-route 指标。只看重采样后的 entry ESS 会掩盖大量 clone 指向同一个世界的退化。

BIR-1 的 `origin_fresh_sampling_*` 只描述当前 population 起源的 fresh batch；fresh build、incremental update、support-extinction reset 和 resample 计数是 backend 累计工作。

BIR-2U 也会报告 reference importance ESS，但它实际使用的仍是 uniform weights；必须结合 `weighting_id` 解读。

Diagnostics 在动作已经提交后读取，是不影响策略的 side channel。读取失败只记录 `InferenceDiagnosticFailure`，不会回滚动作、改变 winner，也不会进入 replay manifest 或 state fingerprint。Web 只向 Marshal/omniscient 观战视角展示 belief diagnostics。

## 参数与执行 profile

正式 registry 默认值和 Web `quick` profile 的主要差异如下：

| Marshal Agent | `full` 粒子数 | `quick` 粒子数 |
| --- | ---: | ---: |
| BIR-1 | 2000 | 384 |
| BIR-2U | 2000 | 128 |
| Rollout BIR-2U | 2000；192 terminal simulations；8 candidates | 64；12 terminal simulations；3 candidates |
| BIR-2S | 2000 | 128 |
| BIR-2E | 128 | 32 |
| BIR-3 | 1000 | 256 |

BIR-3 两个 profile 默认都是每条 chain 1 个 MH step。Belief-Rollout Fugitive 的 `full/quick` 分别使用 192/12 个 terminal simulations 和 8/3 个 candidates。BIR-2E 与 rollout Agent 被设计为慢速教学算法；复杂 action 运行数分钟是允许的。

完整、可复现的参数 schema 在 [registry.py](src/fugitive/agents/registry.py) 中显式定义。固定的 algorithm、proposal、reference target、weighting、seed domain 和 MCMC kernel ID 会进入 AgentSpec。

推断内核中的 `target_id` 是数学兼容指纹：sampler、sample batch 和 MH kernel 用它确认 `log_target` 与接受率属于同一个 target。AgentSpec 中使用 `reference_target_id`：它说明算法用来校正或对照的目标，不暗示未加权的 BIR-1/BIR-2U 已经服从该分布。

seed domain 也属于可复现的算法协议：

| Agent | AgentSpec 字段 | Domain |
| --- | --- | --- |
| BIR-1 / BIR-2U / BIR-2S | `belief_seed_domain` | `fugitive.marshal.sequential-constructive-belief.v1` |
| BIR-2E | `belief_seed_domain` | `fugitive.bir2e.exact-sprint-belief.v1` |
| BIR-3 initial SIR | `initial_seed_domain` | `fugitive.bir3.initial-constructive-belief.v1` |
| BIR-3 MH rejuvenation | `rejuvenation_seed_domain` | `fugitive.bir3.independent-mh.v1` |

BIR-1 的 fresh/reset、BIR-2U 和 BIR-2S 故意共享 sequential domain，以便同 seed、同 observation、同粒子数时比较同一批 ordered proposals；BIR-1 的后续 incremental population 不要求与 observation-local BIR-2U/BIR-2S 相同。BIR-2E 与 BIR-3 使用独立 domain，避免不同推断实验意外共享同一随机流。

## 可复现实验与 replay

`runtime.runner` 提供两个单局入口：已有 Agent 实例使用 `run_match`，registry 名称与参数使用 `run_registered_match`。两者都返回 `MatchRun`；其 `MatchStatus` 明确区分规则终局、显式 watchdog 截断和执行错误。推荐批量研究从 registry 入口开始：

```python
from fugitive.runtime.manifest import ReplayManifest
from fugitive.runtime.replay import replay_manifest
from fugitive.runtime.runner import run_registered_match

run = run_registered_match(
    master_seed=123,
    fugitive_name="hierarchical-random",
    marshal_name="route-count-random",
)

manifest = ReplayManifest.from_json(run.manifest.to_json())
verified = replay_manifest(manifest)

print(run.status.value)
print(run.game_result)
print(verified.final_state_sha256)
```

`master_seed` 会通过 domain-separated hashing 派生互不共用随机流的：

- deck seed；
- Fugitive policy seed；
- Marshal policy seed。

合法 seed 范围是 `0 <= seed <= 2^64 - 1`。Web/JSON 中超过 JavaScript safe integer 范围的 seed 必须作为规范十进制字符串发送。

严格复现需要相同的：

- 代码和规则指纹；
- master seed；
- Agent registry ID；
- resolved profile；
- 全部 Agent 参数与固定算法元数据。

Replay 会逐步重放并验证动作、规则指纹、结果和最终状态，但不会重新调用 Agent 来证明当前代码仍会选择同一动作。

`max_decisions=None` 是正式默认值，表示运行到规则定义的胜负。实验可以显式设置 watchdog；达到限制时状态为 `truncated`，winner 为 `None`，不会虚构平局或赢家。

Web 导出的完整 trace 包含双方私有牌、完整动作和派生 seeds，应把它视为研究数据，而不是可公开分享的单方观战日志。

批量比较使用同一套单局 runner，因此 Agent 配置、seed 派生、错误记录、manifest 和 replay 协议不会另写一套。以下研究命令需在仓库根目录运行；`experiments` 有意不作为可安装核心包或 console script 发布：

```powershell
python -m experiments.tournament `
  --fugitive hierarchical-random `
  --fugitive belief-informed-random `
  --marshal hierarchical-random `
  --marshal route-count-random `
  --games 100 `
  --root-seed 20260726 `
  --output experiment_runs/hr-comparison
```

同一个 game index 在所有 matchup 中共享一个派生 master seed，便于做 paired comparison。每局结束后立即追加 `games.jsonl`，并分别保存 replay manifest、inference diagnostics 和 rollout diagnostics；`summary.json`、`summary.csv` 与 `summary.md` 会同步更新。rollout side channel 包含实际/未用 simulation 预算、paired outcomes、动作差值和标准误，但不进入 replay manifest，也不改变随机轨迹。正式默认仍是 `max_decisions=None`，错误和显式 watchdog 截断不会被算成任一方获胜。

中断后可以续跑，也可以把每个 matchup 的目标局数从较小的校准值增加到较大值：

```powershell
python -m experiments.tournament `
  --fugitive hierarchical-random `
  --fugitive belief-informed-random `
  --marshal hierarchical-random `
  --marshal route-count-random `
  --games 200 `
  --root-seed 20260726 `
  --output experiment_runs/hr-comparison `
  --resume
```

续跑时，Agent 集合、profiles、root seed、规则指纹和 watchdog 设置必须保持不变；已有对局不会重复执行。`experiment_runs/` 默认不进入 Git，正式报告应引用其配置、代码 commit 和原始结果文件。

## PSRO 经验策略搜索

[`algorithm.py`](src/fugitive/psro/algorithm.py) 实现角色分离的完整 Policy-Space Response Oracles 循环：

```text
策略群体
  -> 补齐缺失的完整对局收益格
  -> 求解有限零和经验矩阵的双侧元策略
  -> Fugitive 与 Marshal 分别产生对手混合下的 response candidate
  -> 用独立 seeds 与当前 restricted pure best response 成对比较
  -> 只有提升置信区间下界通过阈值才加入，并只评估新增行/列
  -> 保存 checkpoint
```

经验矩阵统一使用 Marshal 胜率，因此 Marshal 最大化、Fugitive 最小化。内置 meta-solver 是 simultaneous multiplicative weights；它报告当前受限矩阵内的 lower/upper bound、双侧 response residual 和 duality gap。这些是 restricted-game diagnostics，不是完整游戏 exploitability。

[`policy_adapter.py`](src/fugitive/psro/policy_adapter.py) 把通用核心连接到 registry；[`payoff.py`](src/fugitive/psro/payoff.py) 负责完整对局收益、validation 和 holdout；[`runner.py`](src/fugitive/psro/runner.py) 编排 checkpoint 与迭代。每个 pure policy 保存一次真实执行 `AgentSpec` 和一次声明的 non-search planning leaf。search response 在 checkpoint 中只引用对手 `policy_id + weight`；运行 payoff game 时再从当前 `PolicyPopulation` repository 解析 leaf specs。顶层 search 的 planning depth 固定为 1：如果 rollout 选中另一个 search policy，只执行后者声明的 leaf，不会递归启动第二层 rollout。

默认 response candidates 是便宜的 `continuation-count` Fugitive 和 BIR-2U Marshal，默认 population 不含这两个候选。它们不读取当前 opponent mixture，所以默认运行是一次经过验证的 population augmentation：候选被接纳或拒绝后，下一代通常没有新的不同 policy。要研究真正随 mixture 变化的多代 response，可显式使用 `belief-rollout` / `rollout-bir2u`；它们会读取冻结的 opponent leaf mixture，但计算成本也高得多。

payoff evaluation 使用 append-only file-per-game ledger。每个完整对局结束后立即原子写入 raw outcome、manifest 和确定性 sample ID；identity 同时包含规则版本与规则哈希，中断后只补缺失 seed。CLI 默认最多使用 4 个进程，把不同 cell/game 一起并行，并在 stderr 显示 schedule、完成数和复用数。checkpoint 仍按完整 generation 保存，但昂贵的 generation 中间结果已由 ledger 独立保护。

seed schedule 分为互不重叠的 `meta-train`、`response-validation` 和 `final-holdout` 三个域：

- `meta-train` 只用于 restricted payoff matrix 和 meta-strategy；
- `response-validation` 中保存的是 base seeds；每一代先从它们确定性派生一批新的 game seeds。一个派生 seed 从对手 mixture 抽取一个 pure opponent，candidate 与当前 restricted pure best response 再共享同一个完整对局 seed。这样保留代内 common random numbers，同时避免不同 generation 反复使用同一批牌序和策略随机流。角色胜率差形成 paired gains，并用确定性的 percentile bootstrap 求置信区间；只有下界严格高于 `response_minimum_gain` 才接纳 candidate；
- `final-holdout` 从最终双方 mixture 抽样并报告 Marshal payoff 及 bootstrap interval。它不写回 payoff matrix，也不参与 policy 选择。

所有 validation 报告都会跨代累计进 checkpoint；被拒策略不会进入 population。这里的“通过验证”只表示该有限 candidate 在给定样本和置信阈值下优于当前 restricted pure response，不等于它是完整 Fugitive 游戏的精确 best response。

安装后可运行：

```powershell
fugitive-psro `
  --fugitive hierarchical-random `
  --fugitive belief-informed-random `
  --marshal support-catalogue-random `
  --marshal route-count-catalogue-random `
  --games-per-cell 20 `
  --response-validation-games 40 `
  --final-holdout-games 100 `
  --response-minimum-gain 0.0 `
  --confidence-level 0.95 `
  --bootstrap-replicates 10000 `
  --workers 4 `
  --generations 3 `
  --seed 20260726 `
  --ledger experiment_runs/psro/payoffs `
  --output experiment_runs/psro/checkpoint.json
```

`--resume` 从 checkpoint 继续，并使用其中记录的实验配置。ledger 路径不写入 checkpoint：首次运行使用默认 ledger 且沿用同一个 output 路径时可以省略；如果传过自定义 `--ledger`，或 resume 时改变 `--output`，必须再次传入原 ledger 路径，才会复用已经完成的游戏。正式默认没有 decision cutoff。要显式研究 search response，可设置 `--fugitive-response-agent belief-rollout` 或 `--marshal-response-agent rollout-bir2u`，并通过对应的 `--*-response-parameters` JSON 指定预算；这类运行仍可能很慢。

最小 API 示例：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/agents

$body = @{
  mode = "spectate"
  fugitive_agent = "hierarchical-random"
  marshal_agent = "constructive-belief-informed-random"
  execution_profile = "full"
  seed = "123"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/games `
  -ContentType application/json `
  -Body $body
```

## Observation 教学工具

引擎是游戏合法性的权威。对于手工构造 Marshal Observation 的教学实验，可以额外使用轻量契约检查：

```python
from fugitive.agents.marshal.inference.observation_validation import (
    validate_marshal_observation_contract,
)

validate_marshal_observation_contract(marshal_observation)
```

该工具检查：

- public visibility；
- history shape；
- 公开牌和牌堆张数守恒；
- route creation 时可见的资源前缀；
- guess/reveal 一致性。

它不会重放完整 phase machine，不证明 Observation 来自合法完整对局，也不保证至少存在一个相容 hidden world。这个边界是有意保留的，避免为了防御引擎不可能产生的状态而让教学代码失去清晰度。

### 固定 Observation inference benchmark

直接比较不同 BIR 对局胜率会同时混入 belief 质量、动作策略和随机牌局。`experiments.behavior` 会重放同一批 manifest，在每个 Marshal 决策前截取完全相同的 Observation，然后分别运行 BIR-1/2U/2S/2E/3 backend：

```powershell
python -m experiments.behavior `
  experiment_runs/reference/manifests/game-000001.json `
  --agent BIR-1 `
  --agent BIR-2U `
  --agent BIR-2S `
  --parameters '{"particle_count": 2000}' `
  --replicates 3 `
  --output experiment_runs/inference-benchmark.json
```

输出包括 hidden-card Brier score/log loss、真实 hidden route 的 posterior mass/rank/top-k coverage，以及共享猜测 catalogue 上的 joint calibration bins。离线真值只在 `backend.infer(observation)` 返回后用于评分；同一 manifest/replicate 内复用同一个 backend 实例，因此 BIR-1 保留增量 filter 语义，而不同 manifest 或 replicate 之间重新构建。

外层 `comparison_seed` 只由 benchmark root seed、manifest canonical hash 和 replicate 产生，与 backend 名称及参数无关。于是相同粒子数的 BIR-2U/BIR-2S 会收到同一组 ordered constructive proposals，只改变是否使用 importance weights；Agent 名称和参数哈希仍写入结果元数据，但不参与配对 seed。

输出同时提供逐 manifest/replicate 的 `runs` 和 `game_first` 汇总。`game_first` 先在每个 source game 内等权平均 inference replicates，再让不同 source game 等权进入总结果；重复输入同一 manifest 不会让该局获得额外权重。它还保留逐 replicate 的跨游戏汇总，便于做 backend 间的 paired comparison。joint-calibration bins 仍按原始 run 报告，不对已经分箱的结果做含义不清的二次平均。

## 代码结构

```text
src/fugitive/
  game/                           # 规则、状态机、Observation 与动作执行
    model.py
    rules.py
    engine.py
    driver.py
    observation.py
  agents/
    common/                       # 双方 baseline 共用的分布与动作工具
    planning/                     # determinization、rollout、策略混合
    fugitive/                     # Fugitive 各策略，一种算法一个文件
    marshal/                      # Marshal 各策略及共享候选/动作策略
      bir/                        # BIR-1、2U、2S、2E、3 Agent
      inference/                  # Marshal 隐藏世界推断
        constructive/             # 约束、proposal、target、Sprint DP
        particles/                # bootstrap 与 batch-to-belief
    registry.py                  # 显式、可复现 Agent 配置
  runtime/
    manifest.py                  # MatchRun/MatchStatus 与 replay manifest
    runner.py                    # 单局 run_match/run_registered_match
    replay.py                    # 确定性轨迹重放
  psro/                          # 作为核心算法保留在安装包中
    population.py                # pure-policy population
    payoff_matrix.py             # 经验收益矩阵
    solver.py                    # restricted-game meta solver
    algorithm.py                 # PSRO 迭代
    payoff.py                    # 完整对局收益与 holdout
    runner.py                    # checkpointed experiment orchestration
    cli.py                       # fugitive-psro 入口
  web/
    session.py                   # 会话、视角与序列化
    server.py                    # HTTP API 与 CLI
    static/                      # 浏览器界面
  shared/
    reproducibility.py           # seed 与 AgentSpec 协议

experiments/                     # 不属于安装包的研究/分析入口
  tournament/                    # 配对批量对局、报告、CLI
    model.py
    runner.py
    reporting.py
    cli.py
  behavior/                      # 固定 Observation 推断比较
    benchmark.py
    metrics.py
    cli.py
```

依赖方向保持单向：`game` 不依赖 Agent；`agents` 只依赖 `game/shared`；`runtime` 组合引擎与 Agent；`psro` 在 `agents/runtime` 之上实现核心策略搜索；`web` 与仓库外层的 `experiments` 都是调用层。外层实验代码不进入 wheel，也不反向成为 Agent 或规则引擎的依赖。

不同角色和不同 BIR Agent 分别位于独立文件。双方通用的 baseline 工具放在 `agents/common`，搜索共用件放在 `agents/planning`，而 Marshal 专属的 belief machinery 归 `agents/marshal/inference` 所有。共享动作策略和 inference primitives 通过组合注入，而不是通过继承关闭无关父类状态。

## 测试

运行完整测试：

```powershell
python -m pytest
```

常用定向测试：

```powershell
python -m pytest tests/game/test_engine.py
python -m pytest tests/agents/marshal/test_observation_validation.py
python -m pytest tests/agents/marshal/test_unweighted_constructive_bir.py
python -m pytest tests/agents/marshal/test_exact_sprint_bir.py
python -m pytest tests/agents/marshal/test_mcmc_bir.py
python -m pytest tests/agents/common/test_rollout.py
python -m pytest tests/experiments/test_observation_benchmark.py
python -m pytest tests/psro/test_psro.py
python -m pytest tests/runtime/test_reproducibility_integration.py
python -m pytest tests/web/test_web.py
```

当前测试覆盖：

- 首回合、抽牌、Pass、Sprint overpay、多猜、Manhunt 和终局规则；
- 全牌唯一分区、路线递增和完整世界一致性；
- Fugitive/Marshal/public/omniscient observation 脱敏；
- `PathBelief` route count 与均匀 unranking；
- BIR-1 incremental parent-mass conservation、ESS resampling 和 support reset；
- 同 seed 下 BIR-1 fresh = BIR-2U worlds = BIR-2S worlds；
- BIR-1/BIR-2U 等权以及 BIR-2S 只改变 importance weights；
- duplicate proposal multiplicity；
- sequential 与 exact Sprint backend 的小状态枚举对照；
- independent-MH 接受率、kernel metadata 和 diagnostics；
- 双侧 determinization、私有信息隔离、common random numbers 和终局 rollout；
- 固定 Observation proper scores、route coverage 与 joint calibration；
- 固定 Observation 多 replicate、跨 backend common comparison seed 与 U/S 同 proposals；
- PSRO 缺失格增量评估、restricted equilibrium、双侧 response 和 checkpoint resume；
- PSRO policy-ID leaf materialization、非递归 search、逐局 ledger、paired payoff、response validation 与 holdout；
- seed domain separation、manifest replay 和 Web 64-bit seed round trip；
- reviewer 发现的合法 Manhunt long-tail 状态回归。

## 历史实现

旧启发式、搜索、训练、tournament、mini game、早期教程和实验结果位于本地 `archive/`，仅供课程回顾。该目录和 `review/` 均被 Git 忽略，不属于当前安装包、活动源码或正式测试。
