# Review 意见评估与修改方案

**评估日期：** 2026-07-27
**评估对象：** `review_psro_updated.md`（外部审查报告）
**评估基准：** 仓库 HEAD `fa237cd`（`Reorganize core, agents, and experiment packages`）
**方法：** 全部结论均在当前 HEAD 上实际运行验证，不接受快照文档的转述

---

## 0. 总结论

审查报告的**核心判断是正确的，值得接受**。它最重要的两个结论——当前 3×3 经验矩阵是纯策略鞍点、以及下一阶段瓶颈在策略空间而非 PSRO 框架——我都独立复现并确认了。

但报告有 **1 项结论在当前 HEAD 上已经失效**，并且 **完全遗漏了一个可以把 Fugitive Agent 成本降低 1.8 倍的性能问题**——而这恰好直接影响它自己推荐的主线（参数搜索）的可行性。

| 类别 | 数量 | 处理 |
| --- | --- | --- |
| 验证正确、应接受 | 7 项 | 纳入方案 |
| 已失效／基于快照而非仓库 | 1 项 | **从优先级清单删除** |
| 正确但需强化机制说明 | 1 项 | 补充根因后纳入 |
| 报告遗漏、我新发现 | 4 项 | 新增为方案条目 |

---

## 1. 逐项裁定

### 1.1 应接受（已验证）

| 报告章节 | 结论 | 我的验证方式 |
| --- | --- | --- |
| §4.6 纯策略鞍点 | **正确** | 见 §2.1，精确 duality gap = 0.0 |
| §4.6 `with_evaluations` 不能覆盖 cell | **正确** | `payoff_matrix.py:184` 显式 `raise`；相同估计允许，不同估计拒绝 |
| §4.6 默认 response 不读 opponent mixture | **正确** | 见 §2.3，机制比报告描述得更具体 |
| §4.7 缺少参数化策略表示 | **正确，且比报告描述更严重** | 见 §2.4 |
| §4.3 评分常数散落在实现里 | **正确** | `continuation_count.py` 的 `2.0`/`2.75`/`4.0`/`-0.35`/`0.25` 均为字面量 |
| §5.2 参数搜索范围 | **正确，且今天就能跑** | 见 §2.5，14 个建议参数全部已在 registry allowlist 内 |
| §7.3 不要靠加 solver 迭代 | **正确，但报告没说清原因** | 见 §2.2，实际是**反效果** |

### 1.2 应删除：规则文档缺失（§4.2、§8.A.1 第 1 项）

报告称 `docs/rules/CANONICAL_RULES.md` 缺失、`test_embedded_rules_fingerprint_matches_the_canonical_rules_file` 失败，并把"补齐规则文档"列为近期优先级 A 的**第一项**。

在当前 HEAD 上这是**不成立的**：

```
$ ls docs/rules/
CANONICAL_RULES.md   RULESET.md   sources/

$ python -m pytest -q
290 passed
```

规则文档存在，全部 290 个测试通过，包括报告点名的那个 fingerprint 测试。这是 `fugitive(8).zip` 打包脚本漏掉了 `docs/`，不是代码或仓库问题。

**处理：** 从优先级 A 删除第 1 项。报告 §4.2 里真正有效的是它的第 2–4 条子项（修打包脚本、从 clean tree 构建、排除 `__pycache__`），那属于交付整洁性，不属于代码缺陷，优先级低。

> 这也提示一个流程问题：**审查应针对 Git 提交，而不是 zip 快照。** 否则打包遗漏会伪装成代码缺陷，浪费一整个优先级 A 名额。

---

## 2. 验证细节与我的独立发现

### 2.1 纯鞍点：确认，且比报告说得更干净

报告 §4.6 说 `99.56%/99.79%` 的权重"主要是近似求解器的数值平滑"。我用报告给出的 3×3 矩阵直接验证：

```
矩阵（Marshal payoff）        HR-1 F   BIR-1 F   Continuation F
HR-1C                          0.68     0.56      0.45
HR-1.1C                        0.72     0.60      0.45
BIR-2U                         0.86     0.81      0.73

纯鞍点检测：(BIR-2U, Continuation) = 0.73  ← 行最大且列最小
纯策略 profile_metrics：value=0.73, lower=0.73, upper=0.73, duality gap = 0.0
```

精确答案是 `100% / 100%`，duality gap **恰好为 0**。然后我用仓库的 MWU solver 重跑，完整复现了报告的每一位数字：

```
max_it=100000 tol=0.001 → M=[0.0010, 0.0011, 0.9979]  F=[0.0014, 0.0030, 0.9956]  gap=0.0999pp
                              ↑ 99.79% BIR-2U            ↑ 99.56% Continuation      ↑ 报告值
```

所以报告的解释完全正确：那 0.44pp / 0.21pp 是 softmax + 时间平均的残留，没有任何战略含义。**一个 O(RC) 的纯鞍点预检就能直接返回精确解**，比调 tolerance 更有效也更可解释。

### 2.2 新发现：加大 `solver_iterations` 会让收敛**变慢**

报告 §7.3 建议"不要为了更小 numerical gap 增加 solver iterations"，但没给原因。根因在 `solver.py:359-364`：

```python
rate = math.sqrt(8.0 * math.log(max(2, row_count, column_count)) / config.max_iterations)
```

学习率与 `max_iterations` **反向耦合**。实测同一矩阵、同一 tolerance：

| `max_iterations` | 实际学习率 | 达到 tol=0.001 所需迭代 |
| ---: | ---: | ---: |
| 100,000 | 0.009375 | 96,300 |
| 1,000,000 | 0.002965 | **304,000**（3.2×） |

这是标准 MWU 的理论学习率（为跑满全程而设），但配上"提前收敛就退出"的用法就成了陷阱：**用户想调更准而加大 `--solver-iterations`，得到的是更慢的收敛**。这比报告的"不必要"更强——是**有害**。纯鞍点／支配预检可以彻底绕过它。

### 2.3 mixture conditioning 确实存在，但只接在最贵的 Agent 上

报告说默认 response 是"固定 Agent"。准确的机制是：`MixtureConditionedResponseOracle` **确实**会把当前 opponent mixture 冻结进候选参数（`policy_adapter.py:321-403`，还有 leaf hash 校验和 deferred 参数，工程质量很好）——但只在 `opponent_parameter is not None` 时生效，而该值来自：

```python
# planning_leaf.py — 只有两条规则
PlanningLeafRule(Role.FUGITIVE, "belief-rollout",  "continuation-count", "opponent_policies")
PlanningLeafRule(Role.MARSHAL,  "rollout-bir2u",   "...bir2u...",        "opponent_policies")
```

而 CLI 默认 response 是 `continuation-count` 和 BIR-2U，两者都**不在**这张表里 → `search_opponent_parameter()` 返回 `None` → mixture-independent。

**这就是问题的精确形状：唯一支持 mixture 条件化的 Agent，恰好是报告（正确地）判定成本过高的 rollout Agent。** 报告的结论对，但这个因果关系值得写清楚——因为它说明 `ParameterSearchResponseOracle` 不需要新建机制，只需要复用已经存在且已测试的 deferred-parameter 通道。

### 2.4 参数化缺口：三个 CLI 都有，而底层早就支持

报告 §4.7 只说了 PSRO CLI 和 tournament。实际分布是：

| 层 | 同一 Agent 多参数版本 | 证据 |
| --- | --- | --- |
| `run_registered_match()` | ✅ 完全支持 | 已有 `fugitive_parameters` / `marshal_parameters` |
| `PolicySpec` / `resolve_registered_policy` | ✅ 完全支持 | 有 `identifier_name=` |
| PSRO CLI | ⚠️ 每个 agent name 只能一份 | `AGENT=JSON` map，重复即报错 |
| behavior CLI | ⚠️ 同样限制 | `--agent-parameters AGENT=JSON` |
| tournament CLI | ❌ **完全没有参数覆盖** | `choices=` 锁死 registry 名，`TournamentConfig` 只存 `tuple[str, ...]` |

所以 tournament 今天**连一个非默认参数的 variant 都跑不了**，比报告描述的更严重。

好消息是：**底层什么都不缺，缺口纯粹在实验配置层。** 这让 `PolicyVariantSpec` 从"架构重构"降级为"薄适配层"，是整个方案里性价比最高的一项。

### 2.5 报告的参数搜索计划今天就能跑

我逐项核对了 §5.2 的 14 个建议参数与 registry allowlist：

```
Continuation-Count（_CONTINUATION_FUGITIVE_DEFAULTS）
  epsilon ✓  temperature ✓  continuation_weight ✓  concealment_weight ✓
  catch_risk_weight ✓  continuation_depth ✓  overpay_probability ✓        → 7/7

BIR-2U（_constructive_marshal_defaults）
  epsilon ✓  temperature ✓  terminal_bonus_scale ✓  manhunt_alpha ✓
  manhunt_epsilon ✓  max_guess_candidates ✓  particle_count ✓            → 7/7
```

**全部已暴露。** 这有两个后果：

1. 报告 §4.3 的 typed config 重构**不是**参数搜索的前置条件——第一轮 6–10 维搜索用现有 allowlist 就够。报告把它排在优先级 A 第 4 项偏高了，应降级到第二阶段。
2. 唯一的真实阻塞项就是 §2.4 的 CLI 表示问题。

### 2.6 新发现：Fugitive 有 1.83× 的性能空间，且卡在一个次要决策上

报告完全没有分析 Continuation Fugitive 的成本结构。我做了 profile（Continuation vs 便宜 Marshal，单局 5.92s）：

```
_defensive_features          3.786s  ← 占 64%
  └ PathBelief.solve()         345 次, 3.68s
     └ _count_constrained   12,698 次

choose_draw_pile             2.813s  ← 占 47%
  └ _hand_value                371 次, 2.807s
choose_fugitive_action       2.881s
```

**选择从哪个牌堆摸牌，和选择真正的行动一样贵。** 而摸牌只有约 3 个选项。原因是 `_hand_value`（`continuation_count.py:492`）为每个可能摸到的牌重建 synthetic observation，再对其**全部**候选动作调用 `_action_score_with_feature` → `_defensive_features` → `PathBelief.solve()`。

我先测了一个假设——256 条的 `_defence_cache` FIFO 上限在这个扫描里颠簸——**被证伪**：

```
cap=256 → 7.65s    cap=4096 → 7.72s    cap=65536 → 7.78s   （无差异）
```

命中率本来就有 88.5%，那 345 次 solve 是真正互异的 shadow。所以真正的杠杆是**减少需要防御特征的候选数**。实测在摸牌评估里跳过防御项（6 个配对 seed）：

```
baseline                  mean = 6.65s
draw-eval skips defence   mean = 3.64s      → 1.83× 加速
```

**这必须作为消融实验、而不是纯优化来做**——它改变摸牌决策，因此改变策略强度。但注意 `_public_shadow_after` 根本不读 `observation.hand`：防御特征只依赖 route 与候选动作。在摸牌评估里，"我可能摸到哪张牌"对防御项的影响完全来自它开启了哪些新动作。所以**先验上，摸牌层的防御项信息量很低，很可能是可以砍掉的**。

对方案的意义：这 1.83× 直接作用于 Continuation 的自对弈、消融和便宜 Marshal 对局。对 Continuation vs BIR-2U（28s/局，成本由 Marshal 主导）只省约 11%——我不夸大这一点。

### 2.7 成本实测：报告的方案负担得起（报告从未估算）

报告推荐参数搜索作为主线，却没给任何成本数字。实测（每格 4–6 局，方差大，只看量级）：

| 对局 | 单局均值 |
| --- | ---: |
| Continuation vs BIR-2U（2000 粒子，默认） | 27.9s |
| BIR-1 vs BIR-2U | 20.7s |
| HR-1 vs BIR-2U | 24.3s |
| Continuation vs support-catalogue | 7.6s |

**关键发现：`particle_count` 是一条现成的保真度阶梯**，而报告的 successive halving 计划没有利用它：

| BIR-2U `particle_count` | 单局均值 | 相对默认 |
| ---: | ---: | ---: |
| 2000（默认） | 27.9s | 1.0× |
| 512 | 7.1s | 3.9× |
| 256 | 7.9s | 3.5× |
| 128 | 4.8s | **5.8×** |
| 64 | 5.7s | 4.9× |

于是报告的 §5.3 协议可以这样落地（Fugitive 搜索，对手 ≈ 纯 BIR-2U）：

```
阶段 A  36 配置 × 24 配对 seed = 864 局  @128 粒子低保真   ≈ 1.2 CPU-h
阶段 B  前 8 名 × 100 局      = 800 局  @2000 粒子全保真  ≈ 6.2 CPU-h
阶段 C  前 2 名 × 320 配对    = 640 局  @2000 粒子确认    ≈ 5.0 CPU-h
                                                    合计 ≈ 12.4 CPU-h
                                          20 workers → 约 40 分钟 wall time
```

**方案完全负担得起。** 需要显式记录的风险：128 粒子的 BIR-2U 是**不同的策略**，不只是更快的同一策略；低保真筛选可能淘汰掉专门克制全保真 BIR-2U 的配置。所以阶段 B/C 必须回到 2000 粒子，且阶段 A 的保留比例要放宽（保 8 而非保 4）。

### 2.8 次要：`dense()` 是 cell 数的二次复杂度

`marshal_payoff()` 对 `entries` 做线性扫描，`dense()` 对每个 cell 调用一次 → O(cells²)。实测 400 cells = 0.030s，1600 cells = 0.369s，外推 100×100 ≈ 14s/次求解。

当前 3×3 规模下**完全不是问题**，我不建议现在改。只在参数搜索把策略池推到几十个以上时，加一个 `dict[PayoffPair, PayoffEstimate]` 索引即可（约 5 行）。列在延期项里。

---

## 3. 修改方案

排序原则：**先解除阻塞，再降低成本，最后才花 CPU 做实验。** 每一阶段都能独立交付并验证。

### P0 — 诊断与正确性（约 0.5 天，无实验成本）

**P0.1 纯鞍点／支配预检（解决 §2.1 + §2.2）**

在 `solver.py` 新增，不改动 MWU：

```python
def dominance_report(matrix, ...) -> DominanceReport:
    """报告每一侧被严格/弱支配的策略。"""

def pure_saddle_point(matrix) -> tuple[int, int] | None:
    """返回行最大且列最小的 cell；不存在则 None。O(RC)。"""
```

在 `MultiplicativeWeightsMetaSolver.solve()` 入口先调用 `pure_saddle_point`：命中就直接返回精确纯策略解（gap 恰为 0，`iterations=0`），跳过 MWU。这样报告 §4.6 关心的 `99.56%` 会变成干净的 `100%`。

同时把 `dominance_report` 接进 `checkpoint_summary()`，落实报告 §8.A.2。

*验收：* 用 §2.1 的 3×3 矩阵做回归测试，断言权重恰为 `[0,0,1]`、gap 恰为 `0.0`；另加一个无纯鞍点的矩阵，断言仍走 MWU 路径。

**P0.2 学习率脚注**

`MetaSolverConfig.max_iterations` 的 docstring 写明它同时决定默认学习率，加大它会**减慢**收敛；要更紧的 gap 应显式传 `learning_rate`。可选：`MetaSolverDiagnostics` 里已有 `learning_rate`，在 summary 中一并打印。

**P0.3 删除失效条目**

从任何沿用报告优先级清单的地方删掉"补齐 `CANONICAL_RULES.md`"。打包脚本的整洁性问题另开低优先级 issue。

---

### P1 — `PolicyVariantSpec`（约 1–2 天，解除全部阻塞）

这是报告 §4.7 的建议，我完全同意，并按 §2.4 的实测把范围收紧：**底层不动，只加实验配置层。**

新增 `src/fugitive/shared/policy_variant.py`：

```python
@dataclass(frozen=True, slots=True)
class PolicyVariantSpec:
    policy_id: str                    # 实验内稳定可读的纯策略名
    role: Role
    agent: str                        # registry ID
    profile: str = "default"
    parameters: Mapping[str, JSONValue] = field(default_factory=dict)

    def resolve(self) -> PolicySpec: ...        # → registry 唯一构造入口
    def to_dict(self) / from_dict(cls, ...)     # JSON 往返
```

约束（照报告 §4.7 的意思，但要显式实现）：

- `parameters` 的校验**全部委托** registry 的 `user_parameter_defaults` allowlist，**不复制 schema**；
- 同一份 variant 列表内 `policy_id` 唯一；
- 归一化后的完整 AgentSpec 相同但 `policy_id` 不同 → 报错（防止同一策略以两个名字进池，污染支配分析）。

接入三处，均为新增可选参数、保持向后兼容：

| 工具 | 改动 |
| --- | --- |
| PSRO CLI | 新增 `--fugitive-variants FILE.json` / `--marshal-variants FILE.json`，与现有 `--fugitive` 并存 |
| tournament CLI | 新增同名开关；`TournamentConfig` 增加 `fugitive_variants` / `marshal_variants` 字段（`tuple[str,...]` 保持不变以兼容 resume） |
| behavior CLI | 新增 `--variants FILE.json`，解除同一 backend 多参数版本的限制 |

*验收：* 一个 JSON 里放两个 `continuation-count` variant（`temperature` 不同），三个 CLI 都能把它们当成两个独立纯策略跑完并分别记录。

---

### P2 — Continuation 成本削减（约 1 天 + 1 次配对实验）

对应 §2.6，报告完全没有的条目。**必须当消融做，不能当优化偷偷合并。**

**P2.1** 给 `ContinuationCountFugitiveAgent` 加参数 `draw_evaluation_defence: bool = True`（进 registry allowlist，默认值保持当前行为，保证既有结论不变）。`False` 时 `_hand_value` 内跳过防御项。

**P2.2** 配对实验：`draw_evaluation_defence` True vs False，对手 BIR-2U，320 个配对 seed，报告 paired delta + bootstrap 区间 + wall time。

**P2.3** 决策规则（**提前冻结**）：

- 区间包含 0（无强度损失）→ 默认改为 `False`，全部下游实验拿到 1.83× 加速；
- 显著变弱 → 保持 `True`，但把这条测量写进文档，说明 47% 的成本花在摸牌上是**有意**付的；
- 显著变强 → 单独报告，这是个真结果。

*为什么排在参数搜索前面：* 如果默认能改成 `False`，P4 的每一个 Continuation 消融和自对弈都便宜 1.8 倍。反过来做就是白付这笔钱。

---

### P3 — 评分常数 typed config（约 1 天）

报告 §4.3，我同意但**降级**：按 §2.5，它不是参数搜索的前置条件。放在 P4 之后也行，放这里是因为它便宜且能让 P4 的第二轮扩维。

按报告的建议引入 `FugitiveScoringConfig` / `SprintCostConfig`，收拢 `progress_weight=2.0`、`terminal_weight=4.0`、`pass_bias=-0.35`、`hand_diversity_weight=0.25`、`cost_scale=2.75` 及 Sprint cost 权重。

采纳报告的克制态度：**默认值一律不变**，且第一轮不把全部常数加进 registry allowlist——只加已有的 7 个 + 第二轮真正要搜的 2–3 个。

*验收：* 一个"默认 config 产生与重构前逐位相同的动作分布"的回归测试（固定 seed 比对 distribution）。

---

### P4 — 参数搜索（约 12–25 CPU-h，见 §2.7）

依赖 P1（表示）、P2（成本）。按报告 §5.2/§5.3，加上 §2.7 的保真度阶梯。

**P4.1 Continuation vs 近乎纯 BIR-2U** — 搜 §2.5 那 7 个参数，三阶段协议见 §2.7。同一 opponent set 复用同一批 master seed。

**P4.2 BIR-2U vs 近乎纯 Continuation** — 同协议。`particle_count` 按报告 §5.2 的要求当作**策略近似质量 + 计算预算**双重参数，必须同时报告胜率、median wall time、ESS/world diversity 和单位计算收益——不能只按胜率排序。

**P4.3 Continuation 特征消融** — 报告 §8.B.3。`continuation_weight` / `concealment_weight` / `catch_risk_weight` 各自置 0，加上关掉候选相对归一化，隔离各项贡献。

**P4.4 fixed-Observation BIR-2U vs BIR-2S 重跑** — 报告 §4.4。我确认这一项**不需要任何代码改动**：两者是不同 agent name，现有 `--agent-parameters AGENT=JSON` 足够。属于"直接跑"，可与 P0/P1 并行。

**贯穿要求（报告 §6 结尾，我强烈同意）：** 搜索报告必须保存**完整 matchup 向量**，不能只存对 mixture 的平均分。能打破支配结构的策略，特征恰恰是"对当前主导对手强、对旧弱策略反而差"——只存平均分会把它筛掉。

---

### P5 — `ParameterSearchResponseOracle` + 2–3 代 PSRO

依赖 P4。按报告 §6 的接口，但按 §2.3 的发现**复用**而非新建机制：

```python
class ParameterSearchResponseOracle:
    def propose_response(self, request: ResponseOracleRequest) -> PolicySpec | None:
        opponent = request.opponent_mixture          # 已有
        candidates = generate_variants(...)          # P1 的 PolicyVariantSpec
        scores = evaluate_against_mixture(...)       # P4 的搜索协议
        return best_unique_policy(scores)            # 归一化 AgentSpec 去重
```

去重按报告的要求用**归一化后的完整 AgentSpec 哈希**，而不是 `policy_id`。双方都返回已存在 variant 时停止（`run_psro` 已有 `no_new_policies` 停止条件，直接复用）。

按报告 §7.4 跑 2–3 代，用报告 §7.5 的判据评估——我认为这套判据是整份报告最有价值的部分，原样采纳：

- 新策略是否打破当前严格支配关系；
- 是否出现专门克制 BIR-2U 或 Continuation、但对其他策略更弱的策略；
- meta-mixture 在 fresh opponent variants 上是否比任何单一纯策略更稳健；
- 加入 response 后 restricted lower/upper bound 是否实质变化。

**不**把"mixture 自对弈胜率是否高于最佳纯策略"当主要判据。

---

### 延期（同意报告 §8.D，加两项）

报告 §8.D 全部同意（rollout 在线强化、嵌套 planning、payoff cell 自适应加样、executor 统一、Fugitive 组合式重构、Bayesian HPO、CFR 系列）。补充两项：

- **§2.8 的 `dense()` 二次复杂度** — 策略池超过约 30×30 再加 pair→estimate 索引；
- **打包脚本** — §1.2 的真实剩余问题。

同意报告：标准 PSRO 配置继续排除 online-search Agent；rollout 保留为离线 teacher / selective search / action-regret evaluator / 蒸馏数据源。

---

## 4. 与报告优先级清单的差异

| 报告优先级 A | 我的处理 |
| --- | --- |
| 1. 补齐 `CANONICAL_RULES.md` | **删除** — 已存在，290 测试全过（§1.2） |
| 2. dominated-policy + pure-saddle 报告 | **提到 P0.1** — 并扩展为求解器预检，不只是报告层 |
| 3. `PolicyVariantSpec` | **P1** — 唯一真实阻塞项，范围收紧为薄适配层 |
| 4. 关键常数 typed config | **降级到 P3** — 7 个参数已暴露，不是搜索前置条件（§2.5） |
| 5. PSRO 排除 online-search | **同意，无需改动** |
| 6. 重跑 BIR-2U/BIR-2S 消融 | **P4.4** — 确认零代码改动，可立即并行 |
| — | **新增 P0.2** 学习率陷阱（§2.2） |
| — | **新增 P2** Continuation 1.83× 成本削减（§2.6） |
| — | **新增 P4 保真度阶梯**（§2.7） |

推荐执行顺序：

```
P0（诊断，0.5 天）
  ├─ 并行：P4.4（BIR-2U/BIR-2S 重跑，零代码改动）
  └─ P1（PolicyVariantSpec，1–2 天）── 解除阻塞
       └─ P2（Continuation 消融，1 天 + 320 局）── 降低成本
            └─ P3（typed config，1 天）
                 └─ P4（参数搜索，约 12–25 CPU-h）
                      └─ P5（Response Oracle + 2–3 代 PSRO）
```

---

## 5. 对报告整体评价的修订

报告的最终判断：

> 当前 PSRO 是一个已经通过中等规模实验证明可用的有限策略池分析器；它准确选出了 BIR-2U 与 Continuation，但当前策略池过于单调，尚未产生有价值的混合。

**这个判断我确认成立**，并且比报告自己给出的证据更强——精确 duality gap 为 0，不只是"接近"纯鞍点。

需要修订的是**瓶颈的完整描述**。报告说瓶颈是"策略空间缺少有针对性的反制策略"。更准确的是三层，且都不是 PSRO 数学层的问题：

1. **表示层** — 实验配置层无法表达同一 Agent 的多个参数版本，而底层早已支持（§2.4）；
2. **成本层** — Continuation 有 47% 的时间花在摸牌决策上，1.83× 可能是白送的（§2.6）；BIR-2U 的 `particle_count` 是一条未被使用的 5.8× 保真度阶梯（§2.7）；
3. **策略层** — 报告已正确识别的这一层。

报告只看到第 3 层。而第 1、2 层恰好是第 3 层的前提：不解决表示就没法生成 variant，不削减成本就要多付一倍 CPU。
