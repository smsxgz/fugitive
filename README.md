# Fugitive for OpenSpiel

这是一个面向博弈算法研究的 Fugitive 实现。游戏规则由 C++ 实现并注册为
OpenSpiel game；Python 只保留薄适配层和本地对局网页。

当前项目从原 Python 版本重新开始。旧引擎、自研 Agent、PSRO、实验程序和旧测试
已从工作树删除；需要查阅时使用 Git 历史，不再在仓库中保留第二份归档。

## 当前架构

| 路径 | 用途 |
| --- | --- |
| [`cpp/fugitive`](cpp/fugitive) | Fugitive 的权威 C++ 实现和 C++ 测试 |
| [`src/fugitive/web`](src/fugitive/web) | OpenSpiel 状态到现有网页协议的适配层 |
| [`integration`](integration) | 固定的 OpenSpiel 版本和最小 CMake 补丁 |
| [`third_party/open_spiel`](third_party/open_spiel) | 可重建、不会提交的 OpenSpiel 工作树 |

## 代码归属

除了下面明确列出的上游或外部资料，其余当前文件都由本项目维护。我们自己编写和
需要直接修改的文件按职责分为：

| 路径 | 本项目维护的内容 |
| --- | --- |
| [`cpp/fugitive/fugitive.h`](cpp/fugitive/fugitive.h)、[`fugitive.cc`](cpp/fugitive/fugitive.cc) | OpenSpiel game、state、规则、动作、信息状态和序列化接口 |
| [`cpp/fugitive/belief.h`](cpp/fugitive/belief.h)、[`belief.cc`](cpp/fugitive/belief.cc) | 只读取 Marshal information state 的 Route / Completion DP、加权边缘、隐藏历史采样、重放与 Manhunt evaluator |
| [`cpp/fugitive/fugitive_test.cc`](cpp/fugitive/fugitive_test.cc)、[`belief_test.cc`](cpp/fugitive/belief_test.cc) | C++ 规则、OpenSpiel 契约和 DP 核心语义测试 |
| [`cpp/fugitive/belief_experiment.cc`](cpp/fugitive/belief_experiment.cc)、[`baseline_experiment.cc`](cpp/fugitive/baseline_experiment.cc) | 完整牌组 belief profile，以及 L1/L2 配对 seed 对局实验 |
| [`src/fugitive`](src/fugitive) | Python 包入口，以及网页对 `pyspiel.State` 的薄适配层和静态前端 |
| [`tests`](tests) | Python/OpenSpiel 集成测试和网页适配测试 |
| [`examples/train_outcome_sampling.py`](examples/train_outcome_sampling.py) | 调用 OpenSpiel C++ outcome-sampling MCCFR 的最小实验入口 |
| [`scripts`](scripts) | 环境准备、OpenSpiel 构建和网页启动脚本 |
| [`integration`](integration) | 固定上游版本，以及把 Fugitive 加入上游 CMake 的项目补丁 |
| [`environment.yml`](environment.yml)、[`requirements.txt`](requirements.txt)、[`pyproject.toml`](pyproject.toml) | 本项目环境和打包配置 |
| [`docs`](docs) 与本 README | 规则整理、集成设计和使用说明 |

不属于本项目原创代码或内容的部分：

- `third_party/open_spiel` 是 `scripts/setup_openspiel.sh` 下载的 Google DeepMind
  OpenSpiel 上游源码，不提交到本仓库，也不应直接在其中维护 Fugitive 代码；
- `third_party/open_spiel/open_spiel/games/fugitive` 只是指向
  `cpp/fugitive` 的相对软链接，因此链接下看到的 Fugitive 文件仍是本项目代码；
- `docs/rules/sources/fugitive-first-edition-rulebook.pdf` 是外部规则书原文，仅作为
  规则来源保存，不是本项目编写的文档；
- `integration/open_spiel-v2.0.1.patch` 虽然作用于上游 CMake 文件，但补丁本身由
  本项目维护。

项目固定使用 OpenSpiel `v2.0.1`，commit
`112b77704631fc2ce7ad8e4581f6ca09798ce15a`。`setup` 脚本会把
`cpp/fugitive` 软链接到 OpenSpiel 的 `open_spiel/games/fugitive`，再应用只包含
CMake 注册项的补丁。项目源文件因此仍由本仓库维护，第三方 checkout 可以随时删除
并重建。

## 快速开始

需要 Git、Conda，以及 Linux C++ 构建环境。下面的命令会创建名为
`openspiel` 的 Conda 环境、下载固定版本的 OpenSpiel 及其必要依赖，然后编译
Fugitive、belief DP 测试、实验程序和 `pyspiel`：

```bash
bash scripts/setup_openspiel.sh
bash scripts/build_openspiel.sh
```

编译后可直接确认游戏已注册：

```bash
PYTHONPATH="$PWD/src:$PWD/third_party/open_spiel:$PWD/third_party/open_spiel/build/python" \
  conda run -n openspiel python -c \
  'import pyspiel; print(pyspiel.load_game("fugitive(max_rounds=50)"))'
```

启动保留的本地对局网页：

```bash
bash scripts/run_web.sh --host 127.0.0.1 --port 8000
```

然后访问 <http://127.0.0.1:8000>。服务器没有认证，会话只保存在内存中；除非位于
可信网络，否则不要绑定到 `0.0.0.0`。

如需安装 Python 命令入口，可在构建后执行：

```bash
conda run -n openspiel python -m pip install -e .
```

`fugitive` 与 `fugitive-web` 是同一个网页服务入口。运行它们时仍需让 Python
找到本地编译的 `pyspiel`；日常开发优先使用 `scripts/run_web.sh`，它会设置正确的
`PYTHONPATH`。

## 规则与研究边界

基础规则见 [`docs/rules/CANONICAL_RULES.md`](docs/rules/CANONICAL_RULES.md)。
OpenSpiel 要求 `MaxGameLength()` 是有限上界，而旧规则没有回合上限，因此当前研究
variant 新增 `max_rounds` 参数，默认 50。达到上限仍未分出胜负时返回零和收益
`[0, 0]`，即平局。这个 horizon 是计算模型约定，不是原桌游规则。

该 game 声明为双人、顺序、显式随机、不完美信息、零和、终局奖励。洗牌和摸牌使用
OpenSpiel chance node；信息状态与 observation 由游戏实现提供；状态序列化沿用
OpenSpiel 的动作历史机制。复合的“打出 Hideout + Sprint”与“多数字猜测”被编码成
多个微步骤，以避免枚举指数级组合动作。

### 算法选择

OpenSpiel 的算法清单很长，但“仓库中有实现”不等于“适合当前 Fugitive”。按当前
接口和实际冒烟结果分为：

| 结论 | 算法 | 原因 |
| --- | --- | --- |
| 现在可直接运行 | **C++ outcome-sampling MCCFR** | 每次更新只采样终局轨迹；支持顺序、显式 chance、不完美信息、双人零和游戏，是当前首选 |
| 接口兼容但当前不实用 | external-sampling MCCFR | 会在更新方节点遍历全部动作；本机 `max_rounds=1` 的一次 C++ 迭代 30 秒仍未完成 |
| 只适合未来的缩小游戏 | CFR、CFR+、Discounted CFR、sequence-form LP、exact best response / exploitability | 都需要遍历或物化巨大博弈树；本机 `max_rounds=1` 的 CFR/CFR+ 构造和 NashConv 分别在 15 秒内未完成 |
| 是框架，不是现成 solver | PSRO / PSRO v2 | 还必须提供可承受的 best-response oracle；精确 oracle 会枚举全树，RL oracle 又需要 tensor |
| 当前不能正确运行 | IS-MCTS | Marshal 侧已有可重放采样器，但 OpenSpiel 接口要求双方视角；Fugitive 侧采样器和 `ResampleFromInfostate` 尚未实现 |
| 当前不能运行 | Deep CFR、NFSP、DQN、PPO、AlphaZero 等神经方法 | OpenSpiel RL 环境要求 observation/information-state tensor；当前 game 只有字符串接口 |
| 不应作为公平策略 | 普通 MCTS | 搜索真实 `State` 会看到玩家本不应知道的私有信息，只能作全知调试基线 |

最值得先跑的是 C++ outcome-sampling MCCFR。它虽然按轨迹采样，但仍会为访问过的
information state 建表，因此迭代数增加时内存近似随新信息集增长。应先用几十到几百
次迭代验证流程，再监控内存扩大预算；不要把短跑的经验自博弈收益当成收敛证明。

仓库提供一个调用 OpenSpiel 官方 C++ outcome-sampling MCCFR solver 的薄示例：

```bash
PYTHONPATH=src:third_party/open_spiel:third_party/open_spiel/build/python \
  conda run -n openspiel python examples/train_outcome_sampling.py \
  --iterations 100 --evaluation-games 20 --max-rounds 50 --seed 0
```

它训练平均策略并做采样自博弈评估，不会执行完整树 CFR 或精确 exploitability。
OpenSpiel 官方维护的完整算法列表见
<https://openspiel.readthedocs.io/en/latest/algorithms.html>；上表是针对本 game 的筛选，
不是 OpenSpiel 全部功能的列表。

Route / Completion DP 的完整牌组可行性实验可直接运行：

```bash
third_party/open_spiel/build/games/fugitive_belief_experiment \
  --samples_per_bucket 32 --seed_start 0 --max_seeds 1000
```

也可以扫描固定 seed 范围内的每个 Marshal 猜测边界，并分别汇总高 Route support、
高隐藏 Sprint 和普通深局：

```bash
third_party/open_spiel/build/games/fugitive_belief_experiment \
  --mode sweep --seed_start 0 --max_seeds 40
```

默认 `--replay_samples 1`，即每个记录状态采样并重放一次；只测计数性能时可显式设为
`--replay_samples 0`。

程序输出 JSON Lines。Route 层给出满足公开路线、历史猜测和已知抽牌截止时间的路线
支持上界及仅用于诊断的 route-support 频率；Completion 层再分配隐藏 Sprint 身份和
Fugitive 隐藏抽牌，并输出 history-weighted 单牌边缘。`uniform_consistent_history_mass`
是不含 Fugitive 策略概率的相容 chance-history 质量，运行时用 `long double` 近似累计，
不能称为真实后验。

实验默认对每个收集到的信息状态按该质量采样一个完整隐藏历史，由 C++ engine 逐动作
检查合法性并要求 Marshal information string 精确相等。这个 Marshal 侧采样器已经
可用，但不能冒充双方通用的 OpenSpiel `ResampleFromInfostate`；后者还需要独立实现
Fugitive 视角。

Manhunt 有两种 evaluator：小状态精确参考会按
`(route_position, exact_sprint_cards)` 枚举成功观察；完整对局版本先采样完整隐藏历史，
再精确求解有限的经验 belief 树。后者的上下界只属于这批粒子，不是底层 belief 的
统计置信区间。可在真实 Manhunt checkpoint 上同时 profile 两者：

```bash
third_party/open_spiel/build/games/fugitive_belief_experiment \
  --samples_per_bucket 1 --seed_start 1 --max_seeds 1 --replay_samples 0 \
  --manhunt_completion_calls 100000 --manhunt_particles 256
```

跨采样 seed 的粒子收敛实验使用同一 RNG 前缀配对不同粒子数；最大粒子档只是参考，
不是真值：

```bash
third_party/open_spiel/build/games/fugitive_belief_experiment \
  --mode manhunt_convergence --manhunt_checkpoints 16 \
  --manhunt_sample_seeds 16 --manhunt_particle_counts 64,128,256,512 \
  --seed_start 0 --max_seeds 1000
```

16 个去重 Manhunt 入口、每个 16 条采样 seed 的初筛中，64 粒子与 512 粒子的配对
首猜一致率只有 85.5%，value 平均绝对差为 0.0727；256 粒子也只有 93.0% 和 0.0282。
全部 1,024 次经验 belief 求解都精确，但 512 粒子 p95 约 9.9 s，且多数状态的平均 value
仍随粒子数增加而下降。因此 64 不能作为已通过收敛检查的默认值，512 也尚不能称为
收敛真值；进入 N14 前应先复用 Completion 做批量采样，再把参考粒子数向上推。

L1 Fugitive 对 L2 Marshal 的 guard/noguard 配对实验：

```bash
third_party/open_spiel/build/games/fugitive_baseline_experiment \
  --games 1000 --seed_start 0 --max_rounds 50 --manhunt_particles 64 \
  --guess_mode argmax_only --low_exhausted lift --dead_card_sprints 1
```

Fugitive-L1 会在 42 已经是合法动作时优先打出它，并复用 engine 的合法动作与最少
Sprint 选择。`--dead_card_sprints K` 允许它在普通、非 42 的出牌中额外倾倒至多 K 张
已经满足 `card <= previous_hideout` 的死牌；K 只能是 0、1、2，默认 0 以保持旧 L1
实验可复现。本轮主变体 K=1 优先消耗价值 1 的奇数死牌，选好后按数字升序提交，
每一步仍以 engine 的 `LegalActions()` 为准。

Marshal guard 先构造实际 unrestricted 猜测集；若它不能直接覆盖全部隐藏位置，则优先
限制在 `<30`。低牌全部为零概率时，`--low_exhausted lift` 放开限制，`wait` 则明确
猜一张零概率牌作为策略性等待；后者不是 engine Pass。JSON 中
`unrestricted_argmax_ge_30_turns` 记录阈值进入 belief 决策的次数，
`guard_restriction` 只统计 `<30` 重建真正改变猜测集的次数。

Marshal 的 `--guess_mode` 有三种：`argmax_only` 只提交一个正概率 argmax；
`certain_only` 在存在概率为 1 的牌时只提交全部确定项，否则退化为单 argmax；默认的
`certain_plus_argmax` 提交全部确定项，并在尚未覆盖全部隐藏位置时追加一个非确定
argmax。旧 `--add_certain_guesses 0|1` 仍作为前后两种模式的兼容别名。

相同的 seed 0--999 上，旧 L1
（K=0）的最高边缘从未到达 30，guard 也从未改变动作；K=1 后，guard 一侧的 7,027 个
普通猜测节点中有 104 个 unrestricted argmax 不小于 30，并有 25 次 `<30` 重建真正
改变动作。K=1 的 guard/noguard 分别为 `971 M / 29 F` 和 `973 M / 27 F`，K=2 分别为
`972 M / 28 F` 和 `971 M / 29 F`，均无 timeout。

扩大到 10,000 seed 后，K=1 guard/noguard 分别为 `9743 M / 257 F` 和
`9736 M / 264 F`。47 个配对 seed 的胜者不同，其中 guard 独赢 27 局、noguard 独赢
20 局；净差很小，且 1,000 seed 时方向相反。因此当前结论只是“死牌 Sprint 已让 N11
进入真实决策路径”，还不能据此声称 guard 有稳定胜率优势。`wait` 在 50/100/200 的
同一 7 局中持续到 horizon，属于策略循环，不作为主结果。

强制赌博消融固定 K=1、`low_exhausted=lift`，并以不受 30 guard 干预的 noguard 一侧
作为主口径。10,000 seed 中，`argmax_only / certain_only /
certain_plus_argmax` 的 Marshal 胜局分别为 `9736 / 9930 / 9998`，平均回合为
`7.00 / 6.33 / 5.01`。plus 模式的 8,872 次强制赌博有 2,930 次失败，丢掉 3,504 张
本可确定揭示的牌，但其快速 cover-all 收益在当前对手上明显大于局部损失。因此强制
赌博不是免费动作，却是当前 L1-smoke matchup 中最强的三种变体；这不构成对更强
Fugitive 或最优策略的证明。

实现、动作协议、序列化和算法选择的详细说明见
[`docs/OPEN_SPIEL_INTEGRATION.md`](docs/OPEN_SPIEL_INTEGRATION.md)。
