# Fugitive Strategy Lab

这是一个针对双人桌游 *Fugitive* 的整局规则引擎、随机策略 baseline 和本地交互界面。
当前活动代码只保留第一版规则所需的最小实现：不使用事件牌，Pass 不额外摸牌，所有模拟均运行到规则定义的胜负。

规则说明见 [CANONICAL_RULES.md](docs/rules/CANONICAL_RULES.md)。

## 当前 Baseline

| ID | 名称 | 说明 |
| --- | --- | --- |
| `hierarchical-random` | HR-1 | 按摸牌、Pass/Play、Hideout、Sprint 支付和猜测规模分层随机，避免动作编码数量造成偏差。 |
| `belief-informed-random` | BIR-1 | 只使用玩家 Observation，在信息集内通过手牌价值或粒子 belief 对随机动作加权。 |

两个角色都有各自独立的 HR-1 和 BIR-1 Agent。Web 默认运行 BIR-1 对 BIR-1，也可以任意组合或由人类接替一方。

## 安装与运行

需要 Python 3.11 或更高版本。运行时代码只使用标准库；测试需要 pytest。

```powershell
python -m pip install -e ".[test]"
python -m fugitive --host 127.0.0.1 --port 8000
```

也可以使用安装后的任一入口：

```powershell
fugitive --host 127.0.0.1 --port 8000
fugitive-web --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000`。界面支持玩家对 Agent、Agent 对 Agent，以及全知、Fugitive、Marshal、公开四种观战视角。

## 测试

```powershell
python -m pytest
```

可复现实验使用独立的 `fugitive.experiment` 模块。`run_registered_experiment`
从一个 master seed 域分离出牌堆、Fugitive 和 Marshal 三个随机流，并生成含
完整 action trace 的 JSON manifest；`replay_manifest` 会逐动作校验重放结果。
可选的 `max_decisions` 只属于实验 watchdog：达到限制时结果是 `truncated`，
没有虚构赢家。默认值为 `None`，因此正式模拟仍运行到规则定义的胜负。

活动测试覆盖规则引擎、约束 belief、粒子 belief、HR-1、BIR-1、Agent registry 和 Web API。牌堆与 Agent 随机数分别使用注入的 seed，便于复现实验；未指定 seed 的 Web 对局使用新的随机 seed。

## 历史实现

旧启发式、搜索、训练、Tournament、Mini game、实验结果和算法教程已原样移动到 [`archive/legacy_v1`](archive/legacy_v1/README.md)。它们仅供课程回顾，不属于当前包入口，也不参与活动测试。
