# Virtual Diapause Cell

A clock-centred research project for context-aware dormancy knowledge, molecular state estimation and, eventually, experimentally evaluated state-transition prediction.

**独立新项目，2026-10-01 建立。当前交付是模块定义、公开来源目录、知识记录契约与离线校验工具；尚未训练模型，也未证明细胞状态或未来响应预测能力。** FactorBridge 已归档，仅供方法参考，不是本项目依赖。

## 先读这些

| 目的 | 文档 |
| --- | --- |
| 和合作者讨论目标、模型与验收 | [项目说明](docs/PROJECT_DESCRIPTION_CN.md) |
| 审阅更广的生物学模块 | [14 个功能领域](docs/MODULES_CN.md) / [机器可读词典](knowledge/modules.json) |
| 理解 clock 与组织架构 | [架构与评价](docs/ARCHITECTURE_CN.md) |
| 核查论文与数据库 | [证据与来源](docs/SOURCES_CN.md) / [来源注册表](knowledge/sources.json) |
| 决定哪些数据真正可用 | [数据纳入目录](knowledge/datasets.json) |
| 借鉴旧项目、在同一服务器独立部署 | [FactorBridge 参考与服务器说明](docs/FACTORBRIDGE_AND_SERVER_CN.md) |
| 确认本次做了什么 | [v0.1 定义快照](docs/STATUS.md) |

## 定义

样本状态由 **背景、过程分支、分子进程位置、多个功能程序及其偏离、测量覆盖与不确定性** 共同描述。Diapause clock 是特定过程内的坐标，不是把物种、时长、深度和存活压成一个分数。

三个任务分别验收：

1. 知识：从来源定位实验事实，区分直接证据、迁移假设与未知。
2. 状态：从真实观测估计 clock 和 programme 数值；同时报告独立功能终点。
3. 转移：给定当前状态、时间间隔、条件与历史，预测后续分布；需要真实时间或干预数据。

知识范围可广，首个数值任务须窄：优先审计公开胚胎 diapause / reactivation 数据。昆虫 diapause、dauer、细胞 quiescence、植物种子及微生物 dormancy 分层记录，不共用无条件 clock。内部 killifish 继续锁定；论文结果、K13 和 M4 等派生对象也不自动导入训练。

## 当前可以运行

仅需 Python 3.10+，无 GPU、无第三方 Python 依赖：

```bash
python scripts/validate_catalogue.py
python -m unittest discover -s tests -v
```

校验引用、模块标识、GO 快照、证据审核状态与数据准入边界。通过只表示定义文件自洽，不代表生物学证据已人工审核或可开始训练。

更新 GO 锚点时显式保存一个新的快照目录，不覆盖旧版：

```bash
python scripts/fetch_go_anchors.py --output-dir outputs/go_refresh
```

当前 `knowledge/` 的文献笔记全部是待领域专家审阅的来源摘要，`training_eligible=false`。GO 术语仅用于检索与定义；不是已确认的基因成员集。

## 项目边界

不复制旧训练集、adapter、PCA 弱标签或 ageing 主导的混合训练分布。不建多 agent、RL 或 14 个独立 LLM。暂不下载大表达矩阵、不启动正式训练；先完成来源审核、数据准入与独立评价协议。公共仓库不保存内部数据、整篇受限论文或服务器日志。
