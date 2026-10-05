# Virtual Diapause Cell

**当前续跑：[CW1 定向修订：AutoDL 更新、进度与打包](docs/CW1_TARGETED_REVISION_CN.md)。** 冻结上一轮模型，校验复用 51 项；新增短格式身份证据推理、两条链重评、三视图残差读出及支持范围诊断。真实 CPU 检查已执行，新 GPU 推理由服务器启动后记录；不新增 LoRA、不查询预留。

**最新开发轮：[CW1 身份、相对表达与显式 clock–wave：AutoDL 启动、进度和打包](docs/CLOCK_IDENTITY_WAVE_CN.md)。** 沿用已授权开发样本；数值拟合、现有 Qwen 身份证据推理、逐 pool 评价与隐藏读出。本轮不重新运行 171 项，不查询预留，不新增领域 LoRA。服务器执行状态以实际 run 为准。下方保留此前轮次入口。

**上一轮入口：[冻结候选后的真实样本应用、知识重评与 AutoDL 启动](docs/PK3_APPLICATION_CN.md)。** 复用修复后的 PK2 权重；本轮不重新运行 171 条训练。预留查询先冻结，热应激原始 counts 待补，现有标准化表达仅作独立描述性替代。服务器执行由用户启动，结果以该次 run 的工件为准。下面保留此前轮次入口。

A biotime-centred model of diapause state, gene/TF waves and programme dynamics, grounded in context-specific biological evidence.

**独立新项目，2026-10-01 建立。当前推进 All-Module Alpha：复用 v0.4.0 数值模块，实际运行公开数据上的状态、候选 clock、gene/TF/programme waves、端点、转移及组级功能读出。Qwen GPU 分支由 AutoDL 实际运行后记录结果；整体科学能力仍未验证，expression-to-depth 不可用。** FactorBridge 已归档，仅供方法参考，不是本项目依赖。

**本轮入口：[AutoDL 一键启动、进度、打包和内部界面](docs/ALL_MODULE_ALPHA_CN.md) · [数据与语料审计](docs/ALPHA_DATA_AUDIT_CN.md)**。只使用开发划分；单模块性能不作为其他分支的运行门槛。

**当前 PK2 入口：[已有训练的定向修复与续跑](docs/PK2_TARGETED_REPAIR_CN.md) · [首次全队列执行](docs/PK2_FULL_TRAINING_CN.md)。** 修复领域语义误路由和逐 pool 外层 checkpoint 选择；兼容结果逐文件核验后复用。服务器运行由用户启动，本地测试不代表补跑已经执行。私有文件、链接与名单不在此仓库。[上一批抓取与PK1冻结服务](docs/PK2_SERVER_SETUP_CN.md)继续保留。

**PK1：[正式数值训练、进度与报告](docs/PK1_TRAINING_CN.md) · [私有样本准入与数据准备](docs/PK1_PREPARATION_CN.md)。** 3种子 K0–K5 数值实验，复用实际 Alpha adapter；执行状态和结果以实际 run 工件为准。

## 核心优先级 v0.2

**状态与维护 → Exit biotime → gene/TF waves → programme/regulon waves** 构成研究主线。DC01–DC04 是核心任务，M01–M14 是功能解释层；细胞类型异步、Entry–Exit 逆转/持续变化和组成效应贯穿评价。Clock 与 programme 同时验收，不能只优化一个进程分数。

[核心优先级与 thesis 方法衔接](docs/CORE_PRIORITIES_CN.md) · [机器可读任务定义](knowledge/core_tasks.json)

## 先读这些

| 目的 | 文档 |
| --- | --- |
| 和合作者讨论目标、模型与验收 | [项目说明](docs/PROJECT_DESCRIPTION_CN.md) |
| 审阅更广的生物学模块 | [14 个功能领域](docs/MODULES_CN.md) / [机器可读词典](knowledge/modules.json) |
| 理解 clock 与组织架构 | [架构与评价](docs/ARCHITECTURE_CN.md) |
| 核查论文与数据库 | [证据与来源](docs/SOURCES_CN.md) / [来源注册表](knowledge/sources.json) |
| 决定哪些数据真正可用 | [数据纳入目录](knowledge/datasets.json) |
| 借鉴旧项目、在同一服务器独立部署 | [FactorBridge 参考与服务器说明](docs/FACTORBRIDGE_AND_SERVER_CN.md) |
| 确认本次做了什么 | [当前实施状态](docs/STATUS.md) |

## 定义

样本状态由 **背景、过程分支、分子进程位置、多个功能程序及其偏离、测量覆盖与不确定性** 共同描述。Diapause clock 是特定过程内的坐标，不是把物种、时长、深度和存活压成一个分数。

三个任务分别验收：

1. 知识：从来源定位实验事实，区分直接证据、迁移假设与未知。
2. 状态与动态：分别登记 biotime 坐标，恢复 gene/TF 与 programme waves 的形状和幅度，并保留细胞类型异步、参考偏离及独立功能终点。
3. 转移：给定当前状态、时间间隔、条件与历史，预测后续分布；需要真实时间或干预数据。

知识范围可广，数值任务按来源和背景分别审计。昆虫 diapause、dauer、细胞 quiescence、植物种子及微生物 dormancy 分层记录，不共用无条件 clock。内部 killifish 仅已批准的开发样本可用于拟合，其余预留角色保持锁定；论文结果、K13 和 M4 等派生对象不自动导入训练。

## 当前可以运行

[模块整合与真实试点](docs/MODULAR_INTEGRATION_CN.md) · [运行说明](docs/RUN_MODULAR_CN.md) · [实际结果](reports/v0.4.0/RESULTS_CN.md)

完整框架：

```bash
python -m pip install -e ".[test]"
python scripts/doctor.py
python -m pytest -q
python scripts/smoke.py --out work/synthetic_new_run
```

仅校验定义目录：

仅需 Python 3.10+，无 GPU、无第三方 Python 依赖：

```bash
python scripts/validate_catalogue.py
python -m unittest discover -s tests -p test_catalogue.py -v
```

校验引用、模块标识、GO 快照、证据审核状态与数据准入边界。通过只表示定义文件自洽，不代表生物学证据已人工审核或可开始训练。

更新 GO 锚点时显式保存一个新的快照目录，不覆盖旧版：

```bash
python scripts/fetch_go_anchors.py --output-dir outputs/go_refresh
```

当前 `knowledge/` 的文献笔记全部是待领域专家审阅的来源摘要，`training_eligible=false`。全局功能词典的 GO 锚点用于定义；首个公共试点另外按有校验值的 GO/WormBase 快照构造了 24 个直接成员 programme，其覆盖与限制见试点报告。

## 项目边界

不复制 FactorBridge 的旧训练集、adapter、PCA 弱标签或 ageing 主导的混合训练分布。不建多 agent、RL 或 14 个独立 LLM。数据按显式来源队列下载，按任务单独准入；不将 accession、下载记录或规划任务计为已完成训练。公共仓库不保存内部数据、整篇受限论文或服务器日志。
