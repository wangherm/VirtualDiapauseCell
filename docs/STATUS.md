# 实施状态

2026-10-05 CW1 开发轮：[启动、进度与打包](CLOCK_IDENTITY_WAVE_CN.md)。新增身份证据推理、隐藏目标隔离后的参考重拟合、C0/C1/C2、同身份 Ridge/均值基线、逐 pool 留出、语义图对照、gene/TF RNA waves 与固定模型压力测试。152 项软件测试通过；本地真实开发数据已执行 CPU 检查，Qwen 与领域语义工件仍需服务器执行。本次无新 LoRA，无预留查询，无新 depth 或未来预测能力。服务器任务尚未由本地提交；具体私有指标留在私有报告。

2026-10-04 应用轮：[运行说明](PK3_APPLICATION_CN.md)。接入冻结候选、原 train 隐藏基因读出、预留/探索查询、完整可见字段撤证据评价、评价专用新论文家族和私有保存结果浏览器。热应激 counts 待补，归一化历史表达仅作独立描述性替代。软件测试和实际开发输入重现检查在本地完成；本轮服务器查询、Qwen 重评与长期服务尚未执行。具体预留结果及私有名单不在公开仓库。

2026-10-04 定向修复：[启动与审计说明](PK2_TARGETED_REPAIR_CN.md)。修正清单 `new_domain/shuffled_new_domain` 被执行器当作零语义的缺陷；未知条件现在报错。领域缓存必须匹配实际选定 adapter、特征顺序和打乱排列，训练前和重载后分别检查真实前向作用。逐 pool 任务固定最后一步，不再根据外层 pool 选择 checkpoint；旧结果只作开发敏感性证据。新增严格限定来源版本的复用流程，原训练目录保留，新目录补跑 24 条语义与 18 条 pool 任务，已有 Qwen 复用并补充证据使用评价。预留查询仍未启用，修复后的服务器执行尚待用户启动。以下是之前各次交付当时的记录。

2026-10-04：[PK2全队列执行器](PK2_FULL_TRAINING_CN.md)已接入同一代码路径，171逻辑任务全部启用。已核验上一批报告：六个来源抓取完成、PK1服务重启查询通过，但没有新PK2训练。本地实际完成127条数值任务的两步工程联调和新快照83次查询×2次重启；这不是3000/6000步结果。6条Qwen及36条语义任务因本地无CUDA未执行；2条25%学习曲线因该子集缺锚点被阻塞。新文献、校验与等价复用的后续改动另外执行针对性检查。AutoDL正式队列尚未由本地提交，研究能力仍未验证。

新Qwen语料250条，含30篇新原始论文的摘要级弱参考；独立论文家族和800–1500条目标仍未完成。expression-to-depth无匹配cohort，SMAD2缺样本矩阵，NHDF已完成技术重复合并后的描述性插值比较，仍受共同归一化限制，逐项记录。预留角色未解锁。

2026-10-03 PK2 更新：[抓取与部署](PK2_SERVER_SETUP_CN.md)。已实现并实际解析六个新增公共研究，来源文件逐一锁定 SHA256；已实现冻结 PK1 开发权重的 loopback 服务与独立进程重启验证。AutoDL 执行由用户启动。PK2 多研究训练、R0–R6 和新 Qwen 扩训尚未执行。以下保留历史轮次的当时状态，不代表现在所有能力仍为 not_run。

## 历史：All-Module Alpha

2026-10-02 新进展：全模块任务调度、真实数值分支、Qwen 一轮 SFT 与语义对照入口、内部 experimental API/界面已经接入同一代码路径。数值分支已在本地 CPU 实际执行，GPU 分支等待用户在 AutoDL 运行；不可将本地总体写成全部完成。

| 新工作 | 当前实际状态 |
|---|---|
| GSE288723 | 24 个 train/validation pools；旧 12 个 test 未读取为 Alpha 数值输入 |
| GSE291659 | 32 个 GEO 条目、30 列矩阵核对完成；23 个开发 pools；7 个保留列不转数值 |
| 状态＋局部 clock | 两个 context 各 500 步；clock 为 train-only reference-derived 坐标；保存重载完成 |
| 状态读出、gene/TF/programme curves | 实际拟合；19909 gene、433 TF RNA、24 GO programme；无 regulon 边表 |
| 端点＋释放转移 | 观测输入与冻结状态输出两条路径均拟合和评价；共享对照不跨 split |
| 功能 | Table S1 原始群体计数核对；24 个开发观测拟合组协议读出；expression-to-depth 无匹配标签 |
| 知识 | 27 条有源弱参考：18 train/9 validation，三个论文家族；不宣称完成全部领域知识 |
| GPU | 新 LoRA smoke/训练/回答评价/adapter 向量/语义对照尚未本地执行，提供 AutoDL screen 启动脚本 |
| 内部接口 | 真实保存模型的 ASGI 与 loopback HTTP 请求已执行；临时检查服务已关闭；未开启 AutoDL 服务 |
| 总体 | partial / science unvalidated；不因数值模型弱于某个基线而停下其他分支 |

[运行与进度](ALL_MODULE_ALPHA_CN.md) · [准入与边界](ALPHA_DATA_AUDIT_CN.md) · [本地成绩](../reports/all_module_alpha/LOCAL_RESULT_CN.md)

## 历史：v0.4.0 首个重构 pilot

最后核验：2026-10-02。科学定义沿用 v0.2 的 DC01–DC04、M01–M14 和五类不同 biotime 坐标；正式数值代码只有 `src/vdc/` 一条路径。

| 维度 | 实际状态 |
| --- | --- |
| 代码 | 数据、知识、状态、waves、响应、单步转移、功能读出、评价与本地发布入口已整合 |
| 软件执行 | 本地 CPU 73 项测试通过；合成串联通过 |
| 真实数据 | GSE288723 54 个样本文件已下载；36 个未加 UV 的 dauer/exit 样本用于首次 programme 试点 |
| 真实模型 | 小型重构模型 50 步、保存重载、保留样本评价；四种历史的实际时间一阶曲线与残差 |
| 初步效果 | 遮盖 MSE 比均值低约 3%，但 MAE 略差，整体不如保留输入；没有稳定模型优势 |
| 分子 clock / biotime waves | 合成测试通过；真实训练 not_run，无准入分子标签 |
| 真实扰动 / depth | not_run / unavailable；22 条候选不代表已下载训练集 |
| Qwen | GPU smoke、真实训练和语义消融全部 not_run |
| 科学批准 / 服务 | not_approved / not_run |
| AutoDL | 未连接或运行，提供服务器运行说明 |
| holdout | 内部 killifish 必须为 locked_test；旧公共 VAL/L4-OOD 未转为训练 |

详见 [本轮结果](../reports/v0.4.0/RESULTS_CN.md)、[模块整合](MODULAR_INTEGRATION_CN.md)、[运行说明](RUN_MODULAR_CN.md)。此前定义版本由 Git 历史和本地发布包保留。
