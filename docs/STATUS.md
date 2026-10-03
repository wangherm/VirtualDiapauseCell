# 实施状态

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
