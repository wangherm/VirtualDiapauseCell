# 实施状态 v0.4.0

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
