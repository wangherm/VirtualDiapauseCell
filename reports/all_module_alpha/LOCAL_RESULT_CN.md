# All-Module Alpha：本地数值结果

日期：2026-10-02。实际执行环境为 Windows / Python 3.12 / CPU torch 2.14.1。正式首轮配置为 seed 42，状态模型各 500 步，其余曲线/Ridge 为闭式拟合。结果来自 `runs/alpha_local_20261002_v1`；原始权重、预测、评价、reload 与请求文件保存在本地 run，未把它们冒充 GitHub 自带预训练模型。

整体为 **partial / unvalidated**。两个独立公开 RNA context，所有评价均为研究内开发 validation；原 GSE288723 test、内部 killifish、旧 VAL/L4-OOD 未参与本轮拟合和选择。

| 任务 | 实际模型结果 | 对照 | 解释 |
|---|---:|---:|---|
| Dauer programme 隐藏重构 MSE | 0.00009330 | Ridge 0.00021378；均值 0.00034678 | 12 个 validation pools、54 个本轮人工遮盖值 |
| ARD programme 隐藏重构 MSE | 0.00010090 | Ridge 0.00028437；均值 0.00037117 | 7 个 validation pools、40 个遮盖值 |
| Dauer 状态条件分类 accuracy | latent 1.00 | programme Ridge 1.00；多数类 0.667 | 区分 0h 与已释放条件，不是独立生物学/深度标注 |
| 候选 clock clean-input MSE | 0.00825 | reference-derived 目标 | 从表达定义的局部轴蒸馏，不是对独立 clock 真值的误差 |
| Gene 一阶参考 MSE | 0.01366 | train 均值 0.01869 | log1p library-scaled 表达；同一研究；非精细波峰验证 |
| 基因型端点 delta MSE | 0.00018491 | 零效应 0.00094794 | 5 个验证对比共享对照；不是新靶点外推 |
| 状态重构后接端点 MSE | 0.00019729 | 直接观测输入 0.00018491 | 此处上游重构略损害端点预测 |
| 释放转移 MSE | 0.00036048 | last-state 0.00052547；时间＋历史 0.00016441 | 优于不变化，仍弱于简单时间历史基线 |
| 状态重构后接转移 MSE | 0.00026616 | 时间＋历史 0.00016441 | 比直接观测路线改善，但仍未超过该基线 |
| 组级 young adult fraction MSE | 0.02399 | train 均值 0.14979 | 12 个验证群体观测；3 个线性预测超出 [0,1]，原值保留，非校准概率 |

这些结果没有经过大网格、多随机种子或新独立测试。不能据此称已恢复稳定生物因素、已证明因果、已建立跨物种世界模型。

数值保存重载检查均通过。内部接口用实际权重完成状态、条件读出、programme residual、gene/TF 参考、端点、转移和组级功能调用。另启动临时 127.0.0.1 HTTP 服务完成真实请求后关闭；没有远程访问 AutoDL，也没有部署公网。

Qwen 正式 GPU 部分 **not_run locally**：启动脚本会重新校验固定基础权重，跑两步前后向/保存重载 smoke，再从新基础初始化完成 18 个训练问题的一轮 SFT，保存完整训练记录；之后在新进程做原始/领域模型问答和实际 adapter 向量提取。小语料的一轮训练不等于完整休眠知识覆盖，不人为扩充重复题来伪装样本规模。

两个明确未完成的数据能力：regulon 缺准入的 TF-target 边；expression-to-depth 缺 RNA 与功能群体对应。现有功能支路只能接协议/历史，不以 clock 或论文组均值填补这些缺口。

软件测试与启动命令见 [运行说明](../../docs/ALL_MODULE_ALPHA_CN.md)。完整成绩以每次 run 的 `module_status.json`、`REPORT_CN.md`、真实日志和 artifact 指纹为准；本文是该次本地开发结果的摘要。
