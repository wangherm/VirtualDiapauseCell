# Alpha 数据准入与知识边界

## 数值输入

1. **GSE288723**：复用 SHA256 锁定的 Salmon estimated counts，保留小数。原 54 列中，只解析非 UV 的 replicate 1/2（24 个池、19909 基因）；replicate 3 的 12 个旧 test 保留，另外 UV/L3 未准入。本轮不是重新划分原数据。
2. **GSE291659**：[GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE291659) 有 32 个样本条目，实际 deposited matrix 只有 **30 列**。缺 `hlh30_REC_REP4`、`daf1_48_REP3`，不补零、不造观测。使用 replicate 1/2 训练（16 个池）、replicate 3 验证（7 个池），replicate 4 的 7 列保留。表达文件自称 DESeq2 normalized counts，因此按 linear abundance 处理，不能称原始 counts。标题/genotype 到矩阵库名显式映射；不进行跨物种 ID 猜测。
3. 端点为 N2、hlh-30(tm1978)、daf-1(m40)、double genotype 在 ARD/refed 背景下的差异。以同条件、同 replicate block 的 N2 为对照；共享对照与结果始终同 split。共 12 个训练对比、5 个验证对比，不当 17 个独立实验。
4. GSE288723 的 0h→6/24h 转移为条件匹配的 RNA pools，共 8 个训练、8 个验证组对，不是同一动物的跟踪。维护历史独立作为输入，未来表达/未来 clock 不进预测输入。

全部 replicate-index blocks 都是保守分组，资料没有证明跨条件的原始 culture lineage；本轮只做研究内开发评价，不宣称跨研究泛化。

## Clock 与 curves

先按基因 ID 的固定 SHA256 分配 20% gene readout panel，再仅从 train 的其余基因选择最多 512 个高方差基因。标准化、0h/24h 锚点与方向均只用 train。四种维护历史等权。方向和缩放保存为 `axis.npz`，完整 gene order、fit IDs、排除的读出基因和 SHA256 单独记录。

所有曲线使用真实表达或 programme 分数，degree=1。TF 身份来自锁定的 WormBase GAF 中直接 GO:0003700 positive annotation；本次实际映射 433 个 TF 基因。测量是 TF 的 RNA 丰度，不是蛋白活性。GO programme 保持原名，不改名为 regulon。当前审查的 GAF 是 gene→GO 关系，不能提供 TF→target 边，故 regulon 子项仍 blocked_data。

独立读出只意味着 gene membership 与建轴集合不重叠。共享的 library normalization 和 programme rank background 仍存在，不能把这个检验当成完全独立的分子测量。

## 功能读出

来源：[Totska et al., Age Deceleration and Reversal Gene Patterns in Dauer Diapause](https://pmc.ncbi.nlm.nih.gov/articles/PMC12686545/)，Table S1，`ACEL-24-e70253-s009.xlsx`，CC BY 4.0。

实际下载 Europe PMC supplement ZIP，并只读检查 `Fig1B`。从 C:E 提取 Dauer/L4/young adult 原始计数，与 H 列总数逐行核对；不用均值、T-test 或含 `#DIV/0!` 的统计单元格做标签。具体 cell locator、原始计数、SHA256、协议记录在 `data/curated/functional_fig1b.json`。两个保守 replicate-order block 各 12 个群体观测（history 1/10/20d × release 48/72/96/120h），第三个 block 留出不用于开发。

RNA history 为 1/4/15/30d，功能 history 为 1/10/20d，且没有 RNA pool 与功能 assay cohort 对应表。因此只训练 **组协议/历史 → young adult fraction**。不复制组均值成细胞标签，不拟合 expression-to-function，不把该比例命名为 depth。

## 知识语料

`knowledge/alpha/corpus.jsonl` 为逐条从已读取原文选择、核对的 **source-curated weak reference**；不是 Qwen 自产 gold，也未声称经过独立领域专家审核。

- Train：PMC11398561（mTOR–BRD4，10 条）、PMC10866708（FOXO1/lipid/不同细胞背景，8 条）。
- Validation：PMC11146600（miRNA/Dgcr8，9 条）；整篇 family 留出，含 uncertain。
- 问题明确提供短事实证据、来源和 context。这是 evidence-grounded extraction 测试，不冒充闭卷知识记忆。
- 格式、context、source、uncertain 可作精确字段检查；自由答案 exact match 只是保守代理，原始生成答案必须保留供专家复核，不由模型给自己判正确。
- 两个 train 来源为 CC BY 4.0，validation 来源为 CC BY-NC 4.0。条目是带归属的改写；派生语料/adapter 初期限研究用途，保留来源许可说明。
- GSE288723 结果论文、GSE291659 同家族结果、内部材料、原 VAL/L4-OOD 不进入语料。特别是 PMC5143278/GSE81285 虽然高度相关，但旧目录为 VAL，因此整篇排除。
- 旧 `evidence_seed` 的 unreviewed 记录没有整体改成 reviewed。语料仅包含这次有明确 locator 与核对依据的 27 条。
- 公共论文可能早已出现于 Qwen 的预训练，无法完全审计；不把 source-family holdout 当作从未见过知识的证明。

Qwen 向量只编码固定 GO 功能描述，不含目标实验 DEG、末态、验证结果或 clean reference。提取向量时显式加载新领域 adapter，并记录 adapter 文件 hash；状态训练读取实际缓存，同时运行同容量零向量对照。
