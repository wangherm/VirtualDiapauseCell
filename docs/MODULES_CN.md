# 功能解释层词典 v0.2

2026-10-02：M01–M14 及其 42 个子模块保留，现归入功能解释层；顶层研究任务是 [DC01–DC04](CORE_PRIORITIES_CN.md)。功能类别说明 wave 涉及什么，不能替代 biotime、波形或 programme 异步。

这 14 个领域是项目提出的可修订目录，不是数据库认可的统一 diapause 模型。42 个子模块目前均无已审定 gene membership；GO 锚点用于查找词汇。clock、阶段和功能结局另行定义。

| ID | 功能领域 | 核心问题 |
| --- | --- | --- |
| M01 | 环境感知与内分泌决策 | 如何感知季节、营养与释放条件，并决定进入或恢复？ |
| M02 | 细胞周期与生长暂停 | 如何停止复制/分裂并保留重新进入能力？ |
| M03 | 基因组完整性与损伤修复 | 长时间暂停时损伤负担和维护怎样变化？ |
| M04 | 染色质与转录记忆 | 如何建立、维持并解除状态相关调控？ |
| M05 | RNA 加工、保存与转录后控制 | RNA 如何加工、稳定、储存并为恢复准备？ |
| M06 | 蛋白合成与折叠质控 | 怎样降低合成负担并保留可恢复的蛋白稳态？ |
| M07 | 自噬、溶酶体与降解回收 | 物质与细胞器的清除回收怎样支持维持或恢复？ |
| M08 | 能量供给与线粒体状态 | 低活动与恢复时如何维持能量并改变代谢选择？ |
| M09 | 储备物质与脂质重塑 | 如何积累、保存和动用有限资源？ |
| M10 | 应激、防御与氧化还原 | 怎样控制氧化损伤并应对外界挑战？ |
| M11 | 膜、运输与离子/水稳态 | 如何维持屏障、运输、渗透压及可逆物理状态？ |
| M12 | 身份、骨架与组织结构保存 | 暂停期间如何保留细胞身份和组织结构？ |
| M13 | 细胞间通信与生态位协调 | 不同细胞、组织及母体条件如何协调暂停与恢复？ |
| M14 | 存活、细胞死亡与恢复能力 | 暂停能否保存活性，并在刺激后真正恢复？ |

## 每项定义

### M01 环境感知与内分泌决策

Environment sensing and endocrine control。如何感知季节、营养与释放条件，并决定进入或恢复？

- 子过程：光周期/温度及昼夜节律；胰岛素/TOR/AMPK 等营养信号；物种特定的激素与母体条件。
- 可观测通道：刺激历史、激素处理元数据；受体/信号相关 RNA；磷酸化与激素测量（若有）。
- 推断边界：单凭受体 RNA 推断信号强度或跨物种共享内分泌规则。
- 来源：[P03](SOURCES_CN.md#p03), [P06](SOURCES_CN.md#p06), [P07](SOURCES_CN.md#p07), [DB_GO](SOURCES_CN.md#db_go)。
- GO 检索锚点：[GO:0007165](https://www.ebi.ac.uk/QuickGO/term/GO:0007165), [GO:0007623](https://www.ebi.ac.uk/QuickGO/term/GO:0007623)。
- 状态：拟议、待领域审核；适用背景 `core_with_taxon_specific_channels`。

### M02 细胞周期与生长暂停

Cell-cycle and growth arrest。如何停止复制/分裂并保留重新进入能力？

- 子过程：G0/G1/S/G2 与复制许可；生长/细胞体积；重新进入与分裂恢复。
- 可观测通道：复制/细胞周期相关表达；EdU、DNA 含量、分裂及体积测量。
- 推断边界：将所有低增殖状态称为 diapause，或把低 RNA 当作 G0 证明。
- 来源：[P03](SOURCES_CN.md#p03), [P09](SOURCES_CN.md#p09), [DB_GO](SOURCES_CN.md#db_go)。
- GO 检索锚点：[GO:0007049](https://www.ebi.ac.uk/QuickGO/term/GO:0007049)。
- 状态：拟议、待领域审核；适用背景 `core_observation`。

### M03 基因组完整性与损伤修复

Genome integrity and repair。长时间暂停时损伤负担和维护怎样变化？

- 子过程：DNA 损伤检测与修复；复制压力与染色体完整性；转座元件控制。
- 可观测通道：修复相关 RNA；损伤/修复实验、突变和染色体 readout（若有）。
- 推断边界：修复基因上调等于更高修复通量；本次缺少精读的直接 diapause 功能来源。
- 来源：[DB_GO](SOURCES_CN.md#db_go)。
- GO 检索锚点：[GO:0006281](https://www.ebi.ac.uk/QuickGO/term/GO:0006281)。
- 状态：拟议、待领域审核；适用背景 `general_function_direct_evidence_gap`。

### M04 染色质与转录记忆

Chromatin and transcriptional memory。如何建立、维持并解除状态相关调控？

- 子过程：Polycomb/组蛋白调控；DNA 甲基化与 TET；染色质可及性、TF 调控与记忆。
- 可观测通道：RNA、ATAC/ChIP/CUT&Tag 或甲基化；真实遗传/药理扰动与功能结局。
- 推断边界：TF 表达、motif 或计算 regulon 当作真实因果边。
- 来源：[P01](SOURCES_CN.md#p01), [P02](SOURCES_CN.md#p02), [P04](SOURCES_CN.md#p04), [DB_GO](SOURCES_CN.md#db_go)。
- GO 检索锚点：[GO:0006325](https://www.ebi.ac.uk/QuickGO/term/GO:0006325)。
- 状态：拟议、待领域审核；适用背景 `core_mechanistic_seed`。

### M05 RNA 加工、保存与转录后控制

RNA processing and storage。RNA 如何加工、稳定、储存并为恢复准备？

- 子过程：剪接与 RNA 稳定性；miRNA/非编码 RNA；RNA 颗粒与翻译储备。
- 可观测通道：RNA isoform/非编码 RNA；颗粒成像、稳定性与翻译测量（若有）。
- 推断边界：由 bulk RNA 推断 RNA 颗粒或蛋白合成速率；需补充直接休眠论文。
- 来源：[DB_GO](SOURCES_CN.md#db_go)。
- GO 检索锚点：[GO:0006396](https://www.ebi.ac.uk/QuickGO/term/GO:0006396), [GO:0006402](https://www.ebi.ac.uk/QuickGO/term/GO:0006402)。
- 状态：拟议、待领域审核；适用背景 `general_function_direct_evidence_gap`。

### M06 蛋白合成与折叠质控

Protein synthesis and quality control。怎样降低合成负担并保留可恢复的蛋白稳态？

- 子过程：核糖体与翻译调节；伴侣与折叠；ER 应激/UPR 与蛋白损伤。
- 可观测通道：核糖体/伴侣相关 RNA；新生蛋白、蛋白质组与聚集 readout。
- 推断边界：核糖体转录量直接等于翻译速率；各通道不得只平均成 maintenance 分数。
- 来源：[P03](SOURCES_CN.md#p03), [DB_GO](SOURCES_CN.md#db_go)。
- GO 检索锚点：[GO:0006412](https://www.ebi.ac.uk/QuickGO/term/GO:0006412), [GO:0006457](https://www.ebi.ac.uk/QuickGO/term/GO:0006457)。
- 状态：拟议、待领域审核；适用背景 `core_with_submodule_gaps`。

### M07 自噬、溶酶体与降解回收

Autophagy and degradative recycling。物质与细胞器的清除回收怎样支持维持或恢复？

- 子过程：自噬/选择性细胞器回收；溶酶体功能；泛素-蛋白酶体降解。
- 可观测通道：相关 RNA/蛋白；带阻断对照的通量实验与溶酶体测量。
- 推断边界：溶酶体 RNA 或数量等于通量；忽略 HSC 与 fibroblast 的不同背景。
- 来源：[P05](SOURCES_CN.md#p05), [P09](SOURCES_CN.md#p09), [P10](SOURCES_CN.md#p10), [DB_GO](SOURCES_CN.md#db_go)。
- GO 检索锚点：[GO:0006914](https://www.ebi.ac.uk/QuickGO/term/GO:0006914), [GO:0010498](https://www.ebi.ac.uk/QuickGO/term/GO:0010498)。
- 状态：拟议、待领域审核；适用背景 `context_dependent_direction`。

### M08 能量供给与线粒体状态

Energy supply and mitochondrial state。低活动与恢复时如何维持能量并改变代谢选择？

- 子过程：糖酵解/TCA/呼吸；线粒体状态与动力学；能量需求与供需匹配。
- 可观测通道：能源相关表达；ATP、呼吸/酸化、膜电位和示踪（若有）。
- 推断边界：RNA 分数等于 ATP、通量或线粒体健康；不预设所有通道下调。
- 来源：[P03](SOURCES_CN.md#p03), [P05](SOURCES_CN.md#p05), [DB_GO](SOURCES_CN.md#db_go)。
- GO 检索锚点：[GO:0006119](https://www.ebi.ac.uk/QuickGO/term/GO:0006119), [GO:0006096](https://www.ebi.ac.uk/QuickGO/term/GO:0006096)。
- 状态：拟议、待领域审核；适用背景 `core_with_multimodal_need`。

### M09 储备物质与脂质重塑

Resource reserves and lipid remodelling。如何积累、保存和动用有限资源？

- 子过程：脂滴/脂肪酸/甾醇；糖原/保护性糖类；氨基酸等资源利用。
- 可观测通道：代谢相关 RNA；脂质组/代谢组、脂滴及底物消耗。
- 推断边界：脂质积累统一意味着更深休眠；不跨背景合并相反结果。
- 来源：[P02](SOURCES_CN.md#p02), [P05](SOURCES_CN.md#p05), [P06](SOURCES_CN.md#p06), [P08](SOURCES_CN.md#p08), [P12](SOURCES_CN.md#p12), [DB_GO](SOURCES_CN.md#db_go)。
- GO 检索锚点：[GO:0006629](https://www.ebi.ac.uk/QuickGO/term/GO:0006629), [GO:0005975](https://www.ebi.ac.uk/QuickGO/term/GO:0005975)。
- 状态：拟议、待领域审核；适用背景 `core_with_taxon_specific_channels`。

### M10 应激、防御与氧化还原

Stress defence and redox balance。怎样控制氧化损伤并应对外界挑战？

- 子过程：ROS/抗氧化与氧化还原；热/缺氧/脱水应答；解毒、铁/血红素与条件性免疫防御。
- 可观测通道：应激相关 RNA；ROS/损伤、挑战后生存与恢复；免疫仅在有相关组织/处理时单独解释。
- 推断边界：应激表达高等于耐受强；感染/炎症不是普遍 diapause 核心。
- 来源：[P08](SOURCES_CN.md#p08), [P09](SOURCES_CN.md#p09), [DB_GO](SOURCES_CN.md#db_go)。
- GO 检索锚点：[GO:0006979](https://www.ebi.ac.uk/QuickGO/term/GO:0006979), [GO:0006950](https://www.ebi.ac.uk/QuickGO/term/GO:0006950), [GO:0002376](https://www.ebi.ac.uk/QuickGO/term/GO:0002376)。
- 状态：拟议、待领域审核；适用背景 `core_stress_optional_immune`。

### M11 膜、运输与离子/水稳态

Membranes, transport and physical homeostasis。如何维持屏障、运输、渗透压及可逆物理状态？

- 子过程：膜运输/内吞/分泌；离子/pH/钙稳态；水分、渗透保护与屏障。
- 可观测通道：转运相关 RNA；膜完整性、离子/pH、水分和耐受测量。
- 推断边界：表达推断膜运输动力学；dauer 脱水结果直接推广到所有胚胎。
- 来源：[P08](SOURCES_CN.md#p08), [DB_GO](SOURCES_CN.md#db_go)。
- GO 检索锚点：[GO:0055085](https://www.ebi.ac.uk/QuickGO/term/GO:0055085), [GO:0006811](https://www.ebi.ac.uk/QuickGO/term/GO:0006811), [GO:0006874](https://www.ebi.ac.uk/QuickGO/term/GO:0006874)。
- 状态：拟议、待领域审核；适用背景 `transfer_seed_direct_evidence_gaps`。

### M12 身份、骨架与组织结构保存

Identity and structural preservation。暂停期间如何保留细胞身份和组织结构？

- 子过程：细胞类型/干性与分化潜能；骨架/黏附/肌肉等结构；生殖组织等 life-stage 特定子过程。
- 可观测通道：身份 RNA 与组织分层表达；形态、肌肉/结构成像与功能测量。
- 推断边界：细胞捕获比例就是组织组成；marker 就是谱系或真实功能。
- 来源：[P01](SOURCES_CN.md#p01), [P03](SOURCES_CN.md#p03), [P04](SOURCES_CN.md#p04), [P06](SOURCES_CN.md#p06), [DB_GO](SOURCES_CN.md#db_go)。
- GO 检索锚点：[GO:0030154](https://www.ebi.ac.uk/QuickGO/term/GO:0030154), [GO:0007010](https://www.ebi.ac.uk/QuickGO/term/GO:0007010)。
- 状态：拟议、待领域审核；适用背景 `core_multiscale`。

### M13 细胞间通信与生态位协调

Intercellular and niche coordination。不同细胞、组织及母体条件如何协调暂停与恢复？

- 子过程：配体-受体候选与通信；细胞黏附/基质及生态位；母体-胚胎/组织间协调。
- 可观测通道：细胞类型分层表达与条件记录；配体/受体扰动、空间/接触和功能实验。
- 推断边界：RNA 配体受体共现证明通信；Nodal 结果证明所有生态位机制。
- 来源：[P12](SOURCES_CN.md#p12), [DB_GO](SOURCES_CN.md#db_go)。
- GO 检索锚点：[GO:0007154](https://www.ebi.ac.uk/QuickGO/term/GO:0007154), [GO:0007155](https://www.ebi.ac.uk/QuickGO/term/GO:0007155)。
- 状态：拟议、待领域审核；适用背景 `context_specific_with_gaps`。

### M14 存活、细胞死亡与恢复能力

Survival control and recovery competence。暂停能否保存活性，并在刺激后真正恢复？

- 子过程：死亡/凋亡控制；释放后恢复比例与时延；长期功能保存与竞争结局。
- 可观测通道：存活/死亡实验；标准刺激下恢复曲线及后续功能；相关 RNA 仅辅助。
- 推断边界：低死亡通路 RNA 等于活着；低增殖等于深休眠；结局不从 clock 反推。
- 来源：[P01](SOURCES_CN.md#p01), [P03](SOURCES_CN.md#p03), [P05](SOURCES_CN.md#p05), [P09](SOURCES_CN.md#p09), [DB_GO](SOURCES_CN.md#db_go)。
- GO 检索锚点：[GO:0006915](https://www.ebi.ac.uk/QuickGO/term/GO:0006915)。
- 状态：拟议、待领域审核；适用背景 `outcome_linked_not_rna_only`。

## 宽目录，窄测量

M03、M05 当前缺少直接 diapause 论文的完整审核；M10 的免疫、M11 的物理保护和 M13 的生态位子项也不可声称已有普遍证据。保留这些位置是为了指导检索与实验，不能自动给出分数或把它们当训练标签。

原方案的六个领域可作为显示层折叠分类，但新项目不导入内部 thesis 的 K13 成员或结果。M14 同时涉及机制与结局：有实测才填功能值；其余模块也可连接同一个结局，这些关联不表示独立证据。

任何模块都不强制单调，不以预期轨迹覆盖真实偏离。跨物种共同的是功能问题，具体成员、方向、时间尺度与可观测性需单独验证。
