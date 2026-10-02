# 数据与模块接口 v0.4

## A. ObservationBundle

一个bundle对应一个明确的物种/系统/材料/assay/programme定义/预处理上下文。当前不自动混合不同context训练。它包含已审核的programme观测而不是raw FASTQ。

`observations.npz`：

| 键 | 形状 | 含义 |
|---|---|---|
| values | N×P float32 | 实际programme表达观测；缺失位置传输值0，必须配mask |
| mask | N×P bool | 是否可测；false不表示低表达 |
| coverage | N×P float32 | 已测member覆盖率，范围0–1 |
| clock | N float32 | 有依据的分子坐标目标，不是小时数 |
| clock_mask | N bool | 是否存在该坐标监督 |

`manifest.json`：由`ObservationBundle.save()`写入，包含：

- schema_version=`observation-v1`；feature_ids固定顺序；arrays_sha256。
- context：species、system、material、assay、programme_definition_id、preprocessing_id。
- clock_reference_id：如明确版本的M4-like axis；它只是来源标识，必须由数据审核保证意义真实。
- split_policy：`unit_holdout`或`study_holdout`，没有默认跨研究泛化声明。
- rows：每行 observation_id、study_family、biological_unit、split、origin、link_ids。

split只允许train/validation/test/locked_test。origin只允许public/internal/synthetic。内部数据不能进入train或validation。link_ids为**全局有命名空间的**源样本/动物/母因子/共享对照/成对模态标识，用于阻止间接跨split。

`reference_eligible=true`是wave参考拟合的额外人工审核字段，只对未额外扰动、允许作参考的train行设置。不能以程序全部被某个旧PCA认可作为参考资格的唯一标准。

真实训练前，应剔除在train从未可测的programme，并冻结实际保留顺序；不能偷看test以后决定特征。重新整理词表就产生新programme_definition_id。

## B. 表达转换

`normalise_expression(x, mask, scale)`仅处理明确counts/log_expression/linear_abundance三种输入。整数counts示例方法为每样本总量归一后log1p，不声称适合所有bulk设计；已标准化、spike-in或array数据应使用经审查的研究内处理结果并记录preprocessing_id。

`score_programmes`要求输入已经确认的gene IDs和成员映射。它实现一个透明的平均样本内排名基线，可带正负成员权重。不是ssGSEA，不是AUCell，不是thesis M1的top-5%分数，也不是生化pathway activity。

RefSeq transcript→gene合并、ortholog映射、Ensembl版本、探针合并不在该函数内自动猜测。先在现有项目的data adapter中解决并保留映射表。未经测量支持的信号不能从功能文字补成观测。

## C. ResponseDataset

`response.npz`含current N×D、action N×A、target N×P、target_mask；transition额外有elapsed N。

`response.json`含mode、representation_id、input_names、action_names、target_names、context_id、time_unit、endpoint、protocol_id、split_policy、rows。

- endpoint：current为对照端点的programme或冻结表示，target为处理组均值减对照组均值。time不是必填，因为端点比较不是追踪轨迹。
- transition：current为起点表示，target为末态的**同一种**表示。elapsed是真实已知经过时间。当前最小回归器通过平均变化率拟合，适合数值基线，不是精细连续动力学。
- functional：target为具有明确端点与协议的连续实测读出。删失的恢复时间、二元/计数/比例的专用likelihood尚未实现；不得用普通ridge结果冒充校准概率。

每行除Observation行字段外，必须有control_or_initial_ids、outcome_ids，且全部出现在link_ids中。多研究/模态共用对照时，使用相同全局ID。

动作向量需在研究内有固定、可审查的列语义；action_names写清靶点、干预类型、剂量单位等，严禁把处理组DEG/真实末态clock作为动作特征。构成性基因型放在初始背景。若`new_acute_ko=true`同时`same_genotype_already_in_initial=true`，契约拒绝执行。

本框架会检查显式标记`input_contains_future_information=true`，但无法仅凭浮点矩阵自动识别被藏进来的结果信息；接入者仍需审核每列的生成时间与来源。

## D. 知识记录

JSONL每条包含record_id、study_family、source_ref、split、reviewed、kind、text。

- kind=object_description：另需object_id；可用于冻结Qwen向量。
- kind=experimental_evidence：另需evidence_type（observational/measured_perturbation/rescue/computational_prediction/hypothesis）。
- SFT另需prompt（聊天messages）和completion（已审核答案字符串）。

同一原研究的预印本、正文、补表和转述应使用相同study_family。用于数值预测的对象向量不接受结果记录；留出的论文家族在检索评分前排除。基础模型公开预训练的潜在污染无法完全审计。

Qwen不可用时直接报所缺依赖或权重，不用伪向量、随机向量或别的模型静默替代。zero/shuffled是显式消融，不是假装Qwen执行过。

## E. 工件与依赖

State run的`scope`绑定context、特征顺序和clock参考；`bundle_fingerprint`绑定数据及split；`semantic_hash`绑定语义缓存。响应模型的representation_id必须绑定其真实输入来源。

一个新的encoder checkpoint即使latent_dim相同，也不是旧响应头的兼容表示。将新encoder接入旧响应头之前必须重新拟合/验证，不能仅做shape检查。
