# PK2 全队列：启动、观察、恢复与打包

这是在现有 `vdc` 模型、样本准入和数据契约上扩展的执行器。171 个逻辑训练槽全部映射到命令；不能将下载完成、旧服务运行或两步联调计为正式扩训完成。真实 GPU 执行由用户在 AutoDL 启动，本地交付不代表已提交服务器。

## 这一版实际执行什么

| 工作包 | 逻辑槽 | 执行方式 |
|---|---:|---|
| P-W / P-M / P-R / P-X |12|按研究拟合训练内尺度，研究及原始单位加权，保留研究读出|
| R0–R6，三个视图，三个种子 |63|本地、整模型可共享参数、仅编码器迁移；本地重建尺度/clock/语义投影|
| base / domain / shuffle 语义 |36|实际加载对应 Qwen，使用冻结向量进入状态模型|
| 新 Qwen adapter |6|两个学习率×三个种子；各自加载、两步 smoke、保存重载、从 base 开始三个 epoch|
| 逐 pool 开发重采样 |18|仅已授权开发单位，整组排除后重建轴、特征、scaler；不改原角色表|
| 嵌套训练量曲线 |18|25/50/100%；缺锚点单独阻塞；100% 与同配置主任务完全等价时验证复用|
| 细 programme / 2000 gene |18|96 项可覆盖 GO 面板、训练内方差选 gene；gene 使用小型数值 MLP|

通常训练 3000 步，保存 500/1500/3000 步快照；R6 为6000步的额外优化对照，不能称为严格 FLOPs 匹配。所有任务评价固定 development，不查询预留。gene 输入改变了信息量及架构，不能解释为纯架构优劣。

CPU 分支还包括全局/类型条件 waves、公共和本地 regulon、端点、时间/历史加初始状态残差、功能比例读出、整历史/未来时点/未见双突变/整条微阵列系列测试，以及 counts、gene 缺失、cell 下采样和组成偏移。另有观测初始状态与新预训练重构初始状态的端点/转移对照，以及在共同实测小时范围内的RNA→微阵列programme曲线迁移。功能比例有真实协议；没有表达—功能配对数据，expression-to-depth 仍不可用。

公共新增 METTL3、TET、rat、Dauer MTC 均从已核验文件映射 programme。TET 歧义多 locus 不重复累加；缺失值保留 mask。小鼠和大鼠 GO 快照是锁定 hash 的公开全物种注释，首次还需下载约24 MB。已有表达文件复用。

GSE303716 当前是差异表，独立记录原始测序入口及缺失样本矩阵，不进入 P-M/P-R 必需依赖。GSE104616 将技术重复按时点合并，完成公开处理矩阵上的probe曲线插值及均值对照；共同RMA/ComBat处理使它仅能作为描述性比较，不能冒称独立样本泛化或训练内预处理。GSE124109 的公开 FPKM 也有原研究共同处理的限制。

## Qwen 范围和评价

当前可核验语料为250条：原27条、133条GEO元数据弱参考、30篇新原始论文的90条摘要级记录。新论文按相关物种/实验系统保守合为24组，加旧3论文组，未达到30–50个独立论文家族与800–1500条审核记录的目标。没有用 Qwen 回答造 gold，也没有通过改写凑数。文献事实由摘要比对整理，仍需领域专家复核。

六条轨迹都是真实新训练。每条分别保存各epoch，按知识 validation token loss 选择 adapter，再用新进程加载、生成。评价分别报告给定证据抽取、无证据弃答、固定训练检索库无法覆盖留出研究时的弃答；后两项不等价于闭卷机制推理准确率。答案精确匹配会低估正确改写，因此原回答也保存。

## 启动：保留现有 PK1 服务

不要在旧服务仍依赖的目录里直接替换源码或重新 `pip install -e`。下面用 `git fetch` 获取代码，再从明确 commit 解出独立 release；复用原 CUDA 环境的 Python，以显式 `PYTHONPATH` 运行新代码。它不会重置工作区，也不会覆盖 PK1 快照。

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
git -c http.version=HTTP/1.1 fetch origin main
export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
export VDC_PK1_RUN=/你的已完成PK1运行目录
# 若旧指针不存在，显式设置 VDC_ACQUIRE_RUN 为已完成抓取目录。
git show origin/main:scripts/install_pk2_release.sh | bash
```

脚本检查依赖、数据和基础权重后用 `screen` 启动，打印 SESSION 和 RUN_DIR。只有实际启动成功，才有后台执行。缺依赖会明确退出；不会自动替换当前 CUDA PyTorch。

运行时先记录真实硬件和小模型吞吐；CPU起步2个worker，每个最多2个线程。Qwen和选择在GPU执行的数值任务共用一个GPU槽。Qwen未就绪不阻塞本地数值任务；不设置所有seed42完成的全局屏障。每个进程结束，立即保存评价和工件清单，再派发下游。代码/数据/环境签名不一致不能续跑。

## 看进度

```bash
RELEASE="$(cat /root/autodl-tmp/vdc-private/LATEST_PK2_RELEASE.txt)"
JOB_DIR="$(cat /root/autodl-tmp/vdc-private/LATEST_PK2_TRAINING.txt)"
cd "$RELEASE"
python scripts/pk2_training_status.py --run "$JOB_DIR"
tail -n 60 -F "$JOB_DIR/console.log"
```

`Ctrl+C` 只停止 tail；后台任务继续。`screen -ls` 看会话；`nvidia-smi` 看当前显卡。

每任务日志在 `logs/<task>.log`；数值步日志在 `tasks/<task>/model/steps.jsonl`，Qwen实时训练进度在其 worker log，完成后写 `trained/training_log.json`。`resources.json`/`resource_history.jsonl`记录资源；`queue_status.json`覆盖所有171槽；`REPORT_CN.md`列出每条原因。

`evaluated_new`是这次预算下训练并评价结束；`reused_verified`是核验等价工件；`blocked_data` / `blocked_dependency` / `blocked_resource`分别解释缺数据、上游、硬件；`failed`是实际程序错误。表现低于基线不是失败退出条件。`--integration-steps`会明确标记短预算，不是正式结果。

启动器退出码2表示仍有失败/阻塞；不能解读为正在重试。日志停止更新时先看状态和退出码。长驻服务进程不属于训练队列。

## 中断后继续

在原release及原Python/输入/配置上，`export VDC_PK2_RUN="$JOB_DIR"; export VDC_RESUME=1`，重新运行 `launch_pk2_training.sh`，同时恢复初次启动时的输入环境变量。已完成任务先验证hash再复用；未完成任务保留旧尝试日志后重跑该任务。当前调度器的恢复粒度是任务，未宣称自动精确恢复任意中断步。

## 新实验服务

队列耗尽后冻结实际新模型；新服务默认只监听 `127.0.0.1:8766`，原8765不变。端口占用会报告失败，绝不杀掉已有服务。先以独立HTTP客户端完成两次启停重载一致性检查，再启动新服务。

`/analyse`调用新状态模型及与其checkpoint匹配的programme waves；`/public_response`调用端点；`/functional`为协议限定的群体功能比例。`/analyses`和`/analysis?module=...`返回其余分支保存的开发评价，明确不是当前请求的新预测。当前入口只接受已授权开发样本，不支持任意新上传数据或预留样本。

## 打包给我分析

```bash
RELEASE="$(cat /root/autodl-tmp/vdc-private/LATEST_PK2_RELEASE.txt)"
JOB_DIR="$(cat /root/autodl-tmp/vdc-private/LATEST_PK2_TRAINING.txt)"
python "$RELEASE/scripts/pack_pk2_training_report.py" --run "$JOB_DIR"
```

输出私有ZIP包含任务表、实际训练/评价日志、预测、基线和失败原因，不含权重、原始表达矩阵、完整角色表或私有下载链接。单文件过大时清单记录省略。权重保存在服务器任务目录。报告仍含开发样本标识，请私下交回，不提交公开GitHub。
