# PK1 首轮数值训练：启动、进度与报告

这个入口执行新的公共预训练→killifish 数值迁移，并复用上一轮**实际训练过的 Alpha Qwen adapter**生成新语义向量。不是重新启动旧 Alpha，也不代表扩展知识语料、第二轮 Qwen 微调或 PK1 整体服务已经完成。

## 本轮范围

默认 `configs/pk1.json`：3个种子，每个数值模型500步，公共预训练500步。bulk、whole-pool、pool×cell-type 三种视图分别拟合。小型状态模型使用 CPU 两线程；Qwen 向量提取使用 GPU，模型顺序加载。GPU 利用率低不代表数值训练停止。

|路线|公共数值预训练|功能语义|
|---|---|---|
|K0|无|相同容量的零向量对照|
|K1|有|相同容量的零向量对照|
|K2|无|实际 Alpha adapter 向量|
|K3|有|实际 Alpha adapter 向量|
|K4|有|原始 Qwen 向量|
|K5|有|固定打乱的 adapter 向量|

同一种子使用相同的初始权重，迁移时重拟合本地归一化和 clock 头。各路线使用相同的5组额外遮蔽评价，与 Ridge 对照；保存噪声曲线。公共预训练增加的计算量明确记录，不声称等计算量比较。

clock 以训练集 Early→Developing 锚点建立不截断的候选坐标，不能称为已验证 biotime。Late 维护样本参与状态表示，但不监督 Exit clock、不进入 Exit waves。whole-pool 的参考用于 cell-type 视图。首个种子的 K0/K3 拟合 gene、TF-expression 和 programme 曲线，并比较参考坐标输入与模型 clock 输入；它们不等于已验证 TF 调控活性。保留当前样本角色，训练入口不执行冻结后查询。

公共 regulon、端点/转移和有真实计数的组级功能模型分别运行。缺失表达—功能对应标签时不输出 depth。共79项任务；分支失败只阻塞其依赖，退出码2表示未全部完成。`completed_current_scope` 只表示本文件范围完成，不表示研究效果已经验证。

## 先确认文件

旧准备代码包不含本入口。更新 GitHub 代码，或将新的训练代码包**解压覆盖到现有仓库目录**。保留 `.venv-autodl-vdc`、`runs/` 和已下载权重，不要创建第二层仓库。新离线包附有固定版本的公开网络/GAF缓存，私有数据包不需要重新上传。

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
test -f scripts/launch_pk1.sh
test -f /root/autodl-tmp/vdc-private/sample_roles.json
```

如果 PRIVATE 包还没有解压，在 JupyterLab 确认真实上传路径后运行（下面假定它在 `/root/autodl-tmp`）：

```bash
unzip -n /root/autodl-tmp/VDC_PK1_PRIVATE_Prepared_20261003.zip -d /root/autodl-tmp
```

## 正式启动

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/VirtualDiapauseCell
  export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
  # 使用之前实际完成的 Alpha 运行，目录必须保留 adapter 和公共数据。
  export VDC_ALPHA_RUN="$(cat runs/LATEST_ALPHA.txt)"
  bash scripts/launch_pk1.sh
)
```

如果原指针丢失，显式设置 `VDC_ALPHA_RUN` 为真实 Alpha 目录。只有结果报告 ZIP 不够，必须有 `module_status.json`、其清单对应的公开数据和 `knowledge/adapter/adapter_model.safetensors`。不要对旧 Alpha 目录重新跑训练来覆盖它。

启动器先检查 `screen`、批准名单、当前 CUDA 环境及 Qwen 快照，再在后台运行，日志写到私有目录。不自动更换 CUDA 版 PyTorch。环境不在默认位置时设置 `VDC_PYTHON`；Qwen 快照不在默认缓存时设置 `VDC_MODEL_PATH`。依赖安装需要可用的包镜像；模型加载只读本地权重。

## 看进度

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
.venv-autodl-vdc/bin/python scripts/pk1_status.py
JOB_DIR="$(cat runs/LATEST_PK1.txt)"
tail -n 60 -F "$JOB_DIR/console.log"
```

`Ctrl+C` 只结束上面的日志查看。任务仍在 screen 里运行。用 `screen -ls` 看后台会话；`nvidia-smi` 只反映 GPU 分支。状态脚本会显示正在执行的模块及最近一次记录的 step。某模块失败时查看 `$JOB_DIR/logs/模块名.log`。没有 `exit_code.txt` 不代表成功，可能仍在运行或被外部终止。

程序结束后，退出码0表示本轮声明范围全部完成，2表示部分任务失败或阻塞，其他非零值需查看启动日志。结果在 `REPORT_CN.md`、`comparison.json` 和各任务目录。

不要在运行中更新源码或依赖。中断后仅在代码/数据/配置/环境未变时复用同一目录：

```bash
export VDC_RUN_DIR="$(cat runs/LATEST_PK1.txt)"
bash scripts/launch_pk1.sh
```

已完成且校验一致的任务跳过；中断的任务保留在 `failed_attempts`，从该任务开头重跑。它不是步级恢复。修改源码或配置后，应取消 `VDC_RUN_DIR` 并启动新运行。

## 打包给合作者检查

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
.venv-autodl-vdc/bin/python scripts/pack_pk1_report.py
```

命令输出一个 **PRIVATE report ZIP** 路径，包含内部预测、样本标识、日志和评价；从 JupyterLab 下载后私下分享，不要放进 GitHub。权重仍保留在服务器任务目录，报告包不重复打包大型权重。

## 当前执行证据与限制

Windows 已在批准的真实开发观测上完成一轮1种子、5步短预算联调：数值分支21项完成；Qwen 来源/语义依赖未在本机执行，整体如实标记 partial。这个结果只验证接线、保存重载和实际数据兼容性，不能代替 AutoDL 上的500步正式结果。K0–K5的软件测试使用明确的合成 fixture 检查向量身份、输入遮蔽一致性和依赖关系，不冒充真实 Qwen 效果。

未完成范围：扩大知识语料及新一轮 Qwen 训练、新公共时序/干预接入、killifish regulon 的原拟合样本审定、冻结后查询、表达 depth 和 PK1 服务。此次先交付可运行的批处理训练及实际工件。
