# CW1：身份、参考相对表达与显式 clock–wave

本轮沿用已授权 development 样本，复用现有代码和服务器文件。目标是检验“身份/背景 → 参考坐标 → waves 与偏离”。数值组件实际拟合，现有原始/领域 Qwen 实际推理。没有新增 LoRA、171 条网格、外层 agent 或未来预测头。

## 实际任务

执行前保存配置与完整 DAG。一个 CPU 数值任务和一个 GPU 推理任务可并行；每卡只加载一个 Qwen。身份卡完成后 GPU 无须等待所有数值拟合。

|工作|执行内容|
|---|---|
|数据|核验批准角色、source lock、真实文件和成员；一次缓存开发 counts，后续不重复读 master H5AD|
|身份|固定粗分组；原 split 与逐 pool 留出；训练内选 marker，内层整单位校准拒答；数值、原 Qwen、领域 Qwen 对照|
|数值|bulk、whole pool、粗身份 profiles × C0/C1/C2；从隔离隐藏目标后的表达重拟合轴、waves 和读出|
|泛化|实际 pool 数 × C0/C2；闭包排除共享材料/cohort；不借用外层锚点|
|语义|预定 bulk/C2；零图复用主拟合；正确领域语义图、分层打乱图重拟合；核验实际 adapter 与向量|
|独立目标|hash 隐藏基因；另加按预定 GO ID 顺序前三个完整成员家族留出，重拟合参考|
|组合|同一粗身份模型接受源注释、数值推断、原/领域 Qwen 确认身份；拒答传播到下游|
|waves|gene RNA/TF RNA 线性曲线；保存系数、观测、预期、残差、幅度和形状，不宣称 TF 活性|
|压力|固定模型；programme 遮盖10/30/50%，counts保留100/50/25%，每档五次，保存预测；另从真实细胞抽取50/25%，比较固定与偏移组成|

九个开发 core pool 时共 **55 个执行任务**。两项 Qwen 作业各加载一次，内部完成十个身份评价批次。讨论稿九折之外增加原 split 身份评价，用于组合检查；不是新增训练。队列按真实单位生成，不凑数量。

## 方法和解释边界

C0 为现有 rank；C1 加训练 Early 中心化；C2 再加本地相对 log abundance。C1 可能只是可逆换尺度，不增加信息；C2 改变信息量，不是纯架构对照。幅度不是绝对每细胞 RNA。

`identity_crosswalk.json` 是功能粗分组，不是 Cell Ontology 真值或发育谱系。各 pool 内先加总符合原 min-cell 条件的类型 counts，再归一化。聚合不等于整个 pool。当前身份评估是“按源注释聚合后的 pseudobulk 可分性”，不等于盲细胞注释；I0 与 silver 标签同源，不计为独立分类准确率。

身份 marker、尺度、centroid 在每个内层训练折重新计算。支持不足就 unknown。Qwen 读取同一 marker、可溯源符号、候选及数值支持，不读取查询样本名、原类别、condition、时间、clock、隐藏基因或评价答案。第一轮采用保守确认规则：LLM 可以确认/拒答，不能绕过数值门槛。不解析成功、超token预算或无合法证据时拒答，保存实际输出；不造概率、不把答案当训练 gold。

每个支持身份用训练 Early/Developing 锚点构建分子参考，不复制 whole-pool clock。低数量锚点只支持探索性参考。先固定“身份截距＋共享斜率”wave，再使用训练噪声尺度、收缩相关矩阵和 Huber 残差求位置。缺身份、覆盖不足、近乎平坦时拒答；外推坐标不裁剪，但标明超范围，范围外不填受支持残差。当前不输出经校准的置信区间，不把小残差当定位正确证明。

隐藏基因从所有重叠 programme 成员、rank 背景、输入库大小分母、幅度、身份 marker 和轴中一起移除。目标仍为完整库相对表达，有组成依赖，不是 clean truth。粗身份视图的隔离以既定源注释聚合为条件：历史注释可能用过完整表达，因此不能宣称完成原始细胞重注释的端到端独立验证。主指标按生物学单位汇总；direct Ridge 获得相同身份信息，另有同身份均值与全局均值基线。`common_support_comparisons.json` 比较三种表示共同支持的样本，不混淆覆盖率变化与误差改善。完整 family 目标只按静态注释选择；不再可辨识就记录不适用，不回退完整表达轴。

语义分支核验既有 GO 成员、adapter 和向量顺序后，构成功能相似图作为 wave 系数的固定正则项。零图、正确图和成员数量/覆盖分层打乱图使用同一数值输入和强度，并检查真实系数变化。它不是新同源推断，也不证明跨物种细胞身份迁移已验证。

可选神经学生默认关闭，身份 LoRA 等待审定监督。细胞组成任务单次流式读取真实开发细胞，对原 validation pool 做50/25%细胞抽样；固定与偏移组成使用相同总细胞数，各五次。比较抽样观测及原观测，不当作因果干预或 clean truth；当前只接 whole-pool 模型。热应激 counts 待补；旧标准化表达留在原描述分支。没有 expression-to-depth 或未来预测输出。预留角色不变，本轮不查询它们；后续比较必须说明回顾性。

## AutoDL 安装和启动

保留现有 CUDA Python，不重装 torch。以下从远程归档独立 release，避免覆盖旧模型、服务及上传过的工作区：

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/VirtualDiapauseCell
  export VDC_PYTHON="$PWD/.venv-autodl-vdc/bin/python"
  export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
  git -c http.version=HTTP/1.1 fetch origin main
  INSTALLER="$(mktemp /root/autodl-tmp/cw1-install.XXXXXX.sh)"
  git show origin/main:scripts/install_clock_wave_release.sh > "$INSTALLER"
  bash "$INSTALLER"
)
```

脚本从 `LATEST_APPLICATION.txt` 指向的 `config.json` 自动读取修复后 PK2 来源。若无此指针，先设置 `VDC_CW_SOURCE` 为实际完成的修复后 PK2 run，不能填报告 ZIP 或 application run。目录需有 config、queue_status 和 `tasks/select_adapter/selection.json`。私有路径只保存在私有 run。

GitHub 不通时把完整离线包解压进新的 `/root/autodl-tmp/vdc-releases/cw1-offline`，不要覆盖旧 checkout，然后：

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/vdc-releases/cw1-offline
  export VDC_PYTHON=/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python
  export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
  APP_RUN="$(cat "$VDC_PRIVATE_ROOT/LATEST_APPLICATION.txt")"
  export VDC_CW_SOURCE="$("$VDC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$APP_RUN/config.json")"
  bash scripts/launch_clock_wave.sh
)
```

会输出 screen、run 和 release。前台返回不表示完成。首次文件校验/提取较慢；后续使用缓存。不需要再上传已有数据包。

## 进度、续跑、界面和打包

```bash
export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
export VDC_PYTHON=/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python
RUN="$(cat "$VDC_PRIVATE_ROOT/LATEST_CLOCK_WAVE.txt")"
RELEASE="$(cat "$VDC_PRIVATE_ROOT/LATEST_CLOCK_WAVE_RELEASE.txt")"
export PYTHONPATH="$RELEASE/src"
"$VDC_PYTHON" "$RELEASE/scripts/clock_wave_status.py" --run "$RUN"
tail -n 50 -F "$RUN/console.log"
```

`Ctrl+C` 只退出 tail。详细日志在 `logs/任务名.log`。completed 是实际执行并保存工件；not_applicable 是数据/可辨识性不足；failed 是程序或环境错误；blocked 是相关依赖未完成。退出码0表示全队列完成，2表示队列结束但存在未完成项。完整清单见 REPORT_CN.md 和 queue_status.json。

同版本同输入续跑会先校验已完成工件，不重复计算：

```bash
bash "$RELEASE/scripts/resume_clock_wave.sh"
```

队列结束自动执行独立进程 HTTP 检查，再关闭临时检查进程。长期结果界面另开端口8769，不停旧服务：

```bash
screen -L -Logfile "$RUN/service.log" -dmS vdc_cw1_results \
  "$VDC_PYTHON" "$RELEASE/scripts/serve_clock_wave.py" --run "$RUN" --port 8769
curl --fail http://127.0.0.1:8769/health
```

核对返回 snapshot ID 与 results_snapshot.json 相同；端口已占用则选其他端口。经 SSH/Jupyter 私有转发访问，不监听公网。界面是保存的开发预测，**不是任意上传的新推理服务**。

```bash
"$VDC_PYTHON" "$RELEASE/scripts/pack_clock_wave_report.py" --run "$RUN"
```

输出 `VDC_CLOCK_WAVE_PRIVATE_report_*.zip`，含预测、曲线、数值参数、身份卡/回答、指标和日志，不含原始counts或大模型权重。仍含私有信息，只下载后交给本地分析，不推GitHub。
