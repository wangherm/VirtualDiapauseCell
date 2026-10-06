# INT1：统一输入、计算与导出

本轮以 CW-stage-1 为冻结基线。三视图结构固定：bulk 使用 clock＋身份＋可见 programme 残差；core、coarse 使用 clock＋身份，coarse 不开启残差。完整候选固定使用领域语义图，不按本次请求挑选模型。复用已训练 adapter，不新做 LoRA、不重跑 PK2、不查询预留材料。

`full` 是默认模式，必须实际调用领域 Qwen。`numeric_only` 只关闭本次在线调用；数值模型是否含语义由独立的 `model_variant` 决定。完整模式缺权重、GPU或生成失败时返回 `partial`。参考不足的生物学输出是 `unsupported`，不补造功能 depth。

## 在现有 AutoDL 环境中运行

先将代码包解压到新的 release 目录，保留原仓库、模型和 run。进入该 release 后执行：

```bash
export VDC_PYTHON=/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python
export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
export VDC_STAGE_RUN="$(cat "$VDC_PRIVATE_ROOT/LATEST_STAGE.txt")"
"$VDC_PYTHON" -m pip install -e '.[app]'
bash scripts/launch_int1.sh
```

如果 LATEST_STAGE 不是希望使用的阶段，显式设置 `VDC_STAGE_RUN=/实际完成的阶段目录`。不以报告 ZIP 替代原始 run：装配需要配置、开发矩阵、已训练 adapter、语义缓存及真实参数。脚本读取该阶段的来源配置，逐项检查文件指纹、角色、adapter 和文本版本，出错就保留日志。

不要替换已有 CUDA PyTorch。首次安装若没有 screen，可在 AutoDL 终端运行 `apt-get update && apt-get install -y screen`。后台只启动这一轮队列；结果在新的私有 `int1_*` 目录。没有远程执行连接时，本地交付并不表示服务器已经启动。

固定装配包含三视图×四条件：zero 校验复用，base、domain、shuffle 各按原训练单位和配置拟合一次；保存参数、预测与基线。随后独立导出模型，复制到新目录，运行真实开发请求：三视图、单条/批次、CLI、HTTP冷启动与重启、隐藏目标隔离、可见输入变化、问题变化和兼容公共响应。Qwen原始回答与证据保存，不从截断答案抽取类别放行。

```bash
RUN="$(cat /root/autodl-tmp/vdc-private/LATEST_INT1.txt)"
cat "$RUN/status.json"
tail -n 50 -F "$RUN/console.log"
```

`Ctrl+C` 只退出 tail。进度明细在 `assembly/progress.json`；每个拟合有 START/COMPLETED。`full_verified` 表示完整工程验收通过；`partial` 表示已保存输出但至少一次必需调用未通过；`failed` 见具体异常。它们都不等于生物学泛化已验证。退出码依次为 0、2、1。生成失败也保留日志，固定预算结束后不自动追加训练。

任务结束自动在 run 的父目录产生私有报告 ZIP。手动再次打包：

```bash
RELEASE="$(cat "$RUN/RELEASE.txt")"
export PYTHONPATH="$RELEASE/src"
"$VDC_PYTHON" "$RELEASE/scripts/pack_int1_report.py" --run "$RUN"
```

报告包不含原始 counts 或大型 adapter；模型工件保留在 `$RUN/model`。报告、模型、样本角色及整个内部交付文档均不上传公共仓库。继续同一中断任务：设置 `VDC_INT1_RUN="$RUN"`，从其原 `RELEASE.txt` 目录调用启动脚本；改过代码/配置要新建 run，不能继承旧完成状态。

## 使用已装配模型

保持 release 中的源码版本与模型包匹配。设置 `PYTHONPATH="$RELEASE/src"`，基础权重路径通过 `VDC_QWEN_BASE` 或 `--base-path` 指定；模型包已包含实际领域 adapter。

```bash
"$VDC_PYTHON" -m vdc analyse --model "$RUN/model" \
  --request /path/to/request.json --output /path/to/new-result

# 显式无在线 Qwen；仍然使用领域语义数值模型。
"$VDC_PYTHON" -m vdc analyse --model "$RUN/model" \
  --request /path/to/request.json --output /path/to/new-numeric-result \
  --mode numeric_only --variant domain_clock_wave
```

请求 JSON 的文件路径相对该 JSON 解析。CSV 的行是观测profile；不能把未聚合单细胞 atlas 直接当作独立胚胎输入。

```json
{
  "request_id": "my-new-analysis",
  "species": "Nothobranchius furzeri",
  "context": "embryonic_diapause_exit",
  "view": "bulk",
  "scale": "counts",
  "input": {"format": "csv", "matrix": "counts.csv", "samples": "samples.csv"},
  "question": "Describe observed programme deviations and their evidence limitations",
  "regulons": true
}
```

矩阵第一列 `sample_id`，后续列为模型登记的全部基因ID，可以换列序但不能重复/缺失。样本表必须有 `sample_id,biological_unit,identity,identity_source`。可选 view 为 `bulk/core/coarse`；合法身份可从模型 `bundle.json` 的 `views` 或网页 `/capabilities` 读取；未知显式写 `unknown`。本版 killifish 入口接受整数原始counts，禁止将 logCPM、FPKM或scaled表达冒充counts。每次1–64个profile。

H5AD 输入替换 `input` 为：

```json
{"format":"h5ad","path":"profiles.h5ad","counts_layer":"raw_counts","samples":"samples.csv"}
```

已有可靠身份优先继承，Qwen建议不会改写它；未知身份不因解释成功就获得参考。clock不裁剪为0–1，也不是小时。超范围的参考值、缺锚点与外推单独显示。gene预测仅指冻结目标panel，TF RNA／靶集合代理不是真实TF活性。

## 实际输入网页

原 AutoDL run 的快捷启动会从装配来源定位正确的 base：

```bash
bash "$RELEASE/scripts/serve_int1.sh"
curl -f http://127.0.0.1:8769/health
tail -n 30 "$RUN/service.log"
```

若将模型搬到另一台机器，显式设置其基础权重路径后使用同一CLI：

```bash
screen -dmS vdc_int1_app "$VDC_PYTHON" -m vdc serve-app \
  --model "$RUN/model" --output "$RUN/requests" --port 8769 \
  --base-path "$VDC_QWEN_BASE"
```

服务仅绑定服务器 `127.0.0.1:8769`；从自己的电脑使用原 AutoDL SSH 主机与端口增加 `-L 8769:127.0.0.1:8769` 建立隧道，再打开本机 `http://127.0.0.1:8769`。网页接收请求JSON、CSV或H5AD上传，实际计算并下载报告；不是旧结果浏览器。服务不会随装配脚本长期启动，避免占住GPU；上述命令是装配之后的显式启动。

HTTP `POST /analyse` 接受 `{"request":{...内联genes/counts/samples...},"mode":"full","variant":"domain_clock_wave"}`。上传文件用 `/upload`；HTTP JSON禁止任意服务器路径。CLI/Python/HTTP共用 `VirtualDiapauseCell.analyse`，推理不fit。

## 公共响应及报告解释

`responses` 数组中可请求 `endpoint/transition/functional/regulon`，不用伪造killifish counts去调用线虫模型。读取 `/capabilities` 的 `public_models` 得到准确 scope、特征顺序和合法action。端点/转移要求 `species`、完全匹配的 `scope`、一行 `current`、一行 `action`；转移另需一项 `elapsed`。功能请求要求匹配protocol、`hours_after_release`、`history_days`，输出该测量协议下的组比例，不是expression-to-depth。

公共 regulon 请求使用 `module=regulon`、`species=Caenorhabditis elegans`、`genes`、`expression` 和 `scale`；scale为counts、显式estimated_counts或log1p_library_10000，不能将小数estimated counts冒称整数counts。缺覆盖的集合值为null，并列mask/coverage。本地regulon通过主请求的 `regulons=true` 查询，明确是训练内共表达靶基因的可见RNA均值，不能当作因果活性。

报告包括HTML、Markdown、JSON、CSV、图、证据和原始Qwen轨迹。观测、参考、预测和假设分开；匹配Ridge和旧CW-stage在同一新输入上重新计算。public响应另列训练条件均值与时间/历史、零效应/持续状态等适用基线。LLM引用检查只证明编号存在，不自动证明机制解释正确，仍需专家审阅。

本轮使用已经查看的开发样本做工程验收。历史reserved应用结果不重新标成独立测试，缺少功能标签不造depth；模型弱或语义无增益也照实保留，不继续扩网格。
