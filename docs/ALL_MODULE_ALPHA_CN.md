# All-Module Alpha：运行、进度与结果

这一版按全模块首轮运行组织，复用 `vdc.state / waves / response / knowledge`。效果差仍记录、仍可内部调用；缺数据或依赖才阻塞对应分支。它是实验系统，不是已验证的 virtual cell。

## 当前边界

- 本地 CPU 已实际执行两个 context 的状态训练（各 500 步）、局部候选坐标、状态读出、真实 gene/TF/programme 曲线、基因型端点、释放转移、组级功能读出及保存重载。
- Qwen GPU smoke、正式一轮 SFT、基础/adapter 问答、adapter 向量、匹配容量的零向量/正确向量对照由下面的 AutoDL 脚本执行。脚本存在不等于这些步骤已经跑过。
- 所有新结果只使用开发 train/validation。原 GSE288723 的 12 个 test、内部 killifish、原 VAL/L4-OOD 保留。GSE291659 新设 replicate 4 为保留列，不转换成数值输入。
- 功能表有真实群体计数，可以训练“历史＋协议时间 → young adult 比例”。RNA pool 与功能群体无法匹配，**表达 → depth 仍不可用**。
- 当前 GO programme 不是 regulon；尚未准入 TF-target 边表。因此总体通常为 `partial`，并不是整场运行崩溃。

## 1. 在 JupyterLab Terminal 启动

已有仓库时：

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
git -c http.version=HTTP/1.1 pull --ff-only
bash scripts/launch_alpha.sh
```

`launch_alpha.sh` 在 screen 中完成安装、GPU 检查和整个任务链。默认使用以前 VDC 的 `.venv-autodl-vdc/bin/python`。**如果该环境只有 CPU torch**，先在独立 screen 中建立新的专用环境：

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
screen -S vdc_setup
bash scripts/setup_alpha.sh
```

看到 environment ready 后退出该 screen（`exit`），再启动：

```bash
VDC_PYTHON="$PWD/.venv-alpha/bin/python" bash scripts/launch_alpha.sh
```

安装中断连接前，用 **Ctrl+A，再按 D** 脱离 screen。恢复查看：`screen -r vdc_setup`。新环境使用官方 torch 2.8.0 CUDA 12.8 wheel；不替换旧 FactorBridge 环境。官方安装出处：[PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)。已有可用 CUDA 环境也可通过 `VDC_PYTHON=/绝对路径/bin/python` 指定。

基础模型默认读取此前已经校验的缓存：

```text
/root/autodl-tmp/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
```

若路径不同，设置 `VDC_MODEL_PATH=/实际/base/snapshot` 再启动。必须是基础权重，不是以前的 FactorBridge adapter。不默认重下大模型，不关闭 TLS 验证，不悄悄改用其他模型。

所有新数据通过已锁定的官方 URL 下载并验证 SHA256；可复用 `data/raw/` 中相同文件。GO 当前 URL 发生版本漂移时会明确失败，不能默默更换注释。如果网络阻断，上传已校验文件到相同目录即可。

## 2. 看进度

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
JOB_DIR="$(cat runs/LATEST_ALPHA.txt)"
tail -n 50 -F "$JOB_DIR/console.log"
```

Ctrl+C 只退出 `tail`，不会停止 screen 中的训练。

```bash
python scripts/alpha_status.py
tail -n 30 -F "$JOB_DIR/logs/knowledge.log"
tail -n 5 "$JOB_DIR/state_base/steps.jsonl"
nvidia-smi
screen -ls
```

CPU 数值阶段 GPU 空闲正常。Qwen 阶段依次载入模型，不会在单卡上同时跑所有大模型。

状态含义：`running` 正在执行；`completed` 有真实工件；`failed` 看对应日志；`blocked_dependency` 缺少上游/真实 GPU；`blocked_data` 缺合格资料。退出码 **0=全部完成，2=partial，1=启动/调度异常**。模块错误记录到 `module_status.json`，不以 console 最后一行猜测是否全部训练成功。

## 3. 中断后继续

保持同一个代码版本、配置、模型路径，用同一个运行目录：

```bash
VDC_RUN_DIR="$(cat runs/LATEST_ALPHA.txt)" \
VDC_PYTHON="$PWD/.venv-alpha/bin/python" \
bash scripts/launch_alpha.sh
```

已完成模块须通过文件指纹复核才跳过。失败/中断模块的文件移动到该 run 的 `failed_attempts/` 留存，再按原预算重跑该模块；**总调度是模块级续跑，不声称所有模块都有逐 minibatch 精确恢复**。底层 state 和 Qwen CLI 保留 checkpoint 恢复能力。改了代码或配置必须新建 run，避免旧 smoke 签名被错误沿用。

运行过程中不要 `git pull` 或修改代码。代码指纹覆盖实现、配置、语料和派生功能数据；文档修改不影响它。

## 4. 打包给合作者或传回检查

```bash
python scripts/pack_alpha_report.py
```

在当前目录生成 `VirtualDiapauseCell_Alpha_report_时间.zip`。默认包含输入契约、划分、预测、数值拟合工件、训练日志、评价、接口请求/响应和打包清单；大型 `.pt/.safetensors` 与 Trainer checkpoints 不包含，清单明确列出省略项。需要权重一起交付时：

```bash
python scripts/pack_alpha_report.py --weights
```

基础 Qwen 8GB 权重位于外部 HF 缓存，不复制进报告。adapter、数值 best/last 权重保存在该运行目录中。

## 5. 内部实验界面

运行完成或 partial 后，只要实际模型调用检查通过：

```bash
JOB_DIR="$(cat runs/LATEST_ALPHA.txt)"
.venv-alpha/bin/python -m vdc serve-experimental "$JOB_DIR" --port 8000
```

绑定 **127.0.0.1**；通过已有 SSH 隧道或 Jupyter 受限代理访问。若环境装有 Jupyter server-proxy，可在 Jupyter 的 `/proxy/8000/` 路径访问；否则使用 SSH 本地端口转发。不自动绑定公网地址。

页面可以看模块状态/基线，上传 JSON 请求，显示 observed、reconstructed、clock、reference、residual、响应和功能结果。`experimental_release/examples/` 包含由真实开发样本生成的请求文件：

- `state_base.json` / `state_ard.json` → `/analyse`；有语义训练完成后另有 `state_semantic.json`。
- `state_base_pipeline.json` / `state_ard_pipeline.json` → `/pipeline`。
- `endpoint.json` / `transition.json` / `functional.json` → `/response`。
- `waves_gene.json` / `waves_TF.json` / `waves_programme.json` → `/waves`。状态分析响应另含已训练的条件读出。
- `{"query":"FOXO1 lipid dormancy"}` → `/knowledge/search`，这是明确命名的 train-only 检索。
- `/knowledge/answer` 在 adapter 训练和新进程问答完成后才开放，实际加载 adapter；需要 GPU，模型回答不会回写数值或标签。

自有输入必须声明与模型一致的系统、表达尺度及完整稳定基因 ID 集合。本轮支持两个指定的线虫 context，不会把任意物种文件自动判成同一系统。内部 killifish 不通过这个开发入口解锁。

停止服务用 Ctrl+C；停止训练必须明确找到其 screen 再终止进程。不要把退出日志查看当成停止任务。

## 6. 如何读结果

优先看 `REPORT_CN.md`、`summary.csv`、`module_status.json`，再检查：

| 分支 | 主要文件 | 解释边界 |
|---|---|---|
| 数据 | `data/*/admission.json`、manifest | 真正读取的列及样本/对照分组 |
| Clock | `clock_reference/axis.json`、scores | train 0h 与已观测 24h 定义；6h 由表达投影；不是独立真值 |
| 状态 | `state_*/metrics.json`、diagnostics、noise_curve | 隐藏重构、Ridge、动态收缩、clock 蒸馏；不是恢复概率 |
| 状态分类 | `state_readout/result.json` | 0h 条件 vs 释放条件；不是独立功能标签 |
| Waves | `waves/*/validation.npz`、shapes | 真实 gene/TF RNA/GO proxy；低自由度；越出参考范围返回 unavailable |
| 端点 | `endpoint*/result.json` | 构成性基因型对照差异；共享 N2 对照，非急性 KO/独立细胞 |
| 转移 | `transition*/result.json` | 条件匹配群体；与 last-state、时间历史基线比较 |
| 功能 | `functional/result.json` | young adult 群体比例；线性预测可能越界，原值保留并标出，不当校准概率 |
| 知识 | `knowledge/`、`knowledge_*_eval/` | 18 train、9 validation 的来源事实抽取试验；不是全部休眠知识 |
| 语义连接 | `semantics/embeddings.json`、`state_semantic/run.json` | base revision、实际 adapter hash、向量 hash、特征顺序；与 matched-zero 同预算 |
| 实验调用 | `experimental_release/` | 实际权重请求结果；不等于科学批准或已开启远程服务 |

原 research release 审批入口保留不变。`experimental` 不会把 `science_status` 改成 validated。
