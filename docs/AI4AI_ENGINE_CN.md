# R2 研究引擎修复与新阶段续跑

研究目标仍是 universal diapause/dormancy model，概念、模型和问题由 AI 选择。当前所有已授权数据角色可用于自适应研究，原始角色和质控记录继续留档，不宣称存在新的独立盲测。这一版修复工具闭环，不预先规定科学模型，也不重跑 PK2 队列。

## 本版改变

- 公开来源的派生元数据可读到实际内容；混合私有来源仍按私有处理。原始私有样本表可返回批准的样本、时间、材料、单位及链接字段，不返回表达值。分页返回实际 `next_offset`。
- `artifact_inputs` 明确引用准备输出，后续代码通过只读挂载继续处理，使用内容指纹和来源链检查；不让模型猜宿主机路径。
- 准备状态分成 `metadata_ready`、`matrix_ready`、`task_ready`。后者只表示六份必要文件齐全，登记时仍核对数组与科学契约。
- 新任务的 `unit_type` 与 `replication_unit_id` 分开，单位 ID 回查原始 JSON/CSV/TSV/SOFT，保留已存的胚胎、培养和共享对照链接。GSM 或 library ID 本身不证明生物学独立，报告明确保留这一限制。
- 声称严格隐藏目标的任务须运行原始矩阵、目标扰动、可见输入扰动三次特征重算。全基因 CPM 导致目标变化影响输入时拒绝登记。当前源矩阵适配器支持 gene×sample CSV/TSV counts；其他格式需补适配器，不静默通过。一般观测协变量预测可显式登记为 `observed_covariates`，不冒称严格隔离。
- 内容相同、ID 不同的读取复用原结果。连续 8 次决策没有新增工件/任务/评价/导出或下载资产时进入诊断；24 次暂停为 `stalled_no_progress`。新增参数文件不等于模型进步，成绩仍以真实评价和导出为准。这是防空转机制，不是科学成功条件。
- 文献明确标记摘要页、正文和代码，给出正文候选链接。摘要引用不能满足方法证据条件；读过正文/代码且记录方法使用说明也不等于复现作者算法。

## 新公开数据权限

AI 可以调用 `discover_data(source, research_reason)`，为缺失数据或新问题登记 GEO/ENA accession，并获取官方元数据和文件表。然后使用 `probe_download` 查看文件大小，`download` 下载、校验、登记。已知目录之外的合适数据无需逐项人工批准。

当前支持 GSE、PRJEB、PRJNA、PRJDB，下载仅走官方 GEO/ENA 文件域名；其他数据库需新增明确的主机适配器。ENA 文件必须列在对应项目的官方文件表内，自动采用其 MD5 和字节数。GEO 未提供校验和时记录“实收 SHA256＋声明字节数”，不称为供应商校验。协议不从 ETag 推断 MD5，不推断许可证。

每文件默认上限 20 GB，磁盘预留 10 GB，可在冻结配置中调整。不会覆盖来源变更的旧文件。已有匹配的校验缓存可复用；这个新 broker 的中断文件保留为 `.part`，重试从该文件开头重新下载，尚未实现新来源的断点续传。原目录批量下载器保持自己的断点续传能力。某项下载失败只阻塞对应分支。

下载成功、已解析、可建任务、实际用于数值训练是四件事，分别记录。新下载数据不会自动混入已经封存的任务。

## 新阶段与费用记录

`start_research_engine.sh` 是本轮正式入口，需要服务器上的完整停止 run，不能用删减过的报告 ZIP 恢复。

1. 后台执行真实 GSE288723 文件的工程回归：公开元数据 → 派生挂载 → 原始单位核验 → 目标依赖反事实 → 两个同任务候选 → 父子比较 → 全任务拟合、导出、两次重载。无 API 调用，结果单列，不计入自主研究成绩。
2. 加锁读取旧 run，复制并核验准备工件、文献和依赖缓存，原目录不改写。
3. 旧任务、模型分数、API 台账和决策保留在 `history/`；旧任务不直接进入新排行榜。准备结果保留，AI 用新契约登记可比较的新任务。来源用途先记录为历史，不能冒称新阶段已经训练。
4. 冻结新 release，在新目录启动研究。新阶段最多 1000 次研究 API 请求，累计 token 上限仍是 `null`；不再调高。旧阶段次数和已返回 token 单独展示，未知用量保留未知，不抹掉旧账。

当前 AutoDL 已验证的是 CPU 隔离。默认 CPU，DeepSeek 推理仍在远端。只有实际 GPU 隔离资格通过后才可以选择 `VDC_R2_DEVICE_POLICY=ai`，没有跳过隔离的 GPU 入口。新入口复用已合格运行时，不替换 CUDA PyTorch。

工程检查的数据定位优先读取 `VDC_RESEARCH_ASSET_REPORT` 或 `LATEST_RESEARCH_DATA.txt` 指向的下载收据，再读取明确指定的父 run 的 `inventory.json`，最后检查默认缓存。旧下载器可能只复用外部缓存而未复制文件到 `public_assets`；离线 release 也没有 `data/raw`。所有命中的文件仍须匹配固定字节数和 SHA256，不会仅凭文件名接纳。确实缺失时报告具体查找位置，不再抛出无上下文的 `StopIteration`。

## 从 GitHub 启动

先在 JupyterLab 终端运行，`PARENT` 改为实际停止的旧 run：

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/VirtualDiapauseCell
  git -c http.version=HTTP/1.1 fetch origin main
  REV="$(git rev-parse origin/main)"
  RELEASE="/root/autodl-tmp/vdc-releases/engine_${REV:0:12}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
  mkdir -p "$RELEASE"
  git archive "$REV" | tar -x -C "$RELEASE"
  printf '%s\n' "$REV" > "$RELEASE/RELEASE_COMMIT.txt"
  export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
  export VDC_PYTHON=/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python
  export VDC_R2_RUNTIME_ROOT="$VDC_PRIVATE_ROOT/r2_runtime_devices_v2"
  export VDC_R2_DEVICE_POLICY=cpu
  PARENT=/absolute/path/to/stopped_parent_run
  bash "$RELEASE/scripts/start_research_engine.sh" "$PARENT"
)
```

密钥复用环境变量；未设置时终端隐藏输入。找到匹配的历史 API 检查收据则直接复用，不额外收费。如果收据缺失，脚本明确停止，需显式运行 `check_research_api.py`，不是偷偷反复请求。

离线包是同一份 Git 代码。解压到全新的 release 目录后，以相同环境变量调用其中的 `scripts/start_research_engine.sh` 即可。不要覆盖正在运行的 release。

## 看进度与恢复

先看工程检查/迁移日志：

```bash
ROOT=/root/autodl-tmp/vdc-private
tail -n 50 -F "$(cat "$ROOT/LATEST_ENGINE_SETUP.txt")"
```

看到 `NEW_RUN=...`、`SCREEN=...` 后，才表示新阶段已经迁移/派发。`LATEST_RESEARCH_R2.txt` 此时才更新；准备期间它仍可能指向旧 run。`ENGINE_SETUP_EXIT_CODE=0` 仅表示派发完成。

```bash
RUN="$(cat /root/autodl-tmp/vdc-private/LATEST_RESEARCH_R2.txt)"
cat "$RUN/status.json"
tail -n 50 -F "$RUN/console.log"
```

重点看 `experiments`、`comparable_revisions`、`exports`、`artifact_states` 和 `engine_progress`，而不只看 API 次数。`tail -F` 等待新增文本是正常现象，Ctrl+C 退出查看不停止后台任务。

需要在当前动作完成后停止：`touch "$RUN/STOP"`。无进展暂停不会自动收费重启；先核对 `engine_progress`、失败动作和缺失输入。保持代码/配置未变、障碍已排除后，可显式恢复：

```bash
export VDC_R2_RUN="$RUN" VDC_R2_RESUME=1 VDC_R2_RESUME_AFTER_STALL=1
export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private VDC_R2_DEVICE_POLICY=cpu
export VDC_PYTHON=/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python
export VDC_R2_RUNTIME_ROOT="$VDC_PRIVATE_ROOT/r2_runtime_devices_v2"
export VDC_RESEARCH_API_CHECK="$RUN/api_check_receipt.json"
# 需保持正确的 API 环境变量和密钥；不要在日志中打印它们。
bash "$RUN/release/scripts/launch_research_r2.sh"
```

若存在人为创建的 STOP 文件，它仍会阻止继续；不要将暂停后的重复启动当成研究进展。更改代码/配置必须另建 run，不沿用旧 completed 状态。

打包：

```bash
PYTHONPATH="$RUN/release/src" "$VDC_PYTHON" "$RUN/release/scripts/pack_research_r2.py" --run "$RUN"
```

报告是私有工件，含预测、模型和批准的样本元数据，不推送公共 GitHub。原始数组、依赖探针输入与历史任务数组不进入报告包。
