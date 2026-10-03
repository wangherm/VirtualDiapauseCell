# PK2：公共数据抓取与开发服务

本次交付六个新增公共研究的下载、样本元数据与原尺度表达视图，以及将既有 PK1 模型冻结为内部服务的入口。两条分支独立运行。

**这不是 PK2 的 171 条扩训队列。** 新的多研究预训练、R0–R6、Qwen 扩训与跨研究评价尚未执行；当前服务调用已经完成的 PK1 权重。下载文件不等于训练准入。私有开发/预留角色保持原样，私有链接、名单、模型及服务日志不进入 GitHub。

## 1. 更新代码

在已有 AutoDL 项目目录运行。继续使用已验证的 CUDA 环境，不重新创建环境、不替换 PyTorch。

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/VirtualDiapauseCell
  git -c http.version=HTTP/1.1 pull --ff-only
  source .venv-autodl-vdc/bin/activate
  python -m pip install -e '.[pk1]'
  command -v screen
  python scripts/fetch_pk2_public.py --help
  python scripts/serve_pk2.py --help
)
```

若以前用离线包覆盖过文件，`git pull` 可能因本地修改停止。不要 `reset --hard` 或删除数据。可以直接把本次离线代码包解压覆盖到同一目录，然后从 `source .venv-autodl-vdc/bin/activate` 开始执行。代码包不包含数据或私有文件，无需重新上传原准备包。

缺少 screen 时执行 `apt-get update && apt-get install -y screen`。

## 2. 后台抓取公共数据

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
bash scripts/launch_pk2_acquire.sh
```

默认两个下载/处理 worker，BLAS 每进程限制两个线程，保留已有原始缓存。原始文件在 `data/raw/pk2_public/`；每次任务在独立 `runs/pk2_acquire_*`。13 个固定来源文件共约 190 MB，另需空间保存解析结果；不下载 FASTQ、不混入内部数据。

每个文件核对字节数及 SHA256，临时文件使用 `.partial`；大文件断点续传必须通过 Content-Range 检查。服务器不支持 Range 时明确从头下载。一次研究失败不会中断其他研究。相同原始缓存的抓取任务设互斥锁，避免同时写同一临时文件。

| 研究 | 实际核验内容 | 尚未自动完成的训练准备 |
|---|---|---|
| GSE221467 | 14 个样本、TPM BED；按 54,485 个 locus 对齐并保留缺失 mask | 重复 gene ID 的 locus 聚合策略、programme 和分组 |
| GSE202844 | 12 个样本；原始整数 counts（保留 ERCC）和按细胞数归一化 log2 两份视图；精确 GSM 映射 | 选定尺度、去除 spike-in 的训练策略、分组 |
| GSE303716 | 24 个样本元数据、两份 DESeq2 差异表 | **附件不是逐样本表达矩阵**；需要另外获取矩阵或处理原始测序，不能直接进数值预训练 |
| GSE124109 | 30 列 FPKM、精确 GSM 映射 | 已发表全样本处理的限制、programme、分组；时长不当 depth |
| GSE104616 | 24 列 log2 RMA；按明确 title 别名映射 GSM | 技术重复、原联合 RMA/ComBat 和 probe 映射；只作描述性比较候选 |
| GSE3169 | 94 个样本元数据，其中 Dauer MTC 42 张 array、4 条起始 pool 系列、两个平台 | probe→gene、协议时间映射与系列留出；42 张 array 不等于42个独立 pool |

所有输出保留 `training_admitted=false`、`training_executed=false`。尺度不互换，不把缺失测量改为实测零，不把差异统计量作为表达值。已有 GSE288723/GSE291659 保留原始角色，不重新分配已保留样本。GSE223093 仍是后续可选审计项，本脚本不自动下载它。

断网重试（同一代码/来源配置下）：

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
export VDC_ACQUIRE_RUN="$(cat runs/LATEST_PK2_ACQUIRE.txt)"
bash scripts/launch_pk2_acquire.sh
```

已完成研究先验证输出指纹再复用。代码/配置变更时使用新任务目录：`unset VDC_ACQUIRE_RUN` 后重新启动；原始文件缓存仍复用。若文件哈希变化，程序停止该研究，不能用新哈希自动覆盖来源锁。

## 3. 后台启动开发服务

使用服务器上完整的 PK1 run，必须包含 `best.pt` 等权重，报告 ZIP 不能替代它。若 `runs/LATEST_PK1.txt` 已指向正确 run，无需指定其他路径。

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
# 若指针不正确，先设置：export VDC_PK1_RUN=/你的完整PK1运行目录
bash scripts/launch_pk2_service.sh
```

启动过程：核对已批准角色和模型文件 → 复制不可覆盖的独立快照 → 启动两个临时服务进程逐一调用并验证重启一致性 → 常驻 `127.0.0.1:8765`。服务使用 CPU，模型按需加载，不占用 Qwen GPU。旧实验的所有已完成 K0–K5 / seed 模型保留原名称，不根据表现自动换成另一条路线。

快照位于 `$VDC_PRIVATE_ROOT/services/`。只有已存在的开发样本可以查询，不接受任意文件路径或预留样本。界面提供状态、clock、programme、观测 mask、覆盖率、训练均值基线和评价。仅当 encoder 指纹匹配时连接 programme wave。公共端点、转移和群体功能模型提供 API：

- `GET /health`、`GET /capabilities`：快照身份及可用模型。
- `GET /samples?view=bulk`：开发样本；`POST /analyse`：`{"model":"state_bulk_K0_42","observation_ids":["从 samples 接口选择的 ID"]}`。
- `GET /public_cases?module=endpoint`（或 `transition`）；`POST /public_response`：`{"module":"endpoint","observation_id":"返回的开发 case ID"}`。
- `POST /functional`：`{"hours_after_release":24,"history_days":4}`；只允许该模型有依据的小时和历史范围。返回组级 young-adult fraction，不是 expression-to-depth。

本次没有新建 Qwen adapter，不提供在线 Qwen 生成；数值模型中保存的语义 buffer 随实际 checkpoint 使用。Gene/TF 曲线、regulon 仍保存在原研究工件中，本次网页没有把它们扩成新在线预测功能。

用浏览器访问时，在自己电脑另开终端，按 AutoDL 控制台 SSH 主机和端口建立隧道：

```text
ssh -p 你的SSH端口 -L 8765:127.0.0.1:8765 root@你的SSH主机
```

保持该终端连接，在自己电脑打开 `http://127.0.0.1:8765/`。不需要把服务改成公网监听，也不需要把密码写进脚本。若本地8765被占用，可将 `-L` 第一处端口改为8766并打开8766。

## 4. 查看进度、停止与重启

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
.venv-autodl-vdc/bin/python scripts/pk2_status.py
screen -ls

# 下载进度；Ctrl+C 只退出查看
DATA_JOB="$(cat runs/LATEST_PK2_ACQUIRE.txt)"
tail -n 40 -F "$DATA_JOB/console.log"

# 服务进度：另开一个终端
WEB_JOB="$(cat runs/LATEST_PK2_SERVICE.txt)"
tail -n 40 -F "$WEB_JOB/console.log"
```

抓取结束显示 `acquisition_completed: 6`、`TASK_EXIT_CODE=0`；这只表示文件处理完成。失败研究在 `acquisition_status.json` 给原因。服务正常运行时不会出现退出码；`pk2_status.py` 会报告 `HTTP ready snapshot_matches True`。

停止服务：`screen -r 启动时打印的服务SESSION`，按 Ctrl+C；只想离开而保持运行，用 Ctrl+A 后按 D。重启同一冻结快照可使用：

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
WEB_JOB="$(cat runs/LATEST_PK2_SERVICE.txt)"
screen -L -Logfile "$WEB_JOB/restart.log" -dmS "vdc_pk2_restart_$(date -u +%H%M%S)" \
  .venv-autodl-vdc/bin/python scripts/serve_pk2.py serve \
  --snapshot "$WEB_JOB/snapshot" --port "$(cat "$WEB_JOB/port.txt")"
```

必须先停止旧实例。源代码或依赖改变时原快照拒绝加载，重新运行启动脚本冻结新快照。`exit_code.txt` 是最初 launcher 的历史退出码；重启后的现状以 HTTP 身份检查及 `restart.log` 为准。

## 5. 本地已执行及后续边界

本地已使用13个真实文件完成六组抓取/解析；已有真实开发数据的短预算 PK1 模型完成独立 HTTP 查询、停止、重启一致性检查。这个本地检查不是对服务器完整训练结果的重新评价，也没有启动 AutoDL 服务。

下一步是根据抓取后的实际矩阵和元数据完成 grouped split、programme 映射及来源准入，接入多研究预训练与扩展知识训练。当前下载和服务命令不会自动执行这些尚未实现的训练任务。
