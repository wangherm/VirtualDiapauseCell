# AI4AI：补齐公共资产，沿用现有私有开发数据

本轮不重新下载私有数据、不重跑 PK2，也不启动全量新训练。
先补齐清单中公共资产，再把来源、设计、限制和实际核验状态交给研究 LLM。

## 下载范围

`configs/research_public_assets.json` 固定 44 个文件：17 个 GEO 研究的 40 个文件
（501,315,037 字节），以及 PRJEB80774 已选定的 4 个 FASTQ
（4,401,476,858 字节）。约 4.90 GB；已有缓存能进一步减少下载量。
这四个 FASTQ 只是两个 paired-end run，并非整个 ENA 项目，也不提供充分生物学重复。
本轮不自动扩展到目录中的 optional accession 或整个 SRA/ENA 原始读段集合。

GEO 文件来自 NCBI，FASTQ 来自 ENA。GEO 按 SHA256 校验，ENA 按其 MD5 锁核验后
额外保存观察到的 SHA256。大小相同不足以复用。校验失败保留文件并明确报错，
不替换锁、不覆盖旧缓存。下载复用已有 `prepare_public_pilot.fetch_locked` 断点续传实现。

来源记录来自本轮提供的公共 acquisition 清单及仓库已有锁；不是声称已在服务器核验。
GSE221467 的新清单 metadata 哈希与旧锁不同，而且原汇总记为 1/2 文件成功：
本轮保留已有锁，并于本地从 NCBI 实际复核匹配，不盲目采用冲突记录。
所有文件均在服务器运行时再次核验。

`knowledge/research_public_data.json` 是研究设计说明。GSE185815 继续保持历史
holdout/excluded 身份；其他 accession 的下载也不自动修改旧分组。
Counts、estimated counts、FPKM/TPM、标准化值和微阵列比值不能混为 counts。

## 1. 更新后启动后台补全

在原服务器仓库目录运行；若 GitHub 网络不通，可将本轮源代码离线包解压到独立目录，
从该目录运行同一脚本，保留原 Python 环境和缓存路径即可。

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/VirtualDiapauseCell
  git -c http.version=HTTP/1.1 pull --ff-only

  export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
  export VDC_PYTHON=/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python
  export VDC_PUBLIC_CACHE=/root/autodl-tmp/VirtualDiapauseCell/data/raw
  export VDC_DOWNLOAD_WORKERS=2
  bash scripts/launch_research_data.sh
)
```

无需安装新的大型依赖或替换 CUDA/PyTorch。需要已有 `screen`。
如果还有另一个已有公共缓存目录，启动前设置 `VDC_EXTRA_PUBLIC_CACHE`。
新下载放在 `$VDC_PRIVATE_ROOT/public_assets/`；缓存复用记录原文件路径，不再复制一份。
新 run 冻结脚本与清单，使用 `screen`，与 SSH/JupyterLab 连接断开无关。

## 2. 进度和断点

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
PY="$PWD/.venv-autodl-vdc/bin/python"
"$PY" scripts/research_data_status.py
DATA_RUN="$(cat /root/autodl-tmp/vdc-private/LATEST_RESEARCH_DATA.txt)"
tail -n 40 -F "$DATA_RUN/console.log"
```

`Ctrl+C` 只退出看日志。最终必须是 `status: completed`、`verified: 44`、
`TASK_EXIT_CODE=0`。部分文件失败不会取消其他独立下载。
`REPORT_CN.md` 逐项列出复用、下载或失败原因。

网络中断后可再次启动；新 run 会重验并复用文件及 `.partial` 断点。
若要恢复原冻结 run，设置 `VDC_RESEARCH_DATA_RUN="$DATA_RUN"` 后再次运行启动器；
只可在旧 worker 已退出后恢复。代码或清单有修复时用新 run，以免仍调用旧 release。

## 3. 接入 AI4AI 前检查（不花 API token）

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
export VDC_PYTHON="$PWD/.venv-autodl-vdc/bin/python"
export PYTHONPATH="$PWD/src"
export VDC_RESEARCH_CONFIG="$PWD/configs/research_deepseek_data.json"
DATA_RUN="$(cat "$VDC_PRIVATE_ROOT/LATEST_RESEARCH_DATA.txt")"
export VDC_RESEARCH_ASSET_REPORT="$DATA_RUN/status.json"

# 默认由 LATEST_STAGE.txt 找现有开发缓存；也可显式设置 VDC_RESEARCH_DATA。
"$VDC_PYTHON" scripts/run_research.py doctor
```

此步骤核对私有三个缓存视图及 sample_roles 指纹，并重验 44 个公共文件。
它不证明其他所有私有原始档案均已上传；也不下载私有文件。
如果报私有缓存缺失，应检查实际目录或设置 `VDC_RESEARCH_DATA`，不要用随机数据替代。

## 4. 保持服务器 API 变量后启动研究

继续使用服务器上设置的 DeepSeek key、API base、model，不把密钥写进代码或聊天。
新 profile 继承既有 DeepSeek 推理配置、预算和端点锁，并要求公共文件核验完成。
为给新增数据说明留出空间，提示中保留最近两轮详细反馈、更早各轮的简要结果；
不删除失败轮次，完整历史仍在 run 工件中，API 总预算不增加。
若已有同端点、同模型、同生成参数的成功 API check，可复用其 `status.json`；
否则显式做一次付费连通性检查：

```bash
(
  set -euo pipefail
  : "${VDC_RESEARCH_API_KEY:?请先在服务器环境变量中设置 DeepSeek API key}"
  : "${VDC_RESEARCH_API_BASE:?请沿用已配置的 DeepSeek API base}"
  : "${VDC_RESEARCH_MODEL:?请沿用该 profile 对应的模型}"
  if [[ -n "${VDC_RESEARCH_API_CHECK:-}" ]]; then
    "$VDC_PYTHON" scripts/check_research_api.py --check "$VDC_RESEARCH_API_CHECK"
  else
    CHECK="$VDC_PRIVATE_ROOT/runs/research_api_check_$(date -u +%Y%m%dT%H%M%SZ)_$$"
    "$VDC_PYTHON" scripts/check_research_api.py --out "$CHECK"
    export VDC_RESEARCH_API_CHECK="$CHECK/status.json"
  fi
  bash scripts/launch_research.sh
)
```

启动器复制公共核验回执到新 run，研究 manifest 绑定数据说明哈希；修改说明、清单或
实际输入不能沿用旧 completed 状态。研究进度仍用 `scripts/research_status.py`。
下载器不会自行消费 API token。

## AI 实际拿到什么、还不能做什么

每轮 proposal 上下文会加入公共研究的设计、尺度、独立单位要求、限制及实际核验文件数。
不会向 API 发送本地路径、Drive 链接、私有原始矩阵或预留样本结果。
`public_data_context.json` 保存实际使用的说明；说明改变会使旧 run 无法静默续跑。

**当前可执行的数值任务仍是现有 bulk/core/coarse 开发视图上的隐藏基因预测。**
这轮补的是数据资产和研究说明，并没有给三视图执行器新增跨研究矩阵解析器。
LLM 可以据此提出公共研究验证方案，但缺少样本映射、分组准入、基因映射或 FASTQ
定量的方案必须记为待执行，不能算公共数据已进入训练。后续接哪一组数据应依据
研究问题明确接入，保持 study/biological unit 分组及已有 holdout 边界。

当前软件测试使用故障注入和模拟 API；不是服务器下载成功或真实 AI 自迭代成功的证据。
