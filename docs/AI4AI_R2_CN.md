# R2：稳定研究引擎，开放 universal diapause / dormancy 模型研究

本轮目标是跨背景可复用的休眠／静息／恢复模型。Universal 是待检验的适用性，
不预设所有物种共享同一 clock、也不保证最终是单一神经网络。最终可以是共享表示，
也可以是统一入口下的多个背景专家。模型是否保留 Qwen、clock、programme 或当前
基因面板，由实验决定。

R2 复用 VDC 的 API 客户端、付费调用台账、真实缓存、下载器、分组权重、Ridge 和
结果格式。R1 保留为历史入口。R2 使用新的工具控制器，不再限制候选为固定算子图。
这是 VDC 的实现，不是 EvoScientist、AI Scientist-v2、AlphaEvolve 或 PiEvo 的完整复现。

## 固定什么，开放什么

固定四个交付问题：样本状态及差异的表达依据；维护／释放／干预变化及合法预测；
跨背景迁移与校准；实际可加载、可计算、可导出的模型。固定一次比较的输入、目标、
单位、分组、覆盖率、指标和对照，不固定科学结论。

AI 可以调用真实文献检索、全文／公开作者代码读取、公共文件续传、数据准备、依赖
缓存、多文件模型训练、父候选分支、比较、全任务数据重拟合和模型导出。准备工具
直接挂载选中的真实资产，不再仅把公共研究作为文字背景。研究主控选择实验顺序。
结构检查可分页读取压缩包目录、H5/NPZ 形状和表字段，不把私有表达数值发给 API。

全部已授权自有数据和18个公共来源进入 R2 主动研究范围，旧角色保留为历史信息，
不再调用 R1 的旧 holdout 准入过滤器。原始文件和原角色表不修改。每个具体任务
仍要保护预测目标、保持真实 pool／胚胎／培养／共享对照关系；结果是适应性开发或
回顾性比较，不重新声称独立盲测。

数值评价目前支持完整覆盖的二维多输出回归，主指标可选按独立单位等权的 MSE 或
MAE，自动运行同输入 Ridge 和训练均值对照。其他分析可以自由运行并导出工件，
不冒充已接入同一数值排行榜。预处理代码由 AI 编写并保留，任务契约不是生物学
标签有效性的自动证明。源文件已挂载也不自动证明每个实验臂被正确使用。

## AutoDL 运行环境

AutoDL 普通容器不支持嵌套 Docker。因此默认是 Linux chroot + 非 root 身份 +
Landlock 文件访问规则 + seccomp，Docker 保留为显式替代后端。
需要 Linux x86-64、Landlock ABI >= 1、实例 root 的 CAP_SYS_CHROOT 和设备创建能力。
启动时实际验证这些条件；不支持则明确停止，不切换到无隔离的宿主机执行。
工具不依赖 user namespace，也不需要购买裸金属。

候选只看本工具挂载的输入；训练和预测分别运行，预测端没有标签、原始源目录、
控制器 API 密钥或出站 socket。模型与运行库只读，候选只能写临时目录和输出。
公开文献由控制器下载；依赖下载在没有数据挂载的网络进程中进行。候选 stdout
保留在服务器，不整段发送外部 API，失败反馈包含代码位置与结构化错误。

运行库是现有 Python 环境的隔离快照，优先硬链接、跨文件系统时复制，需要额外
磁盘空间。不会重新安装或替换现有 CUDA Torch。默认每个执行工具最多8 CPU、
64 GiB进程组驻留内存阈值、最长2小时；同一时刻一个工具。CPU还有逐进程虚拟
地址空间限制；GPU取消该地址空间限制，避免误拦CUDA的大地址预留。驻留内存
由父进程采样，包含共享页重复计数，并非cgroup硬配额。

默认 `VDC_R2_DEVICE_POLICY=ai`：先检测实际显卡和CUDA，在隔离环境执行FP32
训练、支持时执行BF16前后向、保存及新进程重载；全部通过后，AI获得CPU和CUDA
两种选择。每条prepare/experiment使用 `device: cpu|cuda` 和选择理由，神经网络
或大张量可选GPU，IO/小型Ridge可选CPU。导出与新请求沿用实验选择的设备。
只允许一条GPU任务；不会因为GPU空闲而制造计算。

GPU检查失败会停止启动，不静默降级。CPU模式需显式设置
`VDC_R2_DEVICE_POLICY=cpu`。旧 `VDC_R2_GPU` 开关已由此策略替代。
执行记录含选择设备、耗时、驻留内存及候选进程实际观察到的显存分配；短任务
可能未被nvidia-smi采样捕获，此时标明未观察到，不冒称已经使用GPU计算。
DeepSeek研究决策仍调用远程API，本机GPU用于AI编写的数值模型和数据计算。

## 1. 安装和实际隔离检查

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/VirtualDiapauseCell
  git -c http.version=HTTP/1.1 fetch origin main
  REV="$(git rev-parse origin/main)"
  RELEASE="/root/autodl-tmp/vdc-releases/r2_gpu_${REV:0:12}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
  mkdir -p "$RELEASE"
  git archive "$REV" | tar -x -C "$RELEASE"
  printf '%s\n' "$REV" > "$RELEASE/RELEASE_COMMIT.txt"
  export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
  export VDC_PYTHON="$PWD/.venv-autodl-vdc/bin/python"
  export VDC_R2_DEVICE_POLICY=ai
  export VDC_R2_RUNTIME_ROOT="$VDC_PRIVATE_ROOT/r2_runtime_devices_v2"
  unset VDC_R2_RUN VDC_R2_RESUME VDC_R2_GPU VDC_R2_RETRY_API_ERRORS
  bash "$RELEASE/scripts/start_research_r2.sh"
)
```

使用fetch和git archive得到固定提交，不覆盖旧checkout的本地改动。缺少密钥时
在服务器终端隐藏输入；自动寻找并核验已有成功API记录，不发送新的付费检查。
安装和环境验证在screen后台运行，通过后自动派发研究，不必再次调用launch。
已有旧研究运行时不要同时新开第二条GPU队列。

查看安装日志：`tail -n 60 -F "$(cat /root/autodl-tmp/vdc-private/LATEST_R2_SETUP.txt)"`。
环境检查应看到 `status: passed`、`scope: synthetic_software_only`，随后输出
研究的SCREEN和RUN。这是实际软件连接检查，不是生物学效果。
若报 Landlock/chroot 权限错误，把 setup 日志给维护者；不要改成裸 `exec`。
如需重建运行库，使用新的 `VDC_R2_RUNTIME_ROOT`，不覆盖旧快照。

代码离线包与仓库内容相同，可解压到独立目录执行，不必覆盖服务器有改动的 checkout。
独立目录中应显式设置 `VDC_PYTHON` 为上面的绝对路径。

## 2. API 与资料检查

以下是手动诊断入口；使用上面的start脚本时已经自动执行，无须另开一次研究。

沿用服务器上的 `VDC_RESEARCH_API_KEY`、`VDC_RESEARCH_API_BASE` 和
`VDC_RESEARCH_MODEL`。密钥不写代码、配置或聊天。生成参数与 R1 DeepSeek profile
相同，因此未改客户端时可复用原成功 API receipt，不重新付费测试。

```bash
export VDC_RESEARCH_CONFIG="$PWD/configs/research_r2.json"
export VDC_RESEARCH_API_CHECK=/root/autodl-tmp/vdc-private/runs/research_api_check_20261011T185318Z_9473/status.json
DATA_RUN="$(cat "$VDC_PRIVATE_ROOT/LATEST_RESEARCH_DATA.txt")"
export VDC_RESEARCH_ASSET_REPORT="$DATA_RUN/status.json"
"$VDC_PYTHON" scripts/check_research_api.py --check "$VDC_RESEARCH_API_CHECK"
"$VDC_PYTHON" scripts/run_research_r2.py doctor
```

Doctor 列出实际存在的私有与公共文件。R2 不要求所有来源齐全才开始工作；工具选中
公共文件后会核验锁定哈希。默认扫描私有 `input/archive/raw/sources/data/prepared`
等目录、角色表，以及已有 stage 的数值缓存。若原始全资料位于别处，设置
`VDC_R2_EXTRA_DATA_ROOT` 为该资料目录；不要把含密钥的 home 目录作为资料根目录。
未上传到这台实例的资料仍明确缺失，不因为用户曾上传 Drive 就视为服务器已有。

## 3. 后台启动和恢复

```bash
"$VDC_PYTHON" scripts/run_research_r2.py status
RUN="$(cat "$VDC_PRIVATE_ROOT/LATEST_RESEARCH_R2.txt")"
tail -n 60 -F "$RUN/console.log"
```

每个决策可提交最多4个顺序工具动作，一条失败不取消其他独立动作。按实际预算持续
选择下一步，不再以12轮宣布项目完成。本轮按用户要求扩大为：**100次请求、600万 token
保守预算**，两者先到即停止；至少24次研究决策后才允许主动完成。决策次数不等于
成功实验数，失败与实际改进分别记录，不为凑轮数重复训练。上下文按字节保守计入台账，
不等于供应商真实收费 token。预算耗尽保存部分成果与未完成范围。
600万是100次请求各最多6万预算单位的保守上限，包含输入字节及输出预留，
不表示会实际消耗或收费600万token；实际用量另见API台账。
对于供应商已经完整返回但格式不合法或达到输出长度的回答，保留已付费记录，
启动新的缩短回答决策，连续3次失败则停止。网络超时、鉴权失败或未知送达结果
不会自动重发。无需人工修正每个模型代码错误，独立工具和后续决策仍继续运行。

断线不影响 screen。真正进程退出后可恢复：

```bash
export VDC_R2_RUN="$(cat "$VDC_PRIVATE_ROOT/LATEST_RESEARCH_R2.txt")"
export VDC_R2_RESUME=1
bash scripts/launch_research_r2.sh
```

已经提交但结果不确定的 API 不自动重试。核对服务商与台账后，明确设置
`VDC_R2_RETRY_API_ERRORS=1` 才重试。已完成动作复用；中断中的动作保存为 interrupted，
由 AI 检查后用新 ID 修复，不自动重复长训练。源码、任务、配置或运行库变更使用
新版本，不能把旧 completed 当作新结果。

## 4. 输出、打包和新请求推理

`inventory.json` 是本地资产与来源去向；`decisions/` 是实际决策与上下文；
`actions/` 是代码、执行记录和模型；`tasks/` 是冻结比较及强基线；
`literature/` 是真实取回的公开资料；`exports/` 是实际导出的专家及重载记录。
总状态不会把缺响应、缺迁移或仅登记资料记成 universal model 已完成。

```bash
RUN="$(cat "$VDC_PRIVATE_ROOT/LATEST_RESEARCH_R2.txt")"
"$VDC_PYTHON" scripts/pack_research_r2.py --run "$RUN"
```

报告包包含实际参数和变换，修复 R1 只打包预测而漏模型的问题。报告含私有结果，
不要提交公共 GitHub。原始输入不打包；模型可能含拟合信息，仍按私有材料保管。

统一推理入口（`EXPERT` 换成 exports/index.json 中实际导出的 ID）：

```bash
"$VDC_PYTHON" scripts/run_research_r2.py infer \
  --bundle "$RUN/exports/$EXPERT" \
  --input /path/to/new_query.npz \
  --out /root/autodl-tmp/vdc-private/new_request_001
```

输入是对应任务 `schema.json` 规定的二维 `x`，特征顺序、尺度和上下文必须匹配。
实际启动新进程加载模型，输出预测 NPZ、CSV、JSON 和 HTML；不查找旧 validation 预测。
通用 raw 表达到各专家 x 的转换尚需研究中的准备实现与验证，不能把这个数值入口
宣传为已验证的任意物种 raw 矩阵一键推理。导出会保存准备来源与固定契约。

全量重拟合仅指所选任务中兼容的 fit/query 行，不等于所有项目数据混合训练。
新模型得分、跨物种有效性以及论文算法移植效果都要等服务器实际运行后判断。

## 隔离环境启动权限修复

若旧版在 `probe.log` 中出现 `exec: Permission denied`，表示隔离启动器尚未成功
执行 Python，不是模型训练失败。旧复制逻辑在系统盘到数据盘的跨盘复制中会丢失
可执行位；复制目录还可能继承不允许降权用户访问的权限。

v2 保留可执行文件权限，并只修正隔离副本的目录访问权限，不修改宿主文件，
不关闭 Landlock、降权或网络隔离。使用新的 `r2_runtime_devices_v2` 路径重建环境；
旧 snapshot 与检查凭证不能继续使用。启动脚本会先重新执行 CPU 隔离检查和
真实 GPU 前后向、保存重载检查，通过后才启动研究循环。旧数据和 R1 结果保留。

## 方法来源

- EvoScientist：研究、工程与长期经验记忆，https://arxiv.org/abs/2603.08127
- AI Scientist-v2：实验树与分支探索，https://arxiv.org/abs/2504.08066
- AlphaEvolve：可执行代码与外部评价，https://arxiv.org/abs/2506.13131
- PiEvo：证据冲突时扩展假设空间，https://arxiv.org/abs/2602.06448
- AutoDL 容器环境说明：https://www.autodl.com/docs/env/

上游思想与 VDC 自有实现分别标注。没有安装上述上游框架，也没有把提示中的角色名
当成已实现的算法。本地软件测试、CI 实际隔离测试、AutoDL 环境探测和真实研究结果
是四种不同证据，分别记录。
