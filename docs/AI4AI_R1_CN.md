# VDC AI4AI 第一轮：假设驱动研究

研究 LLM 根据已有知识提出假设、竞争解释、预期及反证条件，生成实验实现，读取真实数值评价并修订。clock 是可选参考，不是被规定正确的模型。

这是 VDC 自有顺序控制器，借鉴 EvoScientist 的实验记忆和 PiEvo 的假设修订思路。**没有运行上游 EvoScientist、PiEvo 或 OpenEvolve 算法，不以复现它们命名。** 保留现有 INT1 应用，复用数据契约、目标隔离、Ridge、分组权重、运行锁与工件工具。

## 实际范围

- 默认复用已授权 bulk/core/coarse 开发缓存；任务是由目标排除后的表达、programme、已知身份预测固定隐藏基因。不能据此声称未来状态、因果干预、真实 biological time 或 depth 验证。
- 每轮实际调用 API 两次：提出假设及实现；依据评价结果反思。默认 12 轮，最多 30 次调用，失败及重试也计入。输入 UTF-8 字节数加输出额度作为保守 token 预算，默认上限 1000000；实际账单以服务商为准。
- 默认 graph 后端允许 AI 组合 PCA、多维状态、非线性基函数、身份交互及直接预测，最终拟合加权 Ridge。它是**受限数值图**，不是任意架构搜索。超出能力的建议记录为未检验。最终 Ridge 标准化特征，单纯 scale 不构成有效新假设。
- 可选 python_docker 允许 AI 写完整 fit/predict Python 程序；服务器必须支持 Docker。fit 只见训练目标，predict 只见输入及模型；无网络、GPU、密钥和主机研究目录。不支持时不在宿主直接执行。
- 特征选择与所有预处理只在 train 拟合，目标不参与输入归一化、rank、programme 或参考轴。pool、共享对照、cohort 连通分组。所有候选评价相同查询与目标，不接受缺失预测换取较低误差。
- 当前 validation 会反复用于研究，全部结果标记 adaptive development，不能称为新独立测试。未改变预留角色。

语料是 `knowledge/pk2/literature.jsonl` 中 reviewed 的 train 证据，按来源和文本去重，排除 QA completion。同时读取 `knowledge/core_tasks.json` 的 thesis 任务、不同 biotime 坐标定义和 waves/depth 区分，标作研究定义而非已证实结论。它们**不是完整 thesis 和全量文献**。可用 `--corpus` 指定已审核同格式 JSONL；换语料须新 run。当前是本地词汇检索；模型建议的 literature_queries 是待查问题，不假装已联网搜索。

API 接收证据、去标识数据规模、假设、代码和聚合指标，不上传表达矩阵、样本编号、私有链接或逐样本预测。密钥仅存在服务器环境变量，不发聊天、不写配置、不提交 Git。

## AutoDL 安装

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
git -c http.version=HTTP/1.1 pull --ff-only
source .venv-autodl-vdc/bin/activate
python -m pip install -e '.[pk1]'
export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
export VDC_PYTHON="$PWD/.venv-autodl-vdc/bin/python"
export PYTHONPATH="$PWD/src"

# 服务商的 base URL，不加 /chat/completions；使用真实模型ID。
export VDC_RESEARCH_API_BASE='https://api.openai.com/v1'
export VDC_RESEARCH_MODEL='gpt-6.1-sol'
read -rsp 'Research API key: ' VDC_RESEARCH_API_KEY; echo
export VDC_RESEARCH_API_KEY

# 默认读取 LATEST_STAGE.txt 对应的 stage/data；必要时明确指定：
# export VDC_RESEARCH_DATA=/你的/cw_stage运行/data
python scripts/run_research.py doctor
```

doctor 不调用 API。官方 OpenAI 起步配置为 gpt-6.1-sol、medium、max_completion_tokens=16000（包括推理和可见输出）、300秒超时。模型须支持 chat/completions 与文本 JSON；默认不使用工具调用。旧兼容服务可显式配置 completion_token_parameter=max_tokens、reasoning_effort=null，不自动降级。若服务商不支持 response_format，可在启动前把 `configs/research_round1.json` 的 `json_response_format` 改为 false；仍校验完整 JSON，不从半截输出抠字段。不兼容其他协议时先适配，不自动换模型。

首次真实连接检查只发一条简单 JSON 请求，不发送研究数据，可能产生少量费用。必须在同一个已设置密钥的终端执行：

```bash
CHECK_DIR="$VDC_PRIVATE_ROOT/runs/research_api_check_$(date -u +%Y%m%dT%H%M%SZ)_$$"
python scripts/check_research_api.py --out "$CHECK_DIR"
export VDC_RESEARCH_API_CHECK="$CHECK_DIR/status.json"
```

检查完整 JSON 和随机校验字段，记录 requested/returned model、实际 token 用量。检查通过只证明连通和协议，不等于科学实验成功。启动器和冻结副本离线核验此记录，不额外调用 API；换模型、地址、生成参数或接口代码必须重新检查。启动失败先查状态，不重复执行付费检查。

失败时查看 `$CHECK_DIR/api/connection_001.json` 的 http_status 和 diagnostic：401 检查密钥；403 检查账号/地区访问资格；404 检查模型权限与地址；429 检查额度和速率；400 检查参数；output_limit_reached 是输出截断。网络超时也保留已提交记录，先检查平台用量；不自动重试或改用其他模型。HTTP 重定向被拒绝，不转发密钥。

参考：[模型](https://developers.openai.com/api/docs/models/gpt-6.1-sol)、[接口参数](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)。密钥只填在服务器环境中。

API 准备好前可仅准备数据及数值基线：

```bash
python scripts/run_research.py prepare \
  --run "$VDC_PRIVATE_ROOT/runs/research_prepare_$(date -u +%Y%m%dT%H%M%SZ)"
```

它明确记录 prepared_no_api_calls，不是 AI 研究轮。正式轮另建 run。

## 后台启动和进度

### DeepSeek 官方 API

官方文档核对日期：2026-10-11。使用 `https://api.deepseek.com` 和 `deepseek-v4-pro`，
思考开启、high、max_tokens=16000；不沿用 OpenAI 密钥或其余额。
参考：[官方入口](https://api-docs.deepseek.com/)、[参数与 JSON 输出](https://api-docs.deepseek.com/api/create-chat-completion/)。
这是同一研究控制器的配置选择，不另建数值管线；旧 OpenAI 失败记录保留，新建 DeepSeek run。

```bash
# 在项目目录且已激活原 venv、设置 VDC_PRIVATE_ROOT/PYTHONPATH 后执行。
export VDC_RESEARCH_API_BASE='https://api.deepseek.com'
export VDC_RESEARCH_MODEL='deepseek-v4-pro'
export VDC_RESEARCH_CONFIG="$PWD/configs/research_deepseek.json"
unset VDC_RESEARCH_API_KEY
read -rsp 'DeepSeek API key: ' VDC_RESEARCH_API_KEY; echo
export VDC_RESEARCH_API_KEY

CHECK_DIR="$VDC_PRIVATE_ROOT/runs/deepseek_api_check_$(date -u +%Y%m%dT%H%M%SZ)_$$"
python scripts/check_research_api.py --out "$CHECK_DIR"
# 仅检查通过后继续；失败不自动重试。
export VDC_RESEARCH_API_CHECK="$CHECK_DIR/status.json"
export VDC_RESEARCH_RUN="$VDC_PRIVATE_ROOT/runs/research_deepseek_$(date -u +%Y%m%dT%H%M%SZ)_$$"
unset VDC_RESEARCH_RESUME VDC_RESEARCH_RETRY_API_ERRORS
bash scripts/launch_research.sh
```

启动器把选中配置复制为 release/configs/research_selected.json，worker 只读冻结副本。
恢复此 run 时将 VDC_RESEARCH_CONFIG 指向该冻结文件，使用同一服务商及模型并保留检查记录。
只从最终 content 解析完整 JSON；reasoning_content 不当作提案，也不会从它抽取答案补救空输出。
检查失败日志现在同时保存 network_reason_type、network_errno、certificate_verify_code，
不记录原始异常文本、代理地址或凭据。本地模拟测试不证明 AutoDL 已连通；需要实际一条检查请求。

```bash
bash scripts/launch_research.sh

RUN="$(cat "$VDC_PRIVATE_ROOT/LATEST_RESEARCH.txt")"
python scripts/research_status.py --run "$RUN"
tail -n 60 -F "$RUN/console.log"
screen -ls
```

启动器保存代码和配置快照，在 screen 运行。Ctrl+C 只结束 tail。启动前可调整轮数、上下文和预算；修改 prompt、代码、数据、语料、协议或模型后要新 run。

以 status.json、exit_code.txt 为准。completed_budget 表示循环走完，不代表假设成立；同时查看 evaluated、reused_equivalent、failed/invalid 数量，各视图误差、覆盖率、独立材料数、动态幅度及反思。图后端在服务器真实数据拟合；研究 LLM 的 API 推理在服务商端执行，不要求本机 GPU 满载。

## 恢复与打包

```bash
export VDC_RESEARCH_RUN="$(cat "$VDC_PRIVATE_ROOT/LATEST_RESEARCH.txt")"
export VDC_RESEARCH_RESUME=1
# API失败或投递结果不确定时，不自动重复付费请求。
# 核查 api/ 与账单后，要重试才设置：
# export VDC_RESEARCH_RETRY_API_ERRORS=1
bash "$VDC_RESEARCH_RUN/release/scripts/launch_research.sh"

python scripts/pack_research_report.py --run "$VDC_RESEARCH_RUN"
```

恢复使用原 release 和签名；已完成的数值实验在反思失败后复用。原数据、模型ID、API地址必须相同，密钥可正常轮换。报告保存完整 run 名、状态与 SHA；重新打包不会称新实验。ZIP 含私有预测与 API 轨迹，仅供内部协作，省略输入矩阵及模型文件；模型保留各 experiment/view/model。

## 可选完整 Python 实验

仅在 docker info 成功的机器上：

```bash
docker build -t vdc-research-candidate:r1 -f docker/research.Dockerfile .
```

启动前设置配置 allow_python_docker=true。runner 锁定实际 image SHA，不自动拉取新镜像。容器受 CPU/RAM/时间限制；宿主磁盘配额需部署管理员提供。首次开启需小预算验证容器执行。默认 graph 不依赖 Docker，不把它称为自由 Python 搜索。两种后端的搜索空间不同，结果分别记录。

## 科学验收

至少一次真实 proposal→fit→保存重载→固定评价→reflection→根据历史再提案，文件齐全。软件模拟 API 的测试不计为真实研究。实验失败不能当作反对生物学假设的证据；只有固定数据上的预测改善也不能当作机制证明。超出当前 assay 的假设保留待验证，旧应用及负结果不删除。
