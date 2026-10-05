# CW1 定向修订：短身份证据合同与残差读出

沿用已完成 CW1 的数据、身份参考、数值模型和样本角色，在**新 run** 中运行。使用原 `run_clock_wave.py`，不重新执行 PK2、不重拟合 clock、不训练新 LoRA。新 GPU 推理仍需在 AutoDL 实际执行；本地代码测试不代表服务器完成。

## 本轮执行范围

九个开发 pool 的默认队列共 61 项：51 项逐文件校验后复用，10 项新增执行。新增为 Base/Domain 两项推理、两项身份链重评、三种视图残差读出、三种视图支持范围诊断。复用数与新执行数分别显示；旧 run 不修改。

- Qwen 原卡每模型 129 条，合计 258 条重新生成，旧长 JSON 不修补或重新计分。
- 每模型另对同样卡运行无数值提示、证据挑战两种条件；总共 774 条新生成。每个模型的前 3 张原开发卡用于工程 preflight，结果复用，不重复计数。三张均需 JSON/schema/证据引用有效，明确 unknown 也可通过；不要求生物学正确或被数值门槛接受。失败只阻止该模型及依赖链，CPU 继续。
- 短身份别名按类别名固定映射，候选展示顺序按 query 哈希固定随机化；无提示时不提供 gate、排名或距离。输出最多 3 个 observed marker slot，零表达与 reference-only marker 不能被引用。别名和 slot 表保存在每条记录。
- 原卡用于保守确认链，不能绕过原数值支持门槛；无提示条件独立评分，不把原数值 gate 偷偷用作模型答案。挑战按 query 哈希预定为遮盖每第四个观测 marker，或删除数值第一候选；不使用真标签构造输入。它是证据压力测试，不是新的模糊生物学样本或 lineage gold。
- 原始答案、完整 prompt、输入/输出 token 数、EOS 是否实际出现、长度上限、JSON/schema/证据状态分开记录。不知道的停止原因记为 unknown，不猜测。

## 数值对照与科学边界

冻结 C2，比较 identity mean、clock＋identity、clock＋identity＋可见 programme residual，以及匹配 identity/rank/amplitude 的直接 Ridge。新增读出只使用原 clock readout 的训练行，固定 alpha=1；同时保存按同一训练行重新拟合的 direct Ridge，避免原 direct baseline 使用更广训练行造成误解。

训练 residual 是原训练内参考的观测偏离，**不是 cross-fit residual**。范围外诊断使用显式线性外推；范围内、外推、共同有限预测分别计分，按生物学单位宏平均，保存所有 query 的覆盖率和动态幅度。未知身份不补数值。目标面板与输入排除规则不变；改变隐藏基因 counts 的数值不变性检查覆盖安全表达、编码特征和 clock。源注释聚合的身份依赖仍单独声明。

支持表按条件与身份拆开，Late 维持单列，不能因为落入范围就解释为正常 Exit。每个身份轴独立校准，不能当作统一的跨类型生理刻度。训练单位锚点重采样固定原轴基因和尺度，只诊断锚点均值敏感性；**不是完整模型置信区间**，单一 anchor pool 会明确标为缺乏重复。部署范围不扩张、不裁剪。保存 C0/C1/固定 Early offset 数组，检验 C1 是固定平移。

热应激原始 counts、后期身份修复、跨物种 mapping、expression-to-depth 不在本次完成声明中；不查询或改变预留角色。

## AutoDL 更新并启动

在原仓库目录执行。使用 `fetch + git archive` 创建干净发行目录，避免之前离线覆盖代码造成的 `git pull` 冲突。只下载代码，无须重新上传数据包。

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/VirtualDiapauseCell
  export VDC_SOURCE_REPO="$PWD"
  export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
  export VDC_PYTHON="$PWD/.venv-autodl-vdc/bin/python"
  # 第一次启动前固定上一轮 CW1；不要改成 PK2 或当前修订 run。
  export VDC_CW_REVISION_SOURCE="$(cat "$VDC_PRIVATE_ROOT/LATEST_CLOCK_WAVE.txt")"
  test -f "$VDC_CW_REVISION_SOURCE/config.json"
  git -c http.version=HTTP/1.1 fetch origin main
  BOOT="$(mktemp /tmp/vdc-cw-revision-XXXXXX.sh)"
  git show origin/main:scripts/install_clock_wave_revision.sh > "$BOOT"
  # 安装器从同一最新版本读取 release 安装逻辑。
  BOOT_DIR="$(mktemp -d /tmp/vdc-cw-bootstrap-XXXXXX)"
  cp "$BOOT" "$BOOT_DIR/install_clock_wave_revision.sh"
  git show origin/main:scripts/install_clock_wave_release.sh > "$BOOT_DIR/install_clock_wave_release.sh"
  bash "$BOOT_DIR/install_clock_wave_revision.sh"
)
```

使用现有 Python/CUDA 环境，不重新安装 PyTorch。后台使用 screen，SSH/Jupyter 断开不会终止任务。原 CW1 必须完整保留，复用校验失败会明确报错，不能静默重新拟合替代。

如果 GitHub 无法访问，将本轮离线**代码包**解压到新目录，进入该目录后执行：

```bash
export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
export VDC_PYTHON=/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python
export VDC_CW_REVISION_SOURCE="$(cat "$VDC_PRIVATE_ROOT/LATEST_CLOCK_WAVE.txt")"
export VDC_CW_SOURCE="$("$VDC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$VDC_CW_REVISION_SOURCE/config.json")"
export VDC_CW_RUN="$VDC_PRIVATE_ROOT/runs/clock_wave_revision_$(date -u +%Y%m%dT%H%M%SZ)_$$"
bash scripts/launch_clock_wave.sh
```

## 进度、恢复和打包

```bash
export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
export VDC_PYTHON=/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python
RUN="$(cat "$VDC_PRIVATE_ROOT/LATEST_CLOCK_WAVE.txt")"
RELEASE="$(cat "$VDC_PRIVATE_ROOT/LATEST_CLOCK_WAVE_RELEASE.txt")"
export PYTHONPATH="$RELEASE/src"
"$VDC_PYTHON" "$RELEASE/scripts/clock_wave_status.py" --run "$RUN"
tail -n 50 -F "$RUN/console.log"
```

`Ctrl+C` 只退出 tail。Qwen 逐卡进度保存在 `tasks/qwen_base/identity_original.json` 等文件，逐条件完成信息在 `logs/qwen_base.log` / `qwen_domain.log`。最终 exit 0 表示全范围完成，2 表示有失败/阻塞；不能把终止当成成功。完成后有独立 HTTP 进程验证，验证后停止，不宣称持续服务正在运行。

原发行版本和配置保持不变时恢复（只补未完成任务）：

```bash
bash "$RELEASE/scripts/resume_clock_wave.sh"
```

再次跑安装器会创建新 run；若 latest 已指向修订轮，安装器会按 `config.json → revision_parent.path` 找回原 CW1。要补原修订 run 时使用 resume，而非再次安装。

打包（包含私有开发预测、prompt 与数值参数，不含原始 counts 或 Qwen 大权重）：

```bash
"$VDC_PYTHON" "$RELEASE/scripts/pack_clock_wave_report.py" --run "$RUN"
```

查看结果浏览器（仅本机，保存结果模式，不执行任意新样本推理）：

```bash
screen -dmS vdc_cw_results "$VDC_PYTHON" "$RELEASE/scripts/serve_clock_wave.py" --run "$RUN" --port 8769
```

按既有 SSH 端口转发访问 `127.0.0.1:8769`，不开放公网。身份确认失败时可看独立数值建议，但不会把数值建议计作 LLM 成功。界面使用实际每身份参考范围着色，不拿统一 0–1 判断支持。
