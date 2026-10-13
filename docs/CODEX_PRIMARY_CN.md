# Codex 主控、DeepSeek 审议、AutoDL 计算

已跑通pilot后，使用[持续研究与途中检查点](CODEX_CONTINUOUS_CN.md)从原工作区续跑；无需重复创建worker。

本轮保留现有研究任务和隔离执行器。Codex CLI 在本地电脑运行，利用可恢复会话、文件与代码工具推进研究；DeepSeek 在需要时审议计划或实验结果，不派发工具、不覆盖主控决策。AutoDL 只接受编号任务并运行原来的数据准备、契约检查、候选训练、评价和导出。

这不是把 OpenAI 请求代理到国内服务器，也不是给两个模型同时写同一份代码。Codex 登录留在本地。原始矩阵、候选训练和模型文件留在 AutoDL；本地取得工具允许的代码、元数据、指标与证据摘要。私有工作目录、连接信息、研究内容和密钥不得提交公共 GitHub。

## 本轮修复

- 同一决策的多项工具结果一起保留；超长记录明确截断并提供 `read_action` 引用，不再只剩最后一项。
- 最新代码、原始样本元数据和任务契约作为持续证据保存；完整投影在 `working_evidence.json`，本地每条结果另存 `evidence/`。
- `working_state` 保存问题、样本关系、处理方式、文件引用、障碍和下一步；这是主控的解释，不能替代宿主校验。
- 区分路径写错、目录误当文件、真正缺文件与哈希不符。
- 任务请求写盘后才提交；断线重新查询/提交同一编号与相同内容，不重新问模型、不重复训练。执行中进程崩溃的未知结果保留人工检查状态，不自动重跑。
- 8 次无进展诊断、24 次无进展停止继续生效。读取次数不等于实验次数。

研究对象仍是 universal diapause/dormancy model；算法与科学概念自由选择。数据关系、目标隔离和真实评价沿用已有契约。全数据适应性研究不会重新获得独立盲测资格。

## 1. AutoDL：创建独立工作 run

先将这一版代码拉取为独立 release，或解压本轮离线包。不要修改或继续覆盖已停止的旧 run。`RELEASE` 指向包含 `src/` 和 `scripts/` 的新目录。

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/VirtualDiapauseCell
  git -c http.version=HTTP/1.1 fetch origin main
  REV="$(git rev-parse origin/main)"
  RELEASE="/root/autodl-tmp/vdc-releases/codex_${REV:0:12}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
  mkdir -p "$RELEASE"
  git archive "$REV" | tar -x -C "$RELEASE"
  printf '%s\n' "$REV" > "$RELEASE/RELEASE_COMMIT.txt"
  export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
  export VDC_PYTHON=/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python
  export VDC_R2_RUNTIME_ROOT="$VDC_PRIVATE_ROOT/r2_runtime_devices_v2"
  export VDC_R2_DEVICE_POLICY=cpu
  bash "$RELEASE/scripts/launch_codex_worker.sh" \
    "$VDC_PRIVATE_ROOT/runs/research_engine_20261012T210922Z_54225"
)
```

记录输出的 `RUN=`。启动器会核验、继承已完成准备材料，冻结自己的 release，并用 `screen` 保持计算工作进程。**它在等待本地 Codex，不会自行调用 DeepSeek 开始研究。** 当前使用已验证的 CPU 隔离；GPU 只有在原有 GPU 隔离实际通过检查后才能改用 `ai`，没有无隔离回退。

离线方式：用 `.venv-autodl-vdc/bin/python -m zipfile -e 包路径 新RELEASE目录` 解压，然后设置同样的环境变量，运行其中的 `launch_codex_worker.sh`。

## 2. 本地 Windows：一次性准备 SSH

使用 Windows 终端建立普通 SSH 登录，并核对 AutoDL 控制台给出的主机地址。之后配置 SSH 公钥，使 `ssh -o BatchMode=yes ...` 能成功登录；不要把密码放在脚本或聊天里。建议在用户自己的 SSH 配置中使用 `autodl-vdc` 别名。

```text
Host autodl-vdc
    HostName <AutoDL控制台主机>
    Port <AutoDL控制台端口>
    User root
    IdentityFile ~/.ssh/vdc_autodl
```

没有现成密钥时，在 Windows PowerShell 中生成并上传**公钥**（私钥不上传；命令中的服务器替换为实际地址）：

```powershell
ssh-keygen -t ed25519 -f "$env:USERPROFILE/.ssh/vdc_autodl"
Get-Content "$env:USERPROFILE/.ssh/vdc_autodl.pub" | ssh -p <端口> root@<主机> 'umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys'
ssh -o BatchMode=yes autodl-vdc 'printf connected'
```

带口令的密钥需先加入本地 ssh-agent；自动控制器不会弹出密码输入。不要设置 `StrictHostKeyChecking=no`。网络中断时工作进程可继续执行已经收到的任务。

## 3. 本地启动主控

官方 Codex CLI 支持非交互执行和明确会话 ID 的恢复。本项目使用本地已有配置，不写死聊天界面的内部模型别名。先运行 `codex login status`；需要登录时使用正常的 `codex login`。ChatGPT 登录与 API 计费是不同渠道，以自己实际登录方式为准。

在本地项目 PowerShell 中：

```powershell
# 如果已有项目 .venv，则直接复用。一个新的本地主控环境只需要 numpy。
# python -m venv .venv
# .venv/Scripts/python -m pip install numpy
codex login status

# 用刚刚 AutoDL 输出的 RUN 替换；这不是旧父 run。
./scripts/start_codex_controller.ps1 `
  -RemoteRun '/root/autodl-tmp/vdc-private/runs/research_codex_实际时间_编号' `
  -Workspace './work/codex_controller_pilot' -Turns 24
```

初轮 24 次是验证工作衔接的调用预算，不是 24 个实验。用相同工作目录再次运行会恢复明确保存的 Codex 会话和待执行编号。需要扩大时显式提高 `-Turns`，无进展停止仍然生效。启动脚本默认前台运行，电脑休眠/进程退出后不会继续派发新任务；服务器已收到的任务继续运行。不要启动两个控制器写同一工作目录。

可以直接传 `-SshTarget user@host -Port 端口 -IdentityFile 本地私钥路径`；这些仅写入已忽略的 `connection.local.json`。不需要把服务器地址写进 GitHub。

## 4. DeepSeek 辅助

当前配置沿用已成功验证过的 DeepSeek 接口和模型。主控在计划评审或结果解释节点通过 `advisor_question` 请求意见。辅助意见写入 `latest_advice.json` 和独立 `advisor/` 日志；其中即使出现 `actions` 也不会执行。

在**本地** PowerShell 设置本地辅助调用环境；不从服务器读取已有密钥：

```powershell
$env:VDC_RESEARCH_API_BASE = 'https://api.deepseek.com'
$env:VDC_RESEARCH_MODEL = 'deepseek-v4-pro'
$secret = Read-Host 'DeepSeek API key（仅本地进程）' -AsSecureString
$env:VDC_RESEARCH_API_KEY = [System.Net.NetworkCredential]::new('', $secret).Password
# 然后运行上面的 start_codex_controller.ps1
```

辅助缺密钥或失败会明确记录，不冒称双模型已参与，也不阻止 Codex 推进数值任务。已失败的辅助调用不会自动重复付费。辅助调用记录、Codex 调用记录和实际实验计数分开保存。

## 5. 进度、暂停、恢复与报告

AutoDL 看真实计算：

```bash
ROOT=/root/autodl-tmp/vdc-private
RUN="$(cat "$ROOT/LATEST_CODEX_WORKER.txt")"
cat "$RUN/worker_status.json"
tail -n 60 -F "$RUN/console.log"
# 导出服务端报告（历史与新结果仍分开）
/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python \
  "$RUN/release/scripts/pack_research_r2.py" --run "$RUN"
```

本地看 `work/codex_controller_pilot/controller_state.json`、`turns/*/codex_receipt.json`、`turns/*/response.json`、`agent/research_notes.md`（主控写入时）和 `latest_advice.json`。Codex 的可写研究区为 `agent/`，宿主管理的任务队列、原始回执和连接配置保留在它的父目录。命令行会显示 `CODEX_START → WAIT_WORKER → COMPLETED`，等待时每分钟提示一次。`WAIT_WORKER` 时检查远端 heartbeat 和 console；`stalled_no_progress` 是停止，不能当作完成。

暂停本地派发：在本地工作目录创建 `STOP` 文件。移除后重新运行原命令；已经提交的请求按相同编号恢复查询。停止远端则在远端 run 下创建 `STOP`，当前动作完成后停止。未知执行结果不得删除 `.running.json` 后盲目重试。

如要恢复一个正常暂停的远端，确认没有未知执行中的请求，移除 `STOP` 并在 `screen` 中使用 `"$VDC_PYTHON" "$RUN/release/scripts/research_worker.py" resume --run "$RUN"`；无进展停止需要先检查具体障碍，不能通过无限自动恢复绕过。

本轮实现的验收分两层：软件与真实公共数据工程检查；**真实 Codex 主控从继承的准备开始，登记任务、拟合候选、比较、修订和导出**。后者必须以新的实际调用及工件为证，预写工程自检不能替代。未接通 SSH、未启动主控前不得写成“研究已恢复”。

官方接口依据：[Codex 非交互运行](https://developers.openai.com/codex/noninteractive)、[Codex SDK](https://developers.openai.com/codex/sdk)。当前采用 CLI，暂不引入额外的多主控框架。
