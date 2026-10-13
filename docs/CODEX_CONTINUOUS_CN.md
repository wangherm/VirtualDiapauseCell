# 持续研究与途中检查点

沿用 Codex 主控、DeepSeek 按需审议、AutoDL 隔离执行器。长期目标是稳定、可使用的 diapause/dormancy model；科学路线、模型及表示由主控选择。本轮只更新本地主控，不新建服务器 run、不重跑历史任务。

## 检查点和停止条件

使用 `--continuous` 或 PowerShell `-Continuous` 后，24轮成为调度批次而不是研究终点。每新增5项成功实验（包括负结果），或距离上次检查点24轮时，宿主保存 `checkpoints/*.json`、Markdown和当时的研究笔记，再继续。重复使用同一动作不会增加计数。检查点区分累计历史和新增动作，并保存指标、评价单位、失败、重载、工件引用、用量及既有预算。JSON是执行证据，模型笔记是解释。

主控下一轮会读取最新检查点，并更新正反证据、未决依赖和下一项有区分力的实验。研究方向是样本状态、恢复/干预以及跨背景适用性，不以任务标签齐全为完成。起始表达的增量需要合法的起始—终点关系；无配对时不能虚构轨迹。所有已用材料是自适应/回顾性研究，不能重新称为盲测。

- 服务器原有无进展保护继续有效，8轮预警、24轮无进展停止。
- 模型声明完成后本地主控标记 `awaiting_stage_review` 并停止，等待人工审查真实输入、预处理、导出、重载和跨背景结果；不自动宣布科学成功。
- Codex失败、额度/认证错误不自动重试；DeepSeek请求失败、缺密钥或累计预算耗尽，在保存当前已完成动作后暂停。
- DeepSeek调用按整个本地工作区累计，不因每轮目录变化而重置费用上限。延续已有配置，不自动购买资源或提高预算。
- 本地工作区创建 `STOP` 文件可停止派发；已经在服务器执行的任务不强行杀死。恢复先移除STOP；未知结果仍需检查。

## 从旧工作区续跑

必须使用原工作区的**绝对路径**。旧版PowerShell相对路径可能落在用户主目录的 `work` 中，不能以仓库路径猜测。先检查其中 `controller_state.json` 和 `connection.local.json`。

本地主控代码改变时，需要一次显式 `--accept-controller-update`。它只允许在无待定任务、下一轮未发起、远端编号一致时迁移；保留旧会话和全部回执，在 `controller_versions/` 写入前后代码指纹。不能把旧completed当作新prompt的结果。可以用 `--brief` 加入操作者说明，按下一轮生效、记录指纹。

```powershell
# 在原窗口保留已有DeepSeek环境变量。填入实际原工作区绝对路径。
$workspace = 'C:/path/to/existing/controller_workspace'
$python = './.venv/Scripts/python.exe'
& $python scripts/run_codex_research.py run `
  --connection "$workspace/connection.local.json" `
  --workspace $workspace --continuous --turns 24 `
  --checkpoint-experiments 5 --checkpoint-turns 24 `
  --accept-controller-update --brief 'C:/path/to/private/continuation_brief.md'
```

也可使用 `start_codex_controller.ps1 -Continuous -AcceptControllerUpdate -Workspace 绝对路径 -Brief 说明文件`。该启动器支持发现桌面应用安装的Codex CLI，并检查当前登录；不修改登录方式。订阅登录与API登录由实际本机认证决定。

查看 `latest_checkpoint.json`、`controller_state.json` 和当前 `turns/编号/events.jsonl`。主控终端出现 `CHECKPOINT` 表示已落盘，`BATCH_SAVED` 表示批次结束且即将继续，不代表达到科学目标。

本机需要保持唤醒、联网及主控终端打开。不要同时启动两个主控，不重新启动旧DeepSeek主控。原始矩阵留在服务器，本地checkpoint、会话和报告属于私有研究记录。
