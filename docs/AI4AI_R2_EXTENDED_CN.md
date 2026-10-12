# R2 扩展运行与历史结果回读

本版默认最多 1000 次研究 API 请求，累计 token 上限为 `null`（无限制）。
失败尝试计入次数；供应商余额、速率与上下文限制仍存在。单次保守上下文预算
60000、输出上限16000、网络读取超时900秒。真实返回用量和未知用量分别登记，
不把字节估计当成真实 token 或费用。模型可以在完成研究范围后结束，不凑实验数。

本版修复长期迭代中的结果可访问性：`list_actions`、`read_action`、
`inspect_preparation` 可以回读已完成准备与结构检查。重复动作 ID 返回已保存结果，
不会静默消失；改变动作内容必须使用新 ID。上下文保留实际任务/实验/导出计数，
准备记录和已取回文献可追溯。检索次数不计为模型训练。

## 从停止的准备轮次接续

使用全新的代码 release；无需重装依赖或重新下载数据。只适用于尚未登记训练任务、
尚未训练/导出模型，且状态为 `budget_exhausted_partial` 或 `paused_by_operator`
的旧 run。脚本持有旧 runner 同款锁，复制并校验准备工件、文献、API 台账、决策和
来源清单；旧 run 保留，新目录记录父版本及所有继承文件指纹。不得从删减过的报告 ZIP
恢复，此操作需要服务器原 run 的准备输出。

在新 release 根目录执行一次：

```bash
export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
export VDC_PYTHON=/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python
export VDC_R2_RUNTIME_ROOT="$VDC_PRIVATE_ROOT/r2_runtime_devices_v2"
bash scripts/start_research_r2_extended.sh /absolute/path/to/stopped_parent_run
```

脚本明确使用已通过隔离检查的 CPU 环境；不声称解决 AutoDL GPU 隔离问题。
API key 已在环境中则复用，否则隐藏输入。客户端更新后执行一次新的小额 API 检查，
然后复制准备记录、冻结新 release、启动 screen。继承的 API 请求占总1000次容量；
例如已使用100次，则研究队列最多再提交900次。检查请求单独记账。
`ready_not_started` 是迁移状态；只有 `SCREEN=...` 才表示已提交后台任务，
后续以实际日志及 status.json 判断是否在运行。

## 看进度、停止、恢复

```bash
ROOT=/root/autodl-tmp/vdc-private
RUN="$(cat "$ROOT/LATEST_RESEARCH_R2.txt")"
cat "$RUN/status.json"
tail -n 60 -F "$RUN/console.log"
```

`API_START` 到 `API_COMPLETED` 之间表示等待供应商回复。`START`/`COMPLETED`
表示工具动作；`REUSED_RESULT` 表示重读已有结果。Ctrl+C 仅退出 tail。
`status.json` 中 continuation、新增动作/决策、实验与导出、用量分开登记。

需在当前决策完成后停下：

```bash
touch "$RUN/STOP"
```

STOP 不会中断正在执行的模型或 API 调用。停止后保留目录，移除 STOP 并显式恢复：

```bash
rm -- "$RUN/STOP"
export VDC_R2_RUN="$RUN" VDC_R2_RESUME=1 VDC_R2_DEVICE_POLICY=cpu
export VDC_RESEARCH_CONFIG="$RUN/release/configs/research_r2_selected.json"
export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
export VDC_PYTHON=/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python
export VDC_R2_RUNTIME_ROOT="$VDC_PRIVATE_ROOT/r2_runtime_devices_v2"
export VDC_RESEARCH_API_BASE=https://api.deepseek.com
export VDC_RESEARCH_MODEL=deepseek-v4-pro
export VDC_RESEARCH_API_CHECK="$RUN/api_check_receipt.json"
if [ -z "${VDC_RESEARCH_API_KEY:-}" ]; then
  read -rsp 'DeepSeek API key: ' VDC_RESEARCH_API_KEY; echo
fi
export VDC_RESEARCH_API_KEY
bash "$RUN/release/scripts/launch_research_r2.sh"
```

网络超时等未知送达错误仍保留台账并停止，不自动无限重发。查明后可显式设置
`VDC_R2_RETRY_API_ERRORS=1` 恢复。实验代码失败继续处理独立动作和后续决策。
本版只是控制器与启动流程更新；不宣称已在用户服务器完成新增训练。
