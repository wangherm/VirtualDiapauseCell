# 人工阶段审查之后续跑

`awaiting_stage_review` / `completed_declared_scope` 是审查暂停。只有操作者批准下一阶段后才恢复。保存私有研究任务书，继续使用原本地工作区及服务器run，不改旧完成记录，不新建研究平台。

先预检（不解除暂停、不调用研究模型）：

```powershell
$workspace = 'C:/path/to/existing/controller_workspace'
$brief = 'C:/path/to/private/next_stage_brief.md'
$next = 151 # 替换为实际本地与远端一致的下一轮
& ./.venv/Scripts/python.exe scripts/advance_codex_stage.py `
  --workspace $workspace --brief $brief --expected-next $next
```

操作者确认本阶段继续后，使用相同参数加 `--apply`。脚本通过现有SSH连接上传小型恢复工具与任务书，使用原run冻结的引擎读取数据；不修改冻结源码、配置、额度或旧工件。它检查下一轮一致、无未完成邮箱请求、无STOP，备份前一阶段状态，保存任务书指纹，然后启动原worker等待主控。任何未知执行结果仍需检查。

预检返回 `validated_not_applied`。应用后应返回 `completed` 和 `worker_ready: true`；这只代表执行器可以接收决策，不代表新实验已开始。重复应用同一尚未开始的阶段不会重置历史；进入后续轮数后不能用旧编号再次应用。

随后按原来的命令运行本地主控，加 `--brief $brief --accept-controller-update --continuous`，继续保留检查点、无进展保护与DeepSeek辅助审议降级。不要启动两个主控。

任务书应明确研究问题、现有正反证据、预期输入输出及阶段验收，但不必指定模型菜单。阶段性失败可以作为结果交付；缺标签、缺支持范围不能靠生成数字补齐。保持旧模型可回放，新版本单独导出。已查看材料不能重新称作独立盲测。

通用代码可公开；包含私有样本、未公开结果的任务书和运行回执保存在私有工作区，不提交仓库。
