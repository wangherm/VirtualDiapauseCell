# PK2 定向修复与补跑

本轮沿用原队列，不重新训练有效的 Qwen，也不改变数据角色。修复后的服务器任务由用户启动。公开仓库只提供代码与协议；私有报告、样本、访问链接和模型不提交。

## 修复范围

|问题|本轮动作|
|---|---|
|清单使用 `new_domain/shuffled_new_domain`，旧执行器只识别 `domain/shuffle`|重跑 24 条领域/打乱语义数值任务；未知模式直接报错|
|缓存生成但没有实际进入模型|验证所选 adapter 文件哈希、向量哈希、feature 顺序、非零值、打乱排列；训练前真实输入前向检查，保存重载后再次与零语义比较|
|逐 pool 的外层观测选择了 best checkpoint|18 条任务改用预定固定 3000 步；训练中不计算外层评价，不产生中间 checkpoint 外层成绩；只评价最后一步|
|单 profile 的动态标准差比例写成 0|不足两条有效观测或目标无变化时写 null；汇总从保存的预测重算，不改历史目录|
|旧检索/无证据题全部为 unknown|保留原成绩，新评估包括按来源定位的支持证据和成对移除证据；保存完整输入、文档 ID、索引指纹、token 数及 always-unknown 对照|
|报告缺少语义与 HTTP 核查材料|补充缓存 metadata/向量、语义输入契约、公开弱参考语料、HTTP 响应与进程日志|

新知识评价是受控的证据使用诊断，不是开放搜索或闭卷知识测试。推理索引明确包括开发验证文献的证据，但不包括 completion；训练划分不变。Adapter 的选择已用过开发题，因此重评不能宣称新的独立测试。

外层 pool 也已经被查看过。本次排除 checkpoint 选择污染，但仍是已有 development 的协议修正，不是全新盲测。后续预留个体、后期 Exit 等仍等候候选与输入规则冻结。

## 执行方式

只对已审核的首轮 PK2 代码指纹启用兼容导入；其他版本报错。复用之前逐文件检查源工件，复制到新目录后再次核验。科学配置、名单、PK1/抓取 manifest 和数值依赖身份必须一致。已复制的旧权重明确标记 `reused_verified`，不计作本轮新训练。

新跑 42 条数值任务，另有基础模型与已选 adapter 的两个知识重评任务、汇总和服务检查。原 25% 子集没有训练锚点的两条学习曲线记为 `inapplicable`，不借用验证锚点、不重抽子集追求完整计数。完整 171 槽仍在总表中。

同一个 runner 的 normal resume 仍要求代码、配置、输入、环境签名完全一致。`--repair-from` 是显式、限定版本的迁移，不绕过 resume 校验。源目录不删除，不覆盖；需要额外磁盘空间复制有效工件，启动时报告所需空间。

从原 Git checkout 执行；把原训练目录填入变量，不要填报告 ZIP。首次执行自动创建新修复目录，不要将 `VDC_PK2_RUN` 设置成旧训练目录。

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/VirtualDiapauseCell
  git -c http.version=HTTP/1.1 fetch origin main
  export VDC_PK2_REPAIR_FROM=/path/to/completed/pk2_training_run
  unset VDC_PK2_RUN VDC_RESUME
  git show origin/main:scripts/install_pk2_repair.sh | bash
)
```

这会从原 run 配置读取现有缓存和输入路径，经 `git archive` 建立独立代码 release，以原 CUDA Python 执行，使用 `screen` 后台运行。不需要重新上传数据，不覆盖旧 checkout，不重新 `pip install -e`。

## 查看进度

```bash
PRIVATE=/root/autodl-tmp/vdc-private
JOB_DIR="$(cat "$PRIVATE/LATEST_PK2_TRAINING.txt")"
RELEASE="$(cat "$PRIVATE/LATEST_PK2_RELEASE.txt")"
PY=/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python
"$PY" "$RELEASE/scripts/pk2_training_status.py" --run "$JOB_DIR"
tail -n 50 -F "$JOB_DIR/console.log"
```

初期 `VERIFIED REUSE` 是核验/复制，不是重新训练。`START` 表示新任务开始；`EVALUATED_NEW` 表示训练与评价完成；`reused_verified` 是原结果复用。`Ctrl+C` 只停止 tail。退出码 0 表示适用任务执行结束，并不表示全部模型超过基线；任何失败保留独立原因。两条明确不适用的学习曲线不作为程序错误。

修复快照默认在 `127.0.0.1:8767`，旧 8765/8766 服务不关闭、不覆盖。该端口被占用会报错，可在首次启动前设置 `VDC_PK2_SERVICE_PORT`。使用新快照查看修正后的语义比较，旧服务的领域/打乱条目属于已知无效的历史条件。最终确认：

```bash
cat "$JOB_DIR/exit_code.txt"
cat "$JOB_DIR/tasks/register_pk2/result.json"
curl --fail http://127.0.0.1:8767/health
```

## 中断恢复

复用同一修复 release 和修复 run；不要再次安装更新代码后冒充同配置 resume。

```bash
export VDC_PK2_RUN="$JOB_DIR" VDC_RESUME=1
export VDC_PYTHON="$PY"
# 从修复 config.json 读取的原始输入路径也必须保持一致。
# 推荐使用下列恢复入口，由它读取已保存配置：
bash "$RELEASE/scripts/resume_pk2_training.sh" "$JOB_DIR"
```

## 总表与打包

`tasks/repair_summary/COMPARISON_CN.md` 汇总所有数值结果，保留 Ridge、幅度和实际知识评价；`REPORT_CN.md` 是全队列执行表。类型条件 waves 及端点、转移、功能基线保留；不默认将最复杂模型宣布为赢家。

```bash
"$PY" "$RELEASE/scripts/pack_pk2_training_report.py" --run "$JOB_DIR"
```

生成的 `VDC_PK2_TRAINING_PRIVATE_report_*.zip` 含私有开发预测与身份，只私下传回，不提交 GitHub。模型权重、原始表达与下载链接不打包。修复结果未实际返回之前，不声称 Qwen 数值增益成立，也不自动调用预留样本。
