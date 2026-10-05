# CW1：先确认真实 run，再导出或补跑

报告 ZIP 的时间是打包时间，不是实验时间。`tail` 使用的 `RUN`/`JOB_DIR` 是当前 shell 变量；在子 shell 启动新 run，并不会修改父 shell 里旧的变量。因此即使 LATEST 已更新，旧的 `--run "$RUN"` 仍可能重新打包旧目录。这是需要核查的一种可能，不能仅凭报告包认定服务器实际发生了它。

本次不修改模型规则、不重新运行 PK2。先检查配置、计划、任务文件校验、原始短协议回答、日志与源码版本；不是按目录名包含 revision 或状态写 completed 就判定完成。没有 SSH 会话时，本地不能确认服务器上是否有其他 run。

## 推荐：一次完成扫描与条件处理

以下代码先 fetch 并创建干净代码目录，不覆盖原运行目录，也不使用遗留的 `RUN` 变量：

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/VirtualDiapauseCell
  export VDC_SOURCE_REPO="$PWD"
  export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
  export VDC_PYTHON="$PWD/.venv-autodl-vdc/bin/python"
  git -c http.version=HTTP/1.1 fetch origin main
  REV="$(git rev-parse origin/main)"
  RELEASE="/root/autodl-tmp/vdc-releases/cw_audit_${REV:0:12}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
  mkdir -p "$RELEASE"
  git archive "$REV" | tar -x -C "$RELEASE"
  printf '%s\n' "$REV" > "$RELEASE/RELEASE_COMMIT.txt"
  bash "$RELEASE/scripts/reconcile_clock_wave.sh" --start-missing
)
```

如只审计、不允许提交任何任务，删去 `--start-missing`。离线代码包解压到新目录后，直接执行其中的 `scripts/reconcile_clock_wave.sh`，使用相同环境变量即可。

扫描默认覆盖私有根目录的 `runs`、原代码仓库的 `runs` 和 LATEST 所指目录；不是宣称搜索了整台服务器。若曾使用其他目录，先单独审计并增加 `--extra-run-root /实际目录`，不要在未扫描的情况下断言不存在新 run。

处理结果：

| next_action | 实际动作 |
|---|---|
| export_completed_revision | 直接导出已有、校验通过的修订结果，不启动训练 |
| export_partial_revision_do_not_restart | 导出真实已完成部分和未完成清单，不重复提交 |
| inspect_existing_revision_do_not_duplicate | 有修订配置但尚无可确认的新完成输出，先查看日志 |
| inspect_uninitialized_or_unreadable_run | 存在未初始化/无法审计目录，避免启动重叠任务 |
| targeted_revision_not_found | 仅当指定 `--start-missing`，从校验通过的原 CW1 新建针对性补跑 |

若选择补跑，仍是原 CW1 runner：51 项校验复用、10 项新增执行。只执行短身份/短证据编号的 Base/Domain 推理及依赖链、三视图 residual readout、Early 等条件支持范围与锚点诊断。此前约定的无数值提示/证据挑战属于身份推理任务；不训练新 LoRA、不重跑 PK2、不扩大生物接受范围、不从旧半截 JSON 抽取答案。

已有修订任务未完成时，应使用其**原始发行代码和配置**恢复。审计报告会尽可能用完整源码指纹匹配实际发行目录；不可用最新代码覆盖旧 run 后强行沿用 completed。需要修改 prompt、协议或特征时建立新 run。新版任务记录绑定配置/代码/任务签名，签名不匹配的 completed 明确拒绝复用。

## 审计与打包的新证据

终端会显示审计 JSON 路径、选中的真实 run、LATEST 是否一致及已导出的 ZIP 路径。请下载该**新输出路径**，不要继续使用旧的 shell `RUN` 值。每个报告新增 `PACKAGING_PROVENANCE.json`：

- 显式打包目录、配置 SHA256、完整协议、源码指纹、任务计划与日志指纹。
- 新版启动 receipt 和运行时源码 manifest；历史 run 缺少时明确写缺失，不补造历史证据。
- 打包当时的 LATEST 指针及是否与显式目录一致。
- 新增、重跑、核验复用、未完成，以及相对原父 run 新增/变化/不变的任务文件清单。
- 稳定 payload 指纹；若发现相同内容的已有报告，明确 `operation: repack_only`。未发现相同包也只称“导出既有结果”，打包永远不是一次训练。

打包器仍不包含原始 counts、Qwen 大权重或完整私有配置；包含的协议和来源摘要、私有预测须留在私有环境。

明确指定修订 run 时可加防误用检查：

```bash
PYTHONPATH="$RELEASE/src" "$VDC_PYTHON" "$RELEASE/scripts/pack_clock_wave_report.py" \
  --run /实际审计选中的修订目录 --require-revision
```

旧 run 会被拒绝，而不会再次生成一个被误解为修复轮的新文件名。包名也加入真实 run ID。

补跑启动后的进度：重新读取指针，再运行对应发行目录的 status，或查看新目录 console.log。`TASK_EXIT_CODE=0` 只表示该 run 当前范围完成；审计还会验证它是否确实属于修订协议。
