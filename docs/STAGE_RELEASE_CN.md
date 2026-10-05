# 阶段封版：唯一推荐运行入口

目标：完成 fixed-evidence Qwen 评价、一个固定收缩诊断、默认路线真实调用、统一结果表和冻结快照，然后停止扩展。使用当前真实 CW1 **v2 修订 run** 为父来源；不是从 PK2 重新开始。

固定队列 34 项：24 项核验复用、4 项身份/依赖链重跑、3 项收缩诊断、3 项默认路线调用。已完成支持范围诊断不重做。旧压力、语义、family/逐 pool 数值网格不进入默认队列，其历史结果留在父 run。

## 1. 更新代码（不覆盖旧发行目录）

GitHub 可访问：

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/VirtualDiapauseCell
  git -c http.version=HTTP/1.1 fetch origin main
  REV="$(git rev-parse origin/main)"
  RELEASE="/root/autodl-tmp/vdc-releases/cw_stage_${REV:0:12}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
  mkdir -p "$RELEASE"
  git archive "$REV" | tar -x -C "$RELEASE"
  printf '%s\n' "$REV" > "$RELEASE/RELEASE_COMMIT.txt"
  bash "$RELEASE/scripts/vdc_stage.sh" start
)
```

默认使用现有 `/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python`，不重新装 CUDA/PyTorch。来源先读 `LATEST_CLOCK_WAVE_REVISION.txt`，若不存在再读 `LATEST_CLOCK_WAVE.txt`；必须核验为已有 v2 修订配置。

如果指针不正确，先显式设置 `VDC_STAGE_SOURCE=/实际的CW1修订run`，不要使用旧 shell `RUN` 猜测。启动时保留原目录并新建阶段 run。可以用前一轮审计脚本核对来源。

GitHub 不通时，上传本次**离线代码包**并解压至新的发行目录，直接执行其中的 `bash scripts/vdc_stage.sh start`。无需上传新的数据准备包。可设置 `VDC_PRIVATE_ROOT`、`VDC_PYTHON` 改变已知环境路径。

## 2. 进度与恢复

启动通过 screen 后台运行，会写 `LATEST_STAGE.txt` 和 `LATEST_STAGE_RELEASE.txt`，原父 run 保持不变。打印的 RUN 文字不自动更新父 shell 变量。

存在阶段指针时再次 `start` 会停止并提示查看/恢复原 run，避免重复提交。不要为绕过提示删除已有指针。

```bash
PRIVATE=/root/autodl-tmp/vdc-private
RELEASE="$(cat "$PRIVATE/LATEST_STAGE_RELEASE.txt")"
RUN="$(cat "$PRIVATE/LATEST_STAGE.txt")"
bash "$RELEASE/scripts/vdc_stage.sh" status
tail -n 50 -F "$RUN/console.log"
```

`Ctrl+C` 仅退出 tail。基础依赖/模型加载错误仍会失败；格式或证据错误记录为错误，并完成三类预定推理，不再停在三条 schema 预检，也不把这些错误算作生物学 unknown。

中断恢复：`bash "$RELEASE/scripts/vdc_stage.sh" resume`。使用该 run 的原始发行目录，不覆盖 prompt/config 后强行沿用 completed。变更任务签名必须建立新 run，历史结果不被改写。

## 3. 封版、打包与浏览

全部任务执行后，生成 `STAGE_SUMMARY.json`、`MODEL_CARD_CN.md`、`stage_snapshot.json`、保存预测与原始回答。范围内外、Early/Late/Exit 等条件分开报告。快照为 `frozen_experimental` 的条件是固定队列全部执行；不要求模型超过基线。任何真实依赖未完成则是 `partial_not_closed`。

随后进行独立 HTTP 进程验证并停止该验证服务，自动打包私有报告。它不是常驻服务。最终 `TASK_EXIT_CODE=0` 表示执行、浏览验证与打包链成功；非零需看日志，不能仅凭快照状态认为所有部署检查都已完成。

需要再次打包或持续查看：

```bash
bash "$RELEASE/scripts/vdc_stage.sh" pack
screen -dmS vdc_stage_results bash "$RELEASE/scripts/vdc_stage.sh" browse
```

通过既有 SSH 转发访问本机 8769 端口。默认展示三视图主路线和阶段总表，“辅助/历史任务”可查看本 run 的其他结果；更早探索仍在原 run。明确是**保存结果浏览**，没有开放任意新数据上传推理。

报告包内 `PACKAGING_PROVENANCE.json` 指明真实 run、源码/配置指纹、复用与新执行分类、相对父 run 的变化和未进入默认队列的历史任务。下载脚本最后打印的 ZIP，不使用前一轮旧路径。

## 4. 收尾边界

固定默认：bulk 原残差候选（alpha=1）＋并列 Ridge；core 简单 clock＋身份；粗身份关闭 residual 修正。alpha=100 仅作为唯一新增诊断，不自动改默认。Qwen 使用固定 `e1/e2/e3` 字段重新生成，不能截断旧 evidence_slots 列表冒充修复。

无新 LoRA、无新公共数据、无 PK2 重跑、无 reserved 查询；不扩支持范围，不造 depth，不把 source annotation 一致率当 lineage gold。阶段报告记录完这些结果与限制后，本轮结束；下一阶段需另行定义问题。
