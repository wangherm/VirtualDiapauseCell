# PK2 修复后的冻结应用轮

本轮复用修复后的 PK2 权重。先固定协议、共同 seed 42、模型身份和开发集读出，再查询授权的预留及探索样本。称为设计知情的回顾性应用，不能把 thesis 中已经研究过的材料称为新盲测。原始角色名单不变。

## 这一轮实际要跑什么

| 分支 | 执行内容 |
| --- | --- |
| 冻结 | 原训练 counts 重现原 programme 输入；核验来源、语义前向检查、模型与角色哈希；保存独立快照 |
| Bulk | R0 零语义、R0 领域语义、R3 原始 Qwen 语义三个候选 |
| 整 pool | R0、R2、R5 三个候选及 Ridge；直接参考 clock 与神经 clock 分别显示 |
| Pool×type | R0 零语义、R0 领域语义研究对照；类型条件 wave 仅在原标签和观测范围支持时计算 |
| 隐藏基因 | 8 个 latent→gene Ridge、3 个直接 programme→gene Ridge；仅原 train 拟合，不选超参数。固定 hash 分区从全部重叠 programme、排名与输入库大小分母移除；最多 256 个目标由 train 方差选择 |
| 热应激替代 | 原始 counts 待用户补齐。已有归一化 raw.X 在 pool 内取均值，再算 rank programme；用同一历史处理中的原 train 对照做描述性差异，不进入 counts 模型，不输出 clock/depth |
| 知识 | 复用已选 adapter，与 base 各做一次修正评价。真正撤证据时同时移除 Source/Context；按事实去重，旧开发家族、新增三个评价专用论文家族、共享撤证据题分别报告 |
| 结果界面 | localhost 上浏览本轮保存的真实查询结果；不接收任意上传，不替换既有 PK1/PK2 服务 |

具体候选与所有规则见 [配置](../configs/application_v1.json)。没有新增 state/Qwen 长训练或重新搜索 171 条路线。知识新增家族仅有 9 条摘要级弱参考，不是全面专家评价；base 预训练是否接触这些公开论文未知。

隐藏读出的目标是全库相对表达，保留组成依赖，不能解释为每细胞绝对 RNA。冻结 state 训练时使用过开发样本的完整 programme；本轮验证的是去除目标后的冻结表示加新读出能力，存在输入分布变化。直接 programme Ridge 和训练均值同时保存。当前表达定位的 gene/TF/wave 残差是描述性比较，不能替代隐藏读出检验。

未来预测尚无匹配开发数据拟合的 killifish 转移器：后期表达只能用于定位，不能称作从 Day1 预测未来。stalled 没有真实恢复结局时不报失败分类准确率；没有匹配功能测量时不造 depth。历史 Exit X 为整数 count-like，UMI 来源记录仍不完整；结果标为适用性测试。跨材料采用显式投影并保存原材料，保留原 StatePredictor 的严格 scope 校验。历史细胞类型不自动重命名为训练标签，未知类型的条件 wave 返回不支持。

## 服务器启动

继续使用现有 CUDA 环境。代码先 fetch，再从远端版本生成独立 release，不在有离线上传修改的旧 checkout 上强行 pull/reset，不重新安装 CUDA Torch。

1. 将小型私有补充包上传服务器，解压到私有根目录下的 `application_sc`，确认直接包含 `manifest.json`、`core.npz`、`core_celltypes.npz`、`stress_train.npz`、`stress_queries.npz`。不要把补充包放进 Git。
2. 将 `VDC_APPLICATION_SOURCE` 设为**已完成的修复轮目录**，其中必须有 `repair_import.json`。不能指向最初含语义错误的训练目录。
3. 执行：

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/VirtualDiapauseCell
  export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
  export VDC_PYTHON="$PWD/.venv-autodl-vdc/bin/python"
  export VDC_APPLICATION_SOURCE=/path/to/completed_pk2_repair
  export VDC_APPLICATION_SC="$VDC_PRIVATE_ROOT/application_sc"
  test -f "$VDC_APPLICATION_SC/manifest.json"
  test -f "$VDC_APPLICATION_SOURCE/repair_import.json"
  git -c http.version=HTTP/1.1 fetch origin main
  INSTALLER="$(mktemp)"
  git show origin/main:scripts/install_application_release.sh > "$INSTALLER"
  bash "$INSTALLER"
)
```

`screen` 保持后台进程。CPU 两个查询 worker，Qwen 评价单独一个 GPU 队列；先冻结，后查询。不会同时加载两份 Qwen。来源校验会读已有文件，某些源目录较大时需要等待。不要改动已启动的 release。

离线代码包解压到**新 release 目录**时，直接在该目录设置上述环境变量后执行 `bash scripts/launch_application.sh`，不需要 Git；`PYTHONPATH` 会指向这份 release。

## 查看进度、结果与续跑

```bash
export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
export VDC_PYTHON=/root/autodl-tmp/VirtualDiapauseCell/.venv-autodl-vdc/bin/python
RUN="$(cat "$VDC_PRIVATE_ROOT/LATEST_APPLICATION.txt")"
RELEASE="$(cat "$VDC_PRIVATE_ROOT/LATEST_APPLICATION_RELEASE.txt")"
export PYTHONPATH="$RELEASE/src"
"$VDC_PYTHON" "$RELEASE/scripts/application_status.py" --run "$RUN"
tail -n 60 -F "$RUN/console.log"
```

Ctrl+C 只退出 tail。阶段详情在 `logs/freeze.log`、`logs/core_celltypes.log`、`logs/knowledge_domain.log` 等；知识评价每十题打印一次进度。`exit_code.txt` 出现后已结束：0 表示执行队列成功，2 表示有分支失败/缺输入，**均不表示生物学验证通过**。`application_status.json` 和 `REPORT_CN.md` 分别记录能力边界、每个 profile 的不支持原因。profile 数不是独立 pool 数。

默认结果界面端口 8768，仅监听 `127.0.0.1`。`deployment.json` 必须为 `running_verified` 才确认长期服务启动成功；`browser_verification.json` 只代表临时独立进程检查。通过自己的 SSH 隧道访问，或先在 Jupyter 文件浏览器读取报告，不开放私有结果公网端口。已有服务占用端口时可设置其他 `VDC_APPLICATION_PORT`；不停止旧服务。

已冻结且进程退出后的失败重试：

```bash
bash "$RELEASE/scripts/resume_application.sh"
```

必须使用原 release、原配置及原补充包。已完成任务逐文件核验后复用；不会因为重试而换模型。若 freeze 未完成而留有部分 snapshot，脚本保留现场并要求新 run；确认没有查询后可原协议重新启动，不删除旧目录。不得利用已经看到的预留结果调参再当作同一评价。

## 打包回传

```bash
"$VDC_PYTHON" "$RELEASE/scripts/pack_application_report.py" --run "$RUN"
```

生成私有报告 ZIP。默认包含逐样本 programme、隐藏基因和 TF 结果、来源/冻结记录、知识 prompt/回答与日志，去掉大尺寸全 gene 数组以便传输。原完整数组留在服务器；包内清单记录删掉的数组名、原文件与打包文件两套校验值。需要全部 gene 数组时加 `--full-gene-arrays`。不包含权重或原 counts。不要上传报告到公开仓库。

主要工件：`freeze.json`、`snapshot/*/hidden/`、`applications/*/result.json`、逐样本 NPZ/JSON、`knowledge/*/answers.json`、`knowledge/*/evaluation.json`、`application_snapshot.json`。

## 补充包复现

已有小包可以直接复用；无需再传主归档。如需从原归档重建：

```bash
python scripts/prepare_application_sc.py \
  --archive /private/original_archive.zip \
  --roles /private/sample_roles.json \
  --programmes /private/prepared/core/programmes.json \
  --source-cache /private/application_source_cache \
  --out /private/new_application_sc
```

准备仅做已授权样本的聚合和评分，不加载预测模型、不读取历史 biotime 标签、不选择候选。train 对照与预留热应激摘要存为不同文件；冻结阶段只拟合 train 对照，之后才读取 stress 查询值计算偏离。
