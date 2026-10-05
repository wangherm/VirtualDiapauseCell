# Virtual Diapause Cell — CW-stage-1

以 **rank＋本地表达幅度 → 身份与背景 → central clock → gene/TF RNA/programme waves → 观测偏离** 为主线的阶段性实验模型。

当前收尾范围已固定，不再扩展数据、架构或大网格。不是已科学验证的通用休眠模型，也不提供 depth 或未来状态预测。

## 唯一推荐入口

**[阶段运行、进度、打包与浏览说明](docs/STAGE_RELEASE_CN.md)**。使用原 CW runner 与已有浏览器：

```bash
bash scripts/vdc_stage.sh start
bash scripts/vdc_stage.sh status
bash scripts/vdc_stage.sh pack
bash scripts/vdc_stage.sh browse
```

先按运行说明设置 AutoDL 环境与已完成的 CW1 修订来源。GitHub 不通时使用离线代码包；无需重传数据或重跑 PK2。`start` 后台执行，`browse` 是仅监听本机的**保存结果浏览**。

## 固定路线

|视图|默认路线|保留的对照|
|---|---|---|
|Bulk|clock＋身份＋可见 programme residual|匹配训练行 direct Ridge，不能隐藏其更低误差|
|Core 整 pool|较简单的 clock＋身份|残差候选与 Ridge|
|粗身份 profile|身份条件 clock/waves；关闭当前 residual 修正|匹配 Ridge 与原负结果|

已有可靠身份优先继承。Qwen 是辅助证据评价，不接管默认身份；身份确认、clock 支持和表达预测分别记录状态。固定字段协议重新生成原卡、无数值提示和证据挑战，不修补旧答案。

## 本轮工作边界

默认队列共 34 项：24 项实际工件校验复用，4 项 Qwen/依赖链重跑，3 项唯一的强收缩诊断，3 项默认数值路线重载调用。探索网格、整套 PK2、旧压力扫描不进入默认队列；历史记录保留。

仅增加一个预定 alpha=100 的残差诊断，不自动更换上述路线。固定规则下完成评价即可收尾，不以全部胜过 Ridge 或覆盖率达到 100% 为条件。

本地已核验真实历史报告并完成数值检查；**新的 fixed-evidence GPU 推理尚需 AutoDL 实际执行**。每次真实状态以 run 的 `STAGE_SUMMARY.json`、`MODEL_CARD_CN.md`、`stage_snapshot.json` 为准。未完成时标记 `partial_not_closed`，完成固定队列后才标记 `frozen_experimental`。

## 模型、证据与限制

[给生物学与计算合作者的阶段说明](docs/STAGE_MODEL_CN.md) · [当前实施状态](docs/STATUS.md)

Clock 是身份内的参考坐标，不是实际小时、恢复百分比或功能深度。Early 范围外、blood 缺锚点及独立锚点重复不足保持显式不可用/外推，不通过修改查询范围掩盖。已查看数据仍为开发或回顾性材料，不重新命名为独立 test。

## 开发检查

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/validate_catalogue.py
```

目录验证和合成测试只检查软件，不表示真实模型训练成功。

## 历史与项目定义

[历史入口归档](docs/archive/README_BEFORE_STAGE.md) · [早期架构](docs/ARCHITECTURE_CN.md) · [功能模块](docs/MODULES_CN.md) · [数据/文献目录](knowledge/datasets.json) · [真实 run 审计](docs/CW1_RUN_AUDIT_CN.md)

历史代码、预测、失败和负结果不删除，默认浏览器通过“辅助/历史任务”查看本 run 的辅助结果；更早探索保存在其原 run。FactorBridge 为已归档参考项目，不是此项目依赖。私有表达、样本角色、链接与模型结果不上传公共仓库。
