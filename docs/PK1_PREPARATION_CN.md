# PK1 数据接入与公共 regulon：已实现范围

本次是 PK1 的数据接入增量，不是完整 PK1 全模块训练结果。旧 Alpha 的入口保留；直接运行 `launch_alpha.sh` 仍然执行 Alpha，不能把它称为新的公共＋killifish 联合训练。

## 数据边界

内部源文件、逐样本角色、下载地址、私有工件全部放在私有目录。公共仓库不保存这些材料。

`vdc.admission` 以协议 ID、逐样本名单和已确认内容指纹允许内部 development 数据。未列入名单的内部数据仍然禁止训练。同一原始材料的细胞、cell-type 伪bulk和其他派生观测继承样本角色及链接；不能更换 biological_unit 来绕过分组。

跨天的共同来源队列单独记录 `cohort_id` 和父样本。时间查询使用 `temporal_query`，不能升级成独立批次验证。现阶段仅准备 development 视图；没有执行冻结后查询，也没有更改旧公共 holdout。

原始 Exit featureCounts 单独解析，核对整数、非负、基因 ID 唯一性与顺序，以及可用 summary 的 Assigned 总数。明确排除名单保留原始文件但禁止拟合。样本编号不是采样小时，历史 biotime 不作标签或选样依据。

单细胞读取明确的 counts 层，先按 pool 或 pool×cell-type 求和，再归一化；每个 cell-type/pool 至少20个细胞，未达门槛的组写入 coverage 审计。历史细胞标签被保留为注释，不声称新训练了细胞类型分类器。

## 在私有服务器准备

沿用已安装 CUDA PyTorch 的虚拟环境，安装本次新增依赖：

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
source .venv-autodl-vdc/bin/activate
python -m pip install -e '.[private-data]'
```

若使用已准备好的 **PRIVATE** 包，解压到仓库外的私有目录。该包带 approved 样本名单、source_lock 和已准备观测，不需要再次批准相同名单。设置：

```bash
export VDC_PRIVATE_ROOT=/root/autodl-tmp/vdc-private
export VDC_ROLE_MANIFEST="$VDC_PRIVATE_ROOT/sample_roles.json"
python scripts/prepare_pk1_private.py verify --roles "$VDC_ROLE_MANIFEST"
```

若选择从原始归档重新准备，下载脚本读取单独的 `download_sources.local.json`。文件含私有链接与本地核验 SHA256，绝不能提交 GitHub。

```bash
python scripts/download_pk1_private.py \
  --config "$VDC_PRIVATE_ROOT/download_sources.local.json" \
  --out "$VDC_PRIVATE_ROOT/downloads" --list-only
```

确认列出的归档名后移除 `--list-only` 下载。下载按固定字节数及 SHA256 核验，失败不打印链接；已经存在但校验失败的文件不会被静默覆盖。远端是否允许完整大文件下载仍由实际下载结果决定。

`prepare_pk1_private.py import --help` 显示导入入口：归档路径及排除样本必须由私有配置提供。导入产生 draft，不自动解锁训练。`restore-approval` 只能恢复**内容和源文件指纹完全一致**的已批准名单，不允许用新数据套旧批准。全新名单需重新审阅。

准备开发观测使用：

```bash
python scripts/prepare_pk1_private.py prepare \
  --private-root "$VDC_PRIVATE_ROOT" \
  --annotation knowledge/snapshots/ensembl_2026-10-03/nfurzeri_go.tsv \
  --public-bundle /path/to/existing/alpha/data/dauer \
  --out "$VDC_PRIVATE_ROOT/prepared_new"
```

输出目录不能覆盖。共享面板由固定外部 GO 成员约束产生，保留其与旧公共面板的差异。没有表达驱动选样，没有生成 depth，没有把 counts 输入当训练成功。

## 公共 regulon 分支

采用 [CelEsT](https://github.com/IBMB-MFP/CelEsT-app) 固定 commit 的 v1.1 网络。版本、字节数和校验码见 `configs/pk1_regulon_source.json`。WormBase GAF 的 ID、symbol、synonym 只接受唯一映射；丢弃歧义与自边。正的网络置信度不被解释为调控效应正负号，打分是 unsigned target-expression proxy。

```bash
python scripts/run_public_regulon.py \
  --public-run /path/to/existing/alpha_run \
  --out runs/pk1_regulon_new --download
```

实际本地开发运行得到429个可用 regulon，参考曲线保存重载通过。在同一支持范围中，验证 MSE 为 0.00007712，训练均值基线为 0.00015187。该结果复用现有公共开发划分及参考派生坐标；不是新独立 test，也不是机制验证。网络原始训练研究与全部后续评价研究的交叉关系尚未完整建立，不能声称网络与所有评价来源严格独立。

## 本次验证和仍未执行的部分

- 软件测试：102 passed。覆盖准入不可伪造、原材料链接、时间查询范围、原始 counts、下载错误不泄漏私有 ID，以及数值参数迁移时保留本地尺度和恢复记录。
- 已执行：真实归档审计、经用户确认的角色冻结、开发观测准备、公共 regulon 打分/曲线拟合/评价/重载。
- 未执行：新 PK1 Qwen 训练、K0–K5 完整对比、killifish 新 clock/waves 拟合、冻结后查询、PK1 整体服务联调。
- 待接入：新增公共时序/干预数据与扩大知识语料；现有小型 Alpha 语料不冒充扩展语料。
- 功能 depth 仍缺表达—功能终点的对应关系，不用随机头或 clock 补数。

因此，本文件不能替代未来完整 PK1 的启动说明。`state.fit_state(pretrained=...)` 已提供受约束的数值迁移能力，但完整路线调度与真实效果尚未据此完成。
