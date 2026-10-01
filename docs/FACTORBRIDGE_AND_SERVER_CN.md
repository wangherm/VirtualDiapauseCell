# FactorBridge 参考与同服务器部署

## 独立项目

FactorBridge 已在 `archive-2026-10-01-pilot` 保存实验快照；本次读取的本地整理提交为 `d359e46`。它继续作为研究历史保存，本项目不修改它的训练代码或数据。新仓库没有 `factorbridge` 依赖、submodule 或复制的训练框架。

| 参考对象 | 可借鉴内容 | 不能直接沿用 |
| --- | --- | --- |
| [data.py](https://github.com/wangherm/FactorBridge/blob/archive-2026-10-01-pilot/factorbridge/data.py) | manifest、真实文件审计、local_split、assert_lineage | 旧 8 个研究分布和训练角色；新任务重新做准入 |
| [factors.py](https://github.com/wangherm/FactorBridge/blob/archive-2026-10-01-pilot/factorbridge/factors.py) | 按 unit bootstrap、数值投影、稳定性与 W/Z | PCA 参考当作新程序是否正确的唯一裁判 |
| [semantic.py](https://github.com/wangherm/FactorBridge/blob/archive-2026-10-01-pilot/factorbridge/semantic.py) | 版本化映射、实测背景下富集、覆盖缺失 | 人/鼠/大鼠映射等于 killifish 覆盖；pathway 名等于机制 |
| [programmes.py](https://github.com/wangherm/FactorBridge/blob/archive-2026-10-01-pilot/factorbridge/programmes.py) | rank score、NaN 掩码、训练参考曲线与未来时间对照 | RNA rank score 叫通量；ageing 曲线当 diapause clock；文件内部耦合不适合直接拷贝 |
| [llm.py](https://github.com/wangherm/FactorBridge/blob/archive-2026-10-01-pilot/factorbridge/llm.py) | completion-only mask、真实 GPU smoke、保存重载与运行签名 | 已失败的 adapter、旧 JSON support-selection 目标及 token-loss-only 模型选择 |
| [evaluate.py](https://github.com/wangherm/FactorBridge/blob/archive-2026-10-01-pilot/factorbridge/evaluate.py) | study 汇总、数值对照、公平候选池与冻结评估 | 旧弱参考恢复率当成新项目的 biological accuracy |

未来确实需要某个数值函数时，逐项移植并注明原版本、许可证情况和回归验证；当前没有为“可能用到”而复制整套代码。归档实验中 56/56 uncertain 是旧任务的负结果，不证明 LLM 对新知识任务必然无用，也不支持直接继续旧 adapter。

## AutoDL 目录约定

```text
/root/autodl-tmp/
  FactorBridge-upload/          原目录先保持不动
  archives/factorbridge/        经校验的归档副本
  VirtualDiapauseCell/          新 GitHub 项目
    .venv/                     独立环境
    data/                      新项目自己的审计与数据
    runs/                      新实验，不覆盖旧 runs
  huggingface/                 可共享已经校验的基础模型缓存
```

本次没有 AutoDL SSH 连接，**未在服务器上执行归档、安装或训练**。不删除旧目录。公共仓库不包含旧 adapter、矩阵、私有知识或服务器日志。

GitHub 连通时在 JupyterLab 终端执行：

```bash
cd /root/autodl-tmp
git -c http.version=HTTP/1.1 clone https://github.com/wangherm/VirtualDiapauseCell.git
cd VirtualDiapauseCell
python scripts/validate_catalogue.py
```

公开仓库不需要账号密码。若仍连接超时，上传本次代码 ZIP 并解压到独立的 `VirtualDiapauseCell` 目录；归档和定义校验无需联网。此阶段不需要安装 PyTorch，也不从旧 `.venv` 导入训练依赖。

## 保存旧项目

先确保旧训练、下载与评估停止。下面脚本默认只显示计划与空间检查；`--execute` 才会创建 tar 副本，不移动或删除源目录，排除可重建的 `.venv` 和 Python bytecode。

```bash
cd /root/autodl-tmp/VirtualDiapauseCell
bash scripts/archive_factorbridge.sh /root/autodl-tmp/FactorBridge-upload /root/autodl-tmp/archives/factorbridge
```

正式归档可能较大，用 screen：

```bash
screen -S archive_factorbridge
cd /root/autodl-tmp/VirtualDiapauseCell
bash scripts/archive_factorbridge.sh /root/autodl-tmp/FactorBridge-upload /root/autodl-tmp/archives/factorbridge --execute
# Ctrl+A 然后 D，离开 screen；用 screen -r archive_factorbridge 返回。
```

归档保留该目录内代码、配置、数据、运行记录与 adapter，输出 tar、SHA256 和内容清单。目录外的 Hugging Face 缓存及其他数据不在此 tar 内，不作清理。源文件变化或 tar 出错时保持 `.partial` 并失败，不宣称归档完成。活动进程检查只作额外保护，不能替代确认所有写入已停止。

后续新模型训练前，重新检查实际 GPU 和依赖并做新项目的真实 smoke。旧环境报告显示约 96 GiB 显存，不能沿用最初“120 GB”的估计；显卡类型和容量以新运行 `nvidia-smi` 为准。
