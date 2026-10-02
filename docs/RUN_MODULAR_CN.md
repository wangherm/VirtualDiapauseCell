# 运行模块化框架

Python 3.10+。在 AutoDL 先检查已有 CUDA PyTorch；不要用本地 Windows 的 CPU wheel 覆盖它。此轮不需要加载 Qwen，不启动联合训练。

```bash
python -m venv .venv
source .venv/bin/activate
# 按机器安装/复用正确的 PyTorch，然后安装项目。
python -m pip install -e '.[test]'
python scripts/doctor.py
python scripts/validate_catalogue.py
python -m pytest -q --basetemp=work/pytest_fresh_run
python scripts/smoke.py --out work/synthetic_fresh_run
python -m vdc --help
```

Windows 激活命令为 `.venv\Scripts\Activate.ps1`。`--basetemp` 必须选择专用的新目录，pytest 会管理该目录。合成 smoke 仅验证代码，不是生物学结果。详细模块命令与数据字段见 [数据契约](DATA_CONTRACTS_CN.md)。

## 一份公共数据的实际流程

每次选择新输出目录。默认不访问网络；加 `--download` 才获取清单中的三个文件。镜像内容如果变化，校验会明确失败；请审核新版本，不能删掉校验绕过。

下载显示文件名、尝试次数和字节进度。网络超时或暂时性服务错误最多尝试 5 次，每次读取超时为 180 秒。已完整校验的缓存复用；小文件的未完成下载从头重试，不假设服务器支持断点续传。SHA256 不匹配不会重试后默许通过。若在数据下载阶段失败，更新代码后可以直接重跑下面的数据准备和试点命令，使用新的输出目录，无需重做已经通过的软件测试。

```bash
python scripts/prepare_public_pilot.py --download --out data/prepared/GSE288723_v1
python -m vdc audit data/prepared/GSE288723_v1
python scripts/run_public_pilot.py data/prepared/GSE288723_v1 --out runs/public_pilot_v1 --steps 50
```

也可把提供的离线包中 `data/raw/GSE288723/` 上传到相同位置，去掉 `--download`。代码沿用同一条路径并做相同哈希检查。

首个试点只训练 50 步 CPU 小模型。输出含状态 checkpoint、优化器、step 日志、latent/programme 预测、独立评价、按历史分层的实际时间曲线与残差。clock、Qwen、真实扰动、功能 depth 和研究批准不会因此自动启用。

## 长任务后台执行

在 AutoDL 使用 `screen`，先进入新项目目录，避免覆盖旧 FactorBridge：

```bash
screen -S vdc-pilot
source .venv/bin/activate
set -euo pipefail
mkdir -p runs
python scripts/prepare_public_pilot.py --download --out data/prepared/GSE288723_server_v1 2>&1 | tee runs/prepare_server_v1.log
python scripts/run_public_pilot.py data/prepared/GSE288723_server_v1 --out runs/public_pilot_server_v1 --steps 50 2>&1 | tee runs/pilot_server_v1.log
```

按 Ctrl-A、D 脱离；`screen -r vdc-pilot` 返回。可用 `tail -f runs/pilot_server_v1.log` 看控制台，训练细项在 `runs/public_pilot_server_v1/state/steps.jsonl`。`status.json` 存在且命令返回成功才表示整条试点完成。不要删除失败日志后冒充首次成功。

此仓库没有连接或操作你的 AutoDL；服务器命令仍需你运行。原 FactorBridge 的服务器归档脚本保持独立，未在本轮自动执行。
