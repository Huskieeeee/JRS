# 云端 CPU 配置

1. 将解压后的项目文件放在 JRS 仓库根目录，能直接看到 `pyproject.toml`。
2. 在 Codex 云环境中选择 JRS 仓库，使用 Python 3.12，允许 Package managers 网络访问。
3. Install script 填写 `bash scripts/dev/cloud_setup.sh`。
4. 安装与检查通过后发布环境。

常用环境变量为 `MPLBACKEND=Agg`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`。安装脚本与检查脚本会在执行时设置这些值；在云环境设置中填写可让它们对其他任务命令同样生效。

后续任务在包含 `pyproject.toml` 的目录执行：

```bash
.venv/bin/python -m floor_tase doctor
.venv/bin/python -m floor_tase smoke --out results/cloud_dev_01 --workers 1
```

日常环境检查运行 18 项测试与 6 回合 CPU 仿真，完整 v1.6 smoke 入口另见 README。该安装流程无需项目 API 密钥，不会安装 Torch。实际 CUDA 测试在具有 CUDA 的服务器上另行运行。

如果上传到 GitHub 的仍然只有 ZIP，仓库尚未提供直接可编辑的主体代码。请先将本包解压，上传内部文件与子目录。已有云环境需要重新准备并发布，才能让后续新任务使用新的安装设置。
