# JRS 主体代码 · Codex CPU 环境

本包从 FLOOR-MAS / JRS-RG v1.6 工程中选取主体代码，交付文件总数 **73 个**。保留完整算法包、22 方法配置、固定集成模型、实验执行和汇总入口；去除历史实验输出、逐回合日志、参考图片与旧版辅助脚本。

## 上传 GitHub

解压后将 **JRS 文件夹内的文件与子目录** 上传到仓库根目录。仓库首页应直接看到 `README.md`、`pyproject.toml`、`src/`、`scripts/`。完整文件列表见 `FILE_LIST.txt`。

## 配置 Codex 云端环境

使用 Python 3.12（最低 3.10）、CPU，网络选择 Package managers。安装脚本填写：

```bash
bash scripts/dev/cloud_setup.sh
```

脚本定位项目目录，创建 `.venv`，安装 NumPy、SciPy、pandas、Matplotlib 及可编辑项目，运行 18 项测试和 6 回合开发验证。项目运行不需要 API 密钥，CPU 安装不包含 Torch。

测试通过后发布环境。后续任务从项目根目录执行：

```bash
.venv/bin/python -m floor_tase doctor
.venv/bin/python -m floor_tase smoke --out results/dev_01 --workers 1
```

完整开发检查：`bash scripts/dev/check.sh`，会自动选用 `.venv/bin/python`。

## 主体代码定位

| 位置 | 职责 |
|---|---|
| `src/floor_tase/model.py`、`environment.py` | 场景、动作、质量状态、原 J 与轨迹审计 |
| `src/floor_tase/recovery_graph.py`、`rg_variants.py` | JRS-RG 图编译、随机评估、事件续接和内部消融 |
| `src/floor_tase/` 其他模块 | 当前方法所需的基础方法、对照与预测支持 |
| `src/floor_tase/experiments/` | 配置、冻结执行、配对统计、报告、绘图与合并 |
| `scripts/` | v1.6 兼容命令和 CPU/多 GPU 运行入口 |
| `models/v15_ensemble_cpu.npz` | v1.5 对照所需的完整集成参数；已包含各成员权重 |
| `tests/` | 机制、配置一致性、脚本兼容和运行流程检查 |

详见 `docs/ARCHITECTURE_ZH.md`，研究机制与实验设计见 `docs/INNOVATION_AND_EXPERIMENTS_ZH.md`。

## v1.6 完整实验入口

开发环境日常使用 6 回合检查。完整 CPU smoke 为 198 回合：

```bash
source .venv/bin/activate
bash scripts/run_v16.sh smoke 1 cpu
```

`probe` 为 792 回合，`main` 为 66,000 回合，参数与原版一致。GPU 入口保留，使用前需在对应服务器安装匹配的 Torch/CUDA 并显式选择 `cuda`。

精简后的运行源码清单改变，已有冻结实验使用新的输出目录。临时环境、生成配置、结果与日志不属于 73 个交付文件，也不应上传 Git。模型与算法未重新训练或调参。历史验证数据仍保留在之前的完整交付包中。
