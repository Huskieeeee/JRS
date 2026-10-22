# 主体代码结构

主体算法与公共接口保持原路径，实验实现采用正式 Python 包，命令脚本保留兼容层。

| 层次 | 文件 | 用途 |
|---|---|---|
| 场景执行 | `model.py`、`environment.py`、`paired_worlds.py` | 生成配对场景、接收动作、执行事件与审计 |
| 主体方法 | `recovery_graph.py`、`rg_variants.py` | 构造恢复图，批量评估，按事件续接 |
| 基础与对照 | `planners.py`、`deterministic_legacy.py`、`release_planner.py` 等 | 支撑原 JRS、固定随机候选、v1.5 和当前对照 |
| 预测 | `surrogate.py`、`ensemble.py`、`prediction.py` | 特征与集成预测，完整参数位于聚合 NPZ |
| 实验 | `experiments/campaigns.py`、`runner.py` | 阶段、设备、分片、冻结、断点恢复与逐回合记录 |
| 评估 | `experiments/analysis.py`、`reporting.py`、`plotting.py` | 配对统计、主比较、消融和 PDF/PNG 输出 |
| 计算实验 | `experiments/benchmark.py`、`budget.py`、`merge.py` | 一致性、耗时、计算预算与分片合并 |
| 入口 | `cli.py`、`scripts/` | 统一 CLI 与旧命令兼容 |

所有 Python 实验模块从 checkout 定位配置和模型；在源码目录执行 `.venv/bin/python -m floor_tase --help` 查看命令。

精简仅选择运行需要的文件，并更新项目说明和测试参考文件的位置。34 个包内 Python 模块与完整工程字节一致；没有修改候选预算、随机种子、目标函数或算法。73 个交付文件已包含目录清单和验证说明。
