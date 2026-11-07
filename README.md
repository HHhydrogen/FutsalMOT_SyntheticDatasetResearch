# FutsalMOT 合成数据研究

BBox阶段已于2026-11-05完成并暂时结束：24个主表模型、48项两场比赛外确认均已终验；原始模型/预测/失败证据保留，有限自动跟进已删除。不自动开启新训练、推理或调参。

- [BBox正式实验、完整结果和纳入标准](docs/exp/BBOX_RESULTS.md)
- [正式机器清单](docs/exp/bbox_formal_registry.json)
- [唯一实验日志](docs/exp/EXPERIMENT_LOG.md)
- [当前待办与职责](docs/plan/TODO.md)

## 现行方案

[PAPER](docs/plan/PAPER.md)、[DATASET](docs/plan/DATASET.md)、[EXPERIMENTS](docs/plan/EXPERIMENTS.md)保留原冻结SHA及原科学边界。本次清理没有重写已执行协议；旧文中的“原始诊断保留”已被2026-11-05用户明确清理指令在对应弃用范围覆盖，详见唯一日志和清理清单。

## 目录

- `src/futsalmot`：BBox正式数据/训练/记录/评价代码及共享基础设施。
- `configs`：固定子集manifest和共享预览配置；`experiment.yaml`仅供通用只读CLI预览，正式参数以注册表绑定的实际job table为准；`pyproject.toml`、`requirements.lock`：包定义与依赖锁。
- `tests`：对应的CPU契约、状态、几何和生命周期回归源码。
- `tools`：正式预检/批量控制/确认接续、汇总与只读观察台；不默认启动实验。
- `docs/exp`：正式结果索引、可提交的小结果及唯一实验日志。
- `runs`、`weights`、`cache`、`.venv`：本机产物，Git忽略；大证据引用G盘原位目录。
- `docs/paper`：已有本地文献库；PDF不混入本次BBox提交。

实际研究Git根为`D:/projects/FutsalMOT_SyntheticDatasetResearch`；UE生产工程在另一个Git根`D:/projects/FutsalMOT_UEDataset`，本次BBox提交不包含其改动。原始数据仍在`G:/FutsalMOT_Dataset`，训练/评价证据在`G:/FutsalMOT_Training/runs`；不通过junction绕过路径边界。

## 当前职责

本聊天仅负责BBox。Seg交接聊天`01a1a627-79a8-7af1-b59e-abb69fd9fa2b`，Pose交接聊天`01a1a627-8a05-7c63-a9ba-c90ae819a89e`；其源码、资料和未完成工作保留，由用户在各聊天安排，不并入本次BBox提交。

## 只读观察台

观察台位于 [`tools/training_monitor/`](tools/training_monitor/)，包含 Python 服务、页面、图表脚本和 Windows 启动入口。双击 [打开训练监控.cmd](tools/training_monitor/打开训练监控.cmd)，或在研究仓库根目录运行：

```powershell
.venv\Scripts\python.exe -B -X utf8 -m tools.training_monitor --port 18765 --open
```

[本地观察台](http://127.0.0.1:18765)显示训练与48项确认记录。启动入口会复用可信的本地服务，或按默认12小时启动新的只读会话。原会话的绝对截止时间仍记录在实验日志中。

## 提交边界

只提交明确的BBox/共享依赖文件和小结果；不使用`git add -A`，不提交模型、环境、GT、raw预测、服务PID登记或无关Seg/Pose/UE工作。当前研究仓库没有配置远端；本次为本地Git提交。
