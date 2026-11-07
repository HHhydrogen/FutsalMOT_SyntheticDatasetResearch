# 研究项目执行规则

## 当前状态与职责

2026-11-05用户明确：BBox实验暂时到此为止，整理正式记录、删除非正式内容并提交一次Git。24主表模型＋48比赛外确认已完整终验并停止；无自动训练、评价、参数搜索或新seed授权。唯一结果入口为`docs/exp/BBOX_RESULTS.md`与`bbox_formal_registry.json`，实际过程写唯一`docs/exp/EXPERIMENT_LOG.md`。

本聊天只负责BBox；Seg、Pose分别归聊天`01a1a627-79a8-7af1-b59e-abb69fd9fa2b`、`01a1a627-8a05-7c63-a9ba-c90ae819a89e`。不修改、删除、测试或启动其它聊天的任务；共享GPU互斥，UE生产和每日g保持。

## 先读与方案

先读README、`docs/plan/TODO.md`，再按需读PAPER/DATASET/EXPERIMENTS。现行三份科学方案和TODO原位维护，不创建并行方案、Prompt或运行archive。EXPERIMENTS当前冻结SHA `32c9415b2d81de7fce6a9221af72495822ffe3b70e537004b7bb64af02b065a2`，PAPER `d48cb4160a513773cde0e910fe1c45c7eb5a63bf058871eafd38e9ad27b71c17`；已执行协议不回写/重封。用户明确新修改优先于旧冻结；如需调整科学定义先讨论，不借实现偷换口径。

## 正式证据与清理

保留24主表训练、4项正式策略/预算对照、48固定确认及域差分析。2026-11-07用户指定删除校准R50/R100及36750步R→R、S→R运行产物，单阶段R仍引用的两份小收据原位保留。正式负结果、失败、中断、准入、来源与源码绑定属于正式证据，不按成绩筛选。原GT、公开权重、缓存、checkpoint、完整状态/逐步日志和native预测原位保留，不伪造缺失。旧S3000截断前缀开发诊断已按用户清理授权退休；其D组仅保留budget1500/short_pair/fixed_mix确切依赖的冻结代码祖先，不能运行或恢复该旧实验。

CPU/toy/微基准等非主线临时输出展示后精确删除；保留有效测试源码与正式依赖的小收据。清理须先清单与引用核验，不删除必要正式来源，不移动整组、不跟随junction、不对媒体/模型重复全hash。Windows递归删除先核最终绝对路径，使用单一PowerShell和`-LiteralPath`，不把枚举结果拼成跨shell删除命令。

## 运行与科学边界

新训练/诊断/推理须用户明确安排及完整绝对协议/资源/互斥/生命周期；已结束BBox不自动重放。人工取消不作为故障恢复，不清未知锁、不按历史PID强杀。新失败保留证据后停止，不自动追加attempt或重置预算。正式比较从公开起点独立训练，末步EMA、不best；固定UID/split/GT不伪装，已观察比赛历史如实登记。R24000R与RS12000R12000S是整体配方比较；三seed不是三场比赛，帧/clip不当独立比赛，SD不是CI，未预定等效/非劣效界限不自行设PASS。AMP、TensorRT、batch或验证频率变化不能热改原冻结实验。

## Git与存储

本研究Git根`D:/projects/FutsalMOT_SyntheticDatasetResearch`；UE根`D:/projects/FutsalMOT_UEDataset`为独立仓库，禁止混入其WIP。简体中文技术文档、说明和提交信息；精确暂存，不`git add -A`、force-push或跳hooks。用户已授权本次BBox提交，不重复请示；无远端不擅自创建远端或推送。

环境/缓存/模型/原GT/raw预测/本机收据不提交；G原数据与D/G正式run只按注册表引用，保留原bytes与原SHA。Seg/Pose本机内容不因未提交而删除。只读观察台沿原绝对2026-11-07T00:00:41.387415Z截止，不自动延长；页面不控制训练或推理。
