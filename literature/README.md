# 文献库

本库是面向研究设计的首轮检索与阅读卡，优先保存论文/作者项目原始链接，不在仓库镜像受版权限制的PDF。每张卡片分别记录原论文明确贡献、对本项目的启发、可比性限制和待核验事项。检索截至2026-10-29；不是穷尽性系统综述，投稿前需补引文链、发表状态、许可与最新后续工作。

## 重点阅读

| 领域 | 文献 | 本研究用途 |
|---|---|---|
| 足球tracking | [SoccerNet-Tracking](SOCCERNET_TRACKING.md) | 足球序列、Tracking-by-detection、比赛级协议与官方数据边界 |
| 多运动MOT | [SportsMOT](SPORTSMOT.md) | 体育域关联难点与多运动视频测试 |
| 足球合成检测 | [SoccerSynth-Detection](SOCCERSYNTH_DETECTION.md) | 直接近邻：UE足球合成数据、检测跨域迁移 |
| 合成MOT | [MOTSynth](MOTSYNTH.md) | 引擎合成数据支持检测/ReID/分割/追踪的先例 |
| 合成训练追踪 | [MOTX](MOTX.md) | 合成关联模型迁移和场景运动因素控制 |
| 足球号码与身份 | [SoccerNet-GSR](SOCCERNET_GSR.md) | 足球任务中队伍/角色/号码/身份和相机标定字段 |
| 多视角Pose/ID | [TrackID3x3](TRACKID3X3.md) | Pose与身份联合的体育tracking近邻 |
| 真实足球Pose | [WorldPose](WORLDPOSE.md) | 足球多人3D Pose及全局身份的真实数据近邻 |
| 运动数据 | [SoccerTrack](SOCCERTRACK.md) | 真实足球追踪数据与多类型相机视角 |

## 初步判断

合成体育检测、合成多人体MOT、球衣号/球员身份和体育Pose数据各自已有近邻。因此潜在新意需要依赖受控组合与证据：逼近目标长边广播视角的可控UE时序监督；号码、Pose、mask、bbox、ID共帧；按比赛隔离真实迁移评估；将数据策略与YOLO容量交叉比较。现阶段只能称“候选研究定位”，需在完整检索后再做原创性主张。
