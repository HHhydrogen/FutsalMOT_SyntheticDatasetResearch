# SoccerNet-Tracking

- 论文：Anthony Cioppa 等，*SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer Videos*，CVPR Workshops 2022；[arXiv](https://arxiv.org/abs/2204.06918)，[CVPRW PDF](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Cioppa_SoccerNet-Tracking_Multiple_Object_Tracking_Dataset_and_Benchmark_in_Soccer_Videos_CVPRW_2022_paper.pdf)。
- 数据：论文报告200个30秒序列、25 fps，来源12场比赛，并提供一段完整45分钟半场作长时追踪。标注含bbox和tracklet ID；目标包括球员、裁判和球。官方repo说明训练/测试/challenge分割。
- 贡献：足球MOT数据集与基准；强调高速运动、多人遮挡、长时身份连续性仍困难。MOT一般使用tracking-by-detection：每帧检测，再关联成轨迹。
- 对本项目：支持比赛/序列级划分和单独报告困难事件；固定检测关联评估与端到端检测+关联评估应分开。官方任务常按比赛片段报告，但研究结论不应只有单场，需跨比赛汇总。
- 可比性：职业足球广播视频、类别含裁判/球，目标分布及官方challenge协议和本地五人制片段不同；公开排行榜不能直接作为本项目排名。已被旧项目观察的比赛仍不可作为新test。
- 待核：新项目候选源与旧12场的比赛ID重叠、当前release许可及是否有未观察比赛。
