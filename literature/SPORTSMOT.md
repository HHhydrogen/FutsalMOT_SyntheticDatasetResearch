# SportsMOT

- 论文：Yutao Cui 等，*SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes*，ICCV 2023；[arXiv](https://arxiv.org/abs/2304.05170)，[CVF论文页](https://openaccess.thecvf.com/content/ICCV2023/html/Cui_SportsMOT_A_Large_Multi-Object_Tracking_Dataset_in_Multiple_Sports_Scenes_ICCV_2023_paper.html)。
- 数据与任务：240个序列、超过150K帧、超过1.6M框，覆盖篮球、排球、足球，追踪场上全部球员。
- 贡献：突出体育场景中的快速/变速运动和相似但可区分外观；提出MixSort，用外观关联补充tracking-by-detection的运动关联，并分析关联是主要挑战之一。
- 对本项目：比只训练bbox更需设置固定检测输入的关联实验；球队相似球衣、遮挡、交叉应纳入场景分层。SportsMOT可作为候选开发来源，但不是纯足球独立比赛数。
- 限制：不同运动和拍摄视角混合；论文结果只在其数据/协议内解释。需单独审查足球子集、train/test身份与许可。
