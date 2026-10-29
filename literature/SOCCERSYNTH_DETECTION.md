# SoccerSynth-Detection

- 论文：Haobin Qin 等，*SoccerSynth-Detection: A Synthetic Dataset for Soccer Player Detection*，arXiv:2501.09281 v2，2025-10-03修订；[arXiv页面](https://arxiv.org/abs/2501.09281)，[作者代码/数据链接](https://github.com/open-starlab/SoccerSynth-Detection)。目前按arXiv预印本引用，不称同行评审发表，投稿时再核发表状态。
- 工作：UE合成足球球员检测数据，使用光照、纹理和运动模糊变化；以YOLOv8n对SoccerNet-Tracking和SportsMOT作跨域检测评估，摘要报告合成训练有直接迁移/预训练效果。
- 对本项目：这是最直接近邻，说明“UE足球合成检测数据”不能作创新主张。应定位于不同目标：长边广播视角/号码/时序ID与关键点同步监督、bbox/Pose/MOT逐阶段验证、模型容量和预算控制。
- 适用限制：该工作主要是单帧球员检测；不因此推出它解决本项目的号码、Pose或ID跟踪问题。不同数据划分、检测类别和指标不得横向抄分。
- 阅读记录：保留论文表格中的绝对结果与设置后再比较；不要只摘摘要中的“优于”表述。
