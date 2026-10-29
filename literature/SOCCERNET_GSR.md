# SoccerNet Game State Reconstruction (GSR)

- 论文：Vladimir Somers 等，*SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap*，CVPR Workshops 2024；[arXiv](https://arxiv.org/abs/2404.11335)，[CVF PDF](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Somers_SoccerNet_Game_State_Reconstruction_End-to-End_Athlete_Tracking_and_Identification_on_CVPRW_2024_paper.pdf)。
- 数据：200个30秒序列，2.36M以上球员场上位置标注，并含角色、球队与球衣号；同时有场地线点/相机校准标注。提出GS-HOTA与端到端基线。
- 贡献：将球员身份与真实场地坐标结合，正式提出足球Game State Reconstruction任务；强调MOT bbox轨迹缺少球员语义身份与场地定位。
- 对本项目：球衣号码字段和广播视角很有现实意义，但“足球MOT以前没号码”不成立。新数据应把号码作为可见外观属性及身份关联证据，记录真实遮挡/不可读，不以号码替代稳定track ID。
- 限制：GSR场地坐标任务不同于普通bbox MOT；SoccerNet公开数据许可和既往暴露要核对。
