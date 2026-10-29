# WorldPose

- 论文：Tianjian Jiang 等，*WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation*，ECCV 2024；[arXiv](https://arxiv.org/abs/2501.02771)，[ECCV论文PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02927.pdf)。
- 数据：2022世界杯多机位资料，论文报告88序列、250万以上3D人体姿态和120km以上轨迹；包含广播相机参数，研究全局3D Pose。
- 贡献：利用静态多机位恢复全局多人体3D姿态/轨迹，并校准移动广播相机。
- 对本项目：足球场景不缺Pose/全局运动公开研究；创新要更明确地针对低成本合成监督、五人制身份跟踪和可控训练效应。
- 限制：高质量真实多机位/3D系统与单目UE合成数据采集条件不同；其3D姿态不等同于本项目2D检测器crop Pose标签。
