# MOTX：合成数据用于MOT关联学习

- 论文：Yuchi Liu 等，*Synthetic Data Are as Good as the Real for Association Knowledge Learning in Multi-object Tracking*，arXiv:2106.16100 v3；[arXiv](https://arxiv.org/abs/2106.16100)。
- 贡献：提出可控3D合成引擎MOTX，配置相机和物体运动，使其接近真实MOT数据；论文报告合成关联知识在若干真实基准上接近真实数据训练，并能研究运动因素。
- 对本项目：强烈支持用户对机位分布的关注。合成相机视角、移动和物体运动应贴合真实目标域，不能仅增加纹理随机化。MOT固定检测/关联模块可单独估计合成身份窗口作用。
- 限制：研究对象是行人MOT，不是足球运动；论文结论受其检测输入、特征和测试集约束。“合成与真实一样好”是论文标题，不是本项目可直接采纳的结论。
