# TrackID3x3

- 论文：Kazuhiro Yamada 等，*TrackID3x3: A Dataset and Algorithm for Multi-Player Tracking with Identification and Pose Estimation in 3x3 Basketball Full-court Videos*，MMSports 2025；[arXiv](https://arxiv.org/abs/2503.18282)，[作者仓库](https://github.com/open-starlab/TrackID3x3)。
- 数据/任务：室内固定、室外固定和无人机三类3x3篮球视角；仓库说明提供6名场上球员bbox，并为部分帧提供10个Pose点；Track-ID关注固定相机下球员跟踪和识别。
- 贡献：体育场景中联合跟踪、识别与Pose数据/基准，覆盖业余固定相机/无人机视角。
- 对本项目：Pose+身份+多视角组合有近邻；差异应落在足球/五人制UE时序生成、控制变量与合成到真实的训练证据，而非任务字段组合本身。
- 限制：篮球动作、队伍人数、球衣和摄像机运动分布不同；不能直接比较数值。
