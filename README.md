# FutsalMOT 合成数据与跨域训练研究

这是一个独立的新实验项目，研究合成足球/五人制数据能否改善真实比赛中的球员检测、姿态估计和身份追踪。原工程与旧实验保持原样。本仓库目前处于**方案与文献梳理阶段**：尚未启动新UE渲染、模型训练或正式评测。

## 研究问题

在真实训练样本和计算预算可比、按比赛隔离训练/验证/测试的条件下，UE合成数据能否提升未见真实比赛中的MOT身份连续性？收益出现在哪些训练策略、比赛难度与模型容量下？BBox和Pose作为上游诊断及次要结果。视角、地面和号码作为数据域匹配/标签质量要求。

## 目录

- `literature/`：核心论文阅读卡、BibTeX及检索范围说明。
- `Docs/LITERATURE_REVIEW_AND_POSITIONING.md`：论文比较、创新边界与候选定位。`Docs/RESEARCH_PLAN_V1.md`：研究假设、阶段路线和冻结门槛。
- `Docs/DATASET_REDISTILLATION_V1.md`：新数据集规格、相机/地面/号码需求和质量门禁。`Docs/SOURCE_ENGINE_AUDIT_V1.md`：源工程实际入口与实现限制。
- `Docs/REAL_DATA_SPLITS_V1.md`：真实数据来源、难度分层和比赛级切分原则。
- `Docs/EXPERIMENT_MATRIX_V1.md`：检测、姿态、跟踪、训练方式及模型尺寸对照。
- `Docs/INTEGRATED_WORKFLOW_V1.md`：三流程、每阶段进入/终止Gate和失败停止规则。
- `Docs/DATASET_CONTRACT_V1.md`：按比赛/seed原子组织、闭合schema和发布目录白名单。
- `schemas/v1/`：结构冻结提案的JSON Schema及Pose顺序定义；生产接入前需验证并绑定SHA。
- `protocols/`：未来冻结协议；运行前需写入全部绝对参数。
- `results/`：新研究结果与失败记录，不回填旧版本。

## 存储

- 新研究媒体根目录：`G:/FutsalMOT_Dataset/synthetic_research_v1`。
- 本仓库保存代码、配置、数据清单、协议和小型结果表；大型图像、mask和逐帧Pose只写入G盘数据根目录。
- 原始工程：`D:/projects/FutsalMOT_UEDataset`；既有研究/权重：`D:/projects/FutsalMOT_Research`。均为参考源，不在本项目里改写或复制。

## 当前边界

1. 旧测试集SNMOT-116/187及封存包不读取、不用于选参；之前已看过的12场比赛全部只可视为开发数据。
2. 训练、验证、测试按完整比赛身份分割；相邻片段、不同相机或同场新时间窗不能算独立比赛。
3. 主结论按预先指定的总体与难度分层报告，不筛容易比赛，不挑最好seed或checkpoint。
4. 数据版本与协议冻结前，不开始正式渲染/训练。所有新参数必须写绝对值，所有结果同时给各比赛绝对成绩、配对差值与区间。
5. 号码渲染、长边相机和绿色地面均须先经过UE真实短样与读回验收；方案文件不代表功能已经实现。

详见 [研究方案](Docs/RESEARCH_PLAN_V1.md) 和 [文献索引](literature/README.md)。







