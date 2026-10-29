# 源工程相机与号码功能核对

核对日期：2026-10-29。仅静态读取 `D:/projects/FutsalMOT_UEDataset` 文档/源码，不启动UE、不开启渲染、不改源工程。

## 相机系统：可复用部分与当前限制

入口和契约：

- `python/src/grf_ue_bridge/config/models.py`：`SimulationCameraConfig` 要求canonical anchor完整为C1–C5；profiles只允许C1–C5和可选P01。C1–C4分辨率允许1920×1080或2560×1440，焦距12/15/18mm；C5可额外3840×2160，焦距8/10/12mm。
- `python/ue/camera_distribution.py`：读取`simulation.camera.distribution.episode_profile`确定实际相机ID；标准`anchor_only`实际运行C1–C5，`anchor_plus_one_partial`为C1–C5+P01。`ue.camera_mapping`把每个相机ID绑定到独立UE Actor与Sequence，实际Profile参数会进入Sequence契约；UE回读实际transform/FOV等写入camera state。
- `python/configs/camera_anchor_smoke_5cam.json`：示例C1–C4位于四角，C5在一端中线；P01为静态长边侧机位，固定`left_half_sideline_high_v1`几何（约16m高、1920×1080、12mm），只覆盖左半场。它不能直接代表完整比赛的主广播机位。
- `Docs/engineering/DATA_CONTRACT.md`说明`coverage=full_field`是所选相机组的联合目标，不代表单个camera footprint自身完整覆盖。

重要限制：schema目前只允许额外P01；P01的模板、FOV和region固定且是半场画面。仅改task JSON不能任意增加P02或新机位，也不能把P01冒充全场。`anchor_plus_two_partial`当前会被运行时代码拒绝。配置结构接受`static_surveillance_ratio`和`dynamic_broadcast_ratio`字段，但在运行时相机选择器中没有发现消费者；因此不能认为填写ratio会自动抽样静态/动态广播轨迹。动态pan/tilt/zoom是否已由其他Sequence实现须单独盘点并实机回读。

建议新项目先以可支持的机制推进：在新任务版本中重新设置C1–C5的受控profiles和camera mapping，使至少一个主视角成为高位长边完整pitch视角；用P01作为额外半场侧机位诊断候选。若需要多个长边偏移/动态broadcast视角，先审查runtime schema与Sequence选择器缺口，再决定扩展，不能只在文档中声明已有。

所有正式角度选择须根据目标真实数据（广播长边或战术全场）和人物像素高度匹配。每个task profile明确位置、旋转、sensor/FOV、焦距、分辨率、Sequence与UE actor；生成后读回camera state并记录逐帧实际值。先比较俯仰、场上覆盖和近/远区人体像素尺寸，再锁完整生产配置。

## 球衣号码系统：已验证原型，不是生产集成

源工程含独立UV号码开发代码和P1号码mask标注器：

- `python/docs/JERSEY_NUMBER_UV_PROTOTYPE.md`记录了1–99确定性分配、UV图集、号码mask与可见号码bbox方案。
- `Source/FutsalMOT/Private/JerseyNumberUVLibrary.cpp`与头文件提供号码参数校验、号码组件/蒙皮mask follower和诊断捕获边界。
- `python/src/grf_ue_bridge/jersey_number_annotator.py`和`annotation_validator.py`可以在appearance v2条件下按逐帧号码mask生成号码bbox。
- 已有短近景证明Player_003/L0、seed42、冻结姿态里背面“94”可清晰显示，正面/完全遮挡不露号；但同一原型文档明确列出七模型/十人、动画/布料/LOD、Sequence/MRQ、多颜色和Object-ID共存/恢复尚未闭环，`production_enabled`和正式v2配置关闭。旧批次仍是无号码数据。

所以建议复用既有协议/原型，不另造号码随机器，但不能把既有原型写成已可批产。新版本前仍需按合法生命周期接入服装、动画和每名玩家，验证高低mip、正背面、移动/遮挡、号码实例mask与RGB同帧同步、身份稳定、保存/恢复。随后完善number字段与OCR可读状态/遮挡的标注契约，再做新UE短样和准入。

## 地板和服装

按源工程`Docs/engineering/GREEN_FLOOR_VARIATION.md`采用既有green-floor catalog/seed算法，序列内固定，小幅色调/亮度为主；不能用旧蓝色RGB声称绿色。球衣继续复用既有appearance seed映射；新号码叠加必须证明材质替换后仍存在，并在native覆盖后实际读回。

## 此审计没有做的事

未运行源代码测试或UE操作，未执行任何task、CameraDistribution生成、MRQ或新数据生产；未修改号码WIP、源工程、已有数据或原始Sequence。下一步需在独立新协议下完成相机Profile准入和号码/绿色短样，不把静态源代码检查当渲染验收。
