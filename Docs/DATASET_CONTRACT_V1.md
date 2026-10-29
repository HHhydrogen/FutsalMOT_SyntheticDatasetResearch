# 新合成数据集结构与Schema契约 v1

状态：结构冻结提案。对应JSON Schema草案位于`../schemas/v1/`，关键点采用`futsal_body12_v1`。第一次生成前必须对schema做针对性验证、把数量/分辨率/FPS/阈值等`TBD`解析为绝对值，并将所有schema SHA写入运行协议；后续变更需提升schema/dataset version。schema草案是设计交付，尚未接入生产validator。

## 设计规则

- 一个真实完整比赛是一个真实数据原子单元；一个完整模拟比赛/seed是一个合成原子单元。任何clip/camera/appearance variant都继承父比赛split，绝不单独随机分组。
- 原始下载真实数据保持只读和原目录格式；不能为统一布局而移动或覆盖源文件。用source manifest/adapter读取或生成许可允许的派生视图。
- 合成发布使用固定canonical结构；MOTChallenge的`img1/gt/seqinfo.ini`便于现有工具读取，同时保留唯一的rich `annotations.jsonl`作为上游标注源。YOLO/MOT/Pose文本标签只允许从它派生，不能手工编辑或反向覆盖。
- Schema采用闭合对象（`additionalProperties: false`）；未知字段、额外目录/文件、未知enum、帧数/摘要不一致均fail closed。缺字段不能靠agent猜测补齐。
- 正式数据使用只读版本发布。变更产生新dataset version和新fingerprint，禁止原地修改、重命名或把旧版文件覆盖成新版。

## 发布目录树

```text
G:/FutsalMOT_Dataset/synthetic_research_v1/
  README.md
  dataset_manifest.json
  schemas/v1/
    dataset_manifest.schema.json
    unit_manifest.schema.json
    frame_annotation.schema.json
    trajectory_record.schema.json
  synthetic/
    train/
      match_seed_<8位小写hex>/
        unit_manifest.json
        trajectory.jsonl
        provenance/
          generation_config.json
          source_code.json
          run_receipt.json
          qc_report.json
        cameras/
          <camera_id>/
            camera.json
            seqinfo.ini
            img1/000001.png ...  # 一基frame编号
            annotations.jsonl     # 一行一帧；rich canonical source
            mask/000001.png ...   # instance id mask
            number_mask/000001.png ... #号码像素mask
            gt/gt.txt             #从annotations派生
            labels/det/000001.txt #如冻结启用，由annotations派生
            labels/pose/000001.txt #如冻结启用，由annotations派生
        checksums.sha256
    dev/
      match_seed_<8位小写hex>/... #完全相同结构
  root_checksums.sha256
```

根目录不收真实下载视频；不放模型权重、临时EXR、调试PNG、notebook缓存、训练checkpoint或未登记文件。若冻结协议决定不开启某项派生产物，则它在`dataset_manifest.artifacts`中明确设为`false`，对应目录全体unit都不存在；不能部分unit自行增减。v1建议Pose和号码mask作为required task outputs；若号码生产未达到门槛，不得发布v1。

## 命名与索引

- `match_seed_<seed>`的seed固定8位小写hex。其内容必须是一场完整模拟episode的整段连续轨迹和它的所有相机输出；同seed所有输出进同一train/dev split。
- `camera_id`取manifest中的固定字典。v1开始后不可通过自由文本新增camera目录。
- 帧文件一基编号六位零填充：`000001.png`起。`annotations.jsonl.frame_index=1`对应`trajectory.jsonl.step=0`和首个Sequence source step；统一写`source_step`和`time_seconds`，禁止0/1基索引混用。
- 空帧不删文件：RGB、instance mask、number mask和annotations仍各有对应记录。不可见对象保留在完整身份/可见性字段内，是否进入MOT/YOLO由固定导出规则决定。
- unit内不放零散 clip 目录。若底层逻辑有多个时间窗，它们记录为同一unit的segments清单，且仍继承同一个split；各camera的frame/time映射要显式。

## 顶层清单必填信息

`dataset_manifest.json`要求：

- `schema`、`schema_version`、`dataset_id`、不可变`dataset_version`、创建时间和状态(`staging/released`)。
- `units[]`：精确列出unit_id、split、seed、frame_count、camera_ids、expected artifacts、unit_manifest相对路径和SHA256。
- `generation_protocol`：配置hash、代码bundle/hash、UE版本、trajectory source version、seed derivation policy、图像分辨率/FPS、相机profile及Pose schema ID。
- `artifacts`：instance_mask、number_mask、pose、MOT、YOLO labels布尔开关；整个版本唯一，不可逐unit临时开关。
- `identity_contract`：entity/track映射表、instance mask IDs、jersey number分配策略、号码mask IDs。
- `split_policy`：synthetic train/dev unit ID全集；不可让生成器按帧/相机重新拆分。
- `expected_counts`、存储上限、哈希算法与完整根fingerprint。

`unit_manifest.json`包含dataset_id/version、unit_id、split、seed、source trajectory UID、frame数/时间范围、所有camera ID与相对路径、每实体稳定`entity_id/track_id/team/role/jersey_number`、生成attempt/config hash、收据与该unit实际文件清单。

## 每camera契约

### `camera.json`

唯一记录camera_id、UE actor/sequence、图像宽高、intrinsics、world position/rotation、FOV/焦距、单位、逐帧或固定相机状态引用。值来自UE实际读回，不从task计划复制后冒充观测。若相机移动，每帧实际transform写状态JSONL；静态camera也保存实际状态和来源收据。

### `annotations.jsonl`

每行严格为一个frame对象，固定字段至少含：`dataset_id`, `dataset_version`, `unit_id`, `camera_id`, `frame_index`, `source_step`, `time_seconds`, `objects[]`。object至少含稳定entity/track ID、类别、team、role、goalkeeper、可见bbox、bbox_source、visible_pixel_count、truncated、visibility、instance_mask_id、jersey_number、号码可见/可读/朝向/number_bbox、固定Pose schema下的keypoint数组和每点visibility。单位、bbox边界约定、visibility取值和unknown语义由`frame_annotation.schema.json`枚举固定。

关键点数量与顺序是版本级常量，由manifest的`pose_schema_id`引用随仓库发布的spec。当前需在生成前决定采用生产级12点契约还是独立验证通过的其他schema；不得把历史文件名COCO17当作17点证据。unknown visibility必须明确写unknown/null，不能推测为visible。

### 图像与mask

- RGB统一为无损PNG、RGB三通道、8-bit；尺寸在manifest逐camera固定。禁止混入JPEG、RGBA、不同色彩空间或不同维度。
- instance `mask/*.png`统一为单通道8-bit PNG，0背景，1..10对应固定球员ID；像素与同帧RGB严格对齐。
- `number_mask/*.png`统一为单通道8-bit PNG，0背景，32..41对应固定球员号码ID；像素与同帧RGB严格对齐。不可读/被遮挡时该人像素为0，但annotation仍保留号码值与可见状态。
- `gt/gt.txt`遵循MOTChallenge九列、一基帧号，完全由可见bbox规则从canonical annotations派生。`seqinfo.ini`的frameRate、seqLength、imWidth/imHeight与真实目录完全相同。
- YOLO labels只允许固定converter重建，使用统一norm坐标、类别字典和pose point schema。它们是派生输出，不是权威数据源。

## 下载的真实数据如何接入

真实数据source目录只读，不放进上述synthetic root。每个比赛建立一个manifest-only适配记录：`source_id`, upstream match/video ID, 原始root绝对路径/locator, upstream文件hash或发行版fingerprint, license/access note, original split, camera/sequence IDs, native schema和转换规则。研究split清单仅存本项目；真实train/dev/test逐比赛列UID。只在许可允许的工作目录生成规范派生labels，不改上游格式/名字。

`SoccerNet-Tracking`等数据源自己的一个match可能由多个clip构成；adapter按match UID把所有片段归到同一逻辑比赛单元，保留原clip ID与时间范围。不得将clips/摄像头拆开后交叉分split，也不得把同比赛重复剪辑计成不同比赛。

## 结构清点算法

正式validator必须从`dataset_manifest`的unit/camera/frame数和artifact开关计算唯一允许的相对路径集合；不能用实际文件列表反向“发现”新字段/新目录。对比磁盘树时既缺一不可也不许多一项。逐个JSONL记录以对应schema验证，并做跨文件语义校验：十个entity/track/mask/号码ID一一映射、号码跨帧稳定、camera帧集合相同、frame_index/source_step/time单调对应、bbox与mask像素包围框一致、关键点顺序与schema相同、所有引用路径位于本unit根内、哈希和计数与root manifest一致。

## 生成Agent强制行为

1. 生成前只读取冻结的schema、unit manifest模板和精确attempt文件；拒绝自行造路径或字段。
2. 写入attempt专属新目录；目标路径存在即fail，不覆盖、不“整理”已发布目录。
3. 不确定字段、schema不兼容、额外/缺失文件、frame-count/hash不符时立即停止；不得按自然语言猜测结构并继续训练。
4. 只有validator全部通过且manifest fingerprint生成后，才可由单一publish步骤把整个unit原子发布；发布清单之外的文件不得迁入。
5. 版本更新必须改schema version或dataset version、更新迁移/兼容声明；旧dataset只读。

## 首次运行前必须补齐

`pose_schema_id`及spec；camera_id/每camera输出尺寸与FPS；精确seed和帧数；train/dev unit清单；instance/number mask dtype和ID映射；required artifacts开关；bbox/visibility规则；磁盘上限；QC阈值；schema文件和SHA。任意一项未冻结，则Gate 1不通过。


