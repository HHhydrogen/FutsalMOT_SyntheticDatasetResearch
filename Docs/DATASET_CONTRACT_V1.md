# 新合成数据集结构与Schema契约 v1（精简版）

状态：结构冻结提案。机器可读schema位于`../schemas/v1/`，Pose采用`futsal_body12_v1`。这是设计文件，不是已经接入UE导出器/生产validator的实现。首次生成前，实操对话需校验schema、填入确切seed/帧数/camera/resolution/FPS与质量阈值，计算schema SHA并冻结；此后改变结构必须升schema/dataset版本。

## 原则

- 一个seed代表一场完整模拟比赛，是不可拆分的输出单元；它的所有camera与帧在同一个split。
- 数据集根目录已经是合成数据集，因此不再增加`synthetic/`父目录。
- 每个seed只保留一个`seed_manifest.json`管理该seed元信息；不再重复拆`group.json`、`clip.json`、`unit_manifest.json`和`provenance/`。
- 每个camera只保留camera元信息、RGB、权威逐帧标注和两类mask。MOT/YOLO标签是派生产物，在训练适配目录按需生成，不混入权威数据集。
- JSON对象闭合（`additionalProperties:false`）；路径白名单来自根manifest。多字段不符、未知字段、额外文件、缺失文件均失败关闭。
- 发布版本只读。改任何RGB/标注/路径/seed都必须产生新版本和新fingerprint。

## 最简目录树

```text
G:/FutsalMOT_Dataset/synthetic_research_v1/
  README.md
  dataset_manifest.json       # 整个数据集版本、split成员、数量、schema/代码绑定
  checksums.sha256            # 一个全局文件，覆盖所有seed内容，不包含自身
  schemas/v1/                 # 很小的结构定义，不随seed复制
    dataset_manifest.schema.json
    seed_manifest.schema.json
    frame_annotation.schema.json
    trajectory_record.schema.json
    camera_state.schema.json
    camera_metadata.schema.json
    pose_schema_futsal_body12_v1.json
  train/
    seed_0000002a/
      seed_manifest.json      #该seed的唯一元数据/生成参数/角色和号码表
      trajectory.jsonl       #共享世界轨迹，一行一源帧
      C1/
        camera.json           #相机实际参数/标定
        img1/000001.png       #RGB，一基六位帧号
        annotations.jsonl    #一行一帧，唯一权威检测/ID/号码/Pose标注
        mask/000001.png       #实例mask
        number_mask/000001.png #号码实例mask
      C2/...
  dev/
    seed_000000b4/...
```

每个seed目录直接是一个完整episode；`C1`、`C2`等目录名只允许来自根manifest冻结的camera ID列表。不同seed内camera目录必须同名、同schema。`frame_index=1`对应首张`img1/000001.png`和`trajectory.jsonl`第一行`step=0`；不使用零基图片文件名。

## 文件职责

- `dataset_manifest.json`：整个数据集的唯一目录索引和split真值。列出确切seed ID、train/dev、每seed帧数、camera IDs、schema版本、生成协议和代码SHA、任务产物开关、预期总数与存储上限。
- `seed_manifest.json`：把原来分散在group/clip/provenance的字段合并。记录seed/root_seed、simulation trajectory UID、episode帧数/FPS、train/dev、camera IDs、十名球员的稳定entity/track/team/role/号码/mask ID映射、完整生成配置SHA、代码SHA和运行收据摘要。
- `trajectory.jsonl`：每个源step保存10名球员的world位置/速度和身份；坐标米制。轨迹只存一份，不在camera下复制。
- `camera.json`：该camera的UE actor/profile ID、分辨率、intrinsics和profile SHA。实际transform放在对应`annotations.jsonl.camera_state`，不另建状态目录。
- `img1/`：无损RGB PNG。固定8-bit、3通道；图像尺寸每camera由根manifest定义。
- `annotations.jsonl`：一行一帧，保存frame索引、时间、实际camera state和十个player对象。每个对象含稳定ID、team/role、jersey number、可见bbox/截断/visibility、号码可见/可读/朝向/号码bbox及固定Pose schema的12个点。
- `mask/`：单通道8-bit PNG，0为背景、1..10固定映射十名球员；与RGB像素同尺寸、同帧。
- `number_mask/`：单通道8-bit PNG，0为背景、32..41固定映射十名球员的号码印花像素；号码不可见时像素为0，annotation仍保留号码值及不可见原因。
- `checksums.sha256`：根目录唯一完整性清单，列出按相对路径排序的所有train/dev内容（seed manifest、trajectory、camera metadata、annotations、RGB和mask）的SHA256；不包含自身和`dataset_manifest.json`。根manifest中的`root_fingerprint`按“去掉root_fingerprint字段后的canonical manifest + checksums.sha256原始字节”计算，不形成循环hash。
- `schemas/v1/`：全数据版本共享的schema，不在每seed重复。结构SHA登记于dataset manifest。`checksums.sha256`只覆盖seed目录内容；schema SHA由根manifest逐项绑定。

## 权威标注与派生标签

`annotations.jsonl`是bbox、ID、jersey和Pose唯一权威源。训练adapter在独立workspace中按需导出YOLO txt、MOT `gt.txt`、`seqinfo.ini`；导出不写回G盘正式数据。不在新数据集保留重复标注格式，避免两个GT彼此漂移。实例和号码mask是原始像素监督，不由bbox标签替代。

每个帧记录严格含：`schema`, `schema_version`, `dataset_id`, `dataset_version`, `seed_id`, `camera_id`, `frame_index`, `source_step`, `time_seconds`, `pose_schema_id`, `camera_state`, `objects`。每帧十个人都保留，包括当帧不可见者；不可见bbox为null、pixel count为0、visibility按枚举填写，不能静默删track。

关键点schema固定为`futsal_body12_v1`，顺序是：left_shoulder、right_shoulder、left_elbow、right_elbow、left_wrist、right_wrist、left_hip、right_hip、left_knee、right_knee、left_ankle、right_ankle。该选择沿用旧工程的生产级12肢体点契约；历史文件名包含“coco17”不改变实际点数。visibility为`0=unknown/invalid`、`1=occluded`、`2=visible`。未知不得推断为可见。若研究需要脸部/头顶17点，必须先单独验收并建立新schema，不可在v1中暗改。

bbox统一为像素半开XYXY `[x1,y1,x2,y2]`，来源为instance mask的tight box；完全不可见为null。mask ID、号码mask ID和身份映射在seed manifest中逐一对应，跨帧不变。号码可见性字段是“像素中能否观察”的标签，不宣称OCR读数准确。

## 真实数据对接

下载数据保留原始位置、原目录和原许可，不拷贝进这个G盘合成根目录。模型阶段维护一个轻量的真实比赛split manifest：`source_id + upstream match/video ID + original clip IDs + source path + licence/reference + native format + adapter version + split`。拆成clip的真实集按完整比赛聚合判断split；同场片段不能跨train/dev/test。训练时由适配器读取原始数据并转成统一训练输入，不改源文件。

## 强制校验和发布

Validator从`dataset_manifest`计算唯一允许的文件树：按seed、冻结camera IDs、frame count及mask/image开关预期生成路径集合，检查磁盘树必须完全相等。不得从现有文件反推并接纳新的camera/字段。还需核验：每seed十个ID映射唯一且号码不重复、各camera帧完整且同步、annotation每帧十人、frame/source_step/time映射正确、bbox等于mask像素tight box、Pose点数与顺序正确、号码/track映射跨帧稳定、所有文件SHA与清单相符。

生成器只写新attempt路径，目标存在即拒绝；validator全通过后才发布整个seed。根manifest只允许冻结清单中的seed。任一错误最多进行一次定位修订；仍失败就停止并记录，不猜测格式、不追加样本、不覆盖已发布版本。


