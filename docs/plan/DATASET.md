# 数据与整理方案

更新：2026-10-31。本文件合并真实数据配置、合成成品需求与结构整理规则。原地迭代，不保留多轮方案或另建schema提案。实际生成规则以生产工程活动冻结协议为准，本文件不修改它。

## 当前事实与入口

真实a04源内交付和本轮real结构整理均已闭合。本轮核验目录/引用、原划分/UID保持、硬链接与GT/元数据及两来源代表图读回，不重复全量媒体hash、解码或像素验收。

| 面板 | 比赛train/val/test | 完整clip | 原生帧 | 检测图 | 仅train检测图 |
| --- | --- | ---: | ---: | ---: | ---: |
| SoccerNet | 3/1/2 | 29 | 21750 | 4350 | 2100 |
| SportsMOT足球 | 6/1/2 | 29 | 18974 | 3802 | 2933 |

现行浏览与后续读取入口为[SoccerNet dataset.json](G:/FutsalMOT_Dataset/real/soccernet/dataset.json)与[SportsMOT dataset.json](G:/FutsalMOT_Dataset/real/sportsmot/dataset.json)，每来源一个自包含索引，sequences/<原clip>目录中图像、GT与来源元数据完整可定位。各29clip，共40724图像、8152检测样本；40869个同卷硬链接包含145份原GT/元数据，没有复制媒体内容。

旧G sources与manifest及D历史原件保留作来源证据，新消费者以以上canonical入口为准，不继续拼旧分散路径。一次[real迁移与核验收据](G:/FutsalMOT_Dataset/real/RELOCATION.json)记录原路径、目标和验证层级。旧a04的plan_documents是冻结时历史引用，不改写；研究方案仍在本仓库docs/plan。

新dataset.json保留比赛/观察史、原UID与ID namespace、目标/ignore语义和精确detection_frames，不重复每帧绝对路径对象；两个索引共256843字节。原始输入均按只读使用，硬链接共享底层文件，适配器不能原地改图或GT。

合成生产进度和最终发布只读D:/projects/FutsalMOT_UEDataset/TASK_STATUS_AND_NEXT_STEPS.md及最终manifest，本方案不复制易过期的attempt路径和进度数字。本轮未重组任何合成目录；不把已有seed目录数当成全版已验收，也不恢复旧固定seed/130机位/300帧上限规划。

## 整理边界

real已解决同clip媒体/GT散落于existing_original_v1、official_a04_*和source_metadata_v1的问题：按原逐帧映射合并浏览目录，不用旧目录整体搬运假定完整。被a04最终dataset排除或complete=false的候选不进入新面板，即使旧候选行selected=true也不补入配额。

合成的seed内部已经符合约定：seed_manifest.json、trajectory.jsonl、person_keypoints_coco.json，加C5/C1/C2/C3/C4及实际局部机位；每机位为camera.json、annotations.jsonl、img1、mask、number_mask。需要整理的是外层多attempt、_staging、test与最终发布入口，不是再造seed内部层级。

## 已审核的浏览结构

用户已审核以下结构。real部分已按精确清单和两来源小样实施并通过核验；synthetic仍只作为生产闭合后的目标，当前没有新建或迁移其目录。不能在生产时改名或搬走工作目录。

```text
G:/FutsalMOT_Dataset/
  real/
    soccernet/
      dataset.json              # 一个入口：身份、split、序列与任务视图引用
      sequences/SNMOT-xxx/
        img1/                   # 每个完整clip在同一目录可读
        gt/gt.txt
        seqinfo.ini
        gameinfo.ini            # 来源提供时保留
    sportsmot/
      dataset.json
      sequences/<原clip_id>/    # img1、gt、来源元数据；不虚构gameinfo
  synthetic/
    dataset.json                # 来源版本、实际seed/机位/帧数、标注说明
    train/seed_<原8位hex>/      # 原三根文件与原机位子树保持
    dev/seed_<原8位hex>/
```

real不再复制train/val/test三套媒体，split由索引表达；synthetic保留已约定seed级train/dev物理结构。同一帧可服务多任务，不另存一套检测图和一套分割图。完整逐帧引用只在必要的清单中保存一次；任务视图以clip/帧号/原UID引用，不重复整段frame对象。训练专用YOLO标签和可再生缓存按需创建，不能混入原生数据当新真值。

不同数据集的原标注格式不强制变成一个巨大schema：真实MOT保留官方GT，合成保留现行JSONL/整数mask/COCO17。顶层索引只统一入口和引用规则。具体索引字段与迁移表需在首次整理前明确，不能凭本树擅改冻结GT。

## 整理步骤与停止Gate

1. real与synthetic分别确认无写入者。用户已授权real先做，本轮已确认其下载/解压闭合；synthetic必须等全版验收、渲染/导出/回写消费者闭合后再做，不能用real通过替代合成准入。
2. 固定现有dataset、来源、split、原UID与路径表；确定以上目标结构和容量峰值。整理只改变位置/索引，不能偷偷重新抽帧、改比赛划分或删不利场景。
3. real两来源各一clip的同文件身份、GT/元数据与代表图读回已通过，不重复做。未来synthetic取一个已通过seed做结构/引用小样；不全量重新解码/hash媒体，不重新证明生成像素质量。
4. 一次全量整理，以精确清单登记旧→新路径、数量/字节、来源指纹及校验层级；原收据不回写，不伪造旧路径。版本或新路径必须唯一，禁止覆盖不同成品。
5. 新入口、引用闭合、split/UID不变及小样可读通过后交付。仅在确认无人使用并满足既定生命周期时清理精确列出的临时媒体，保留必要原始证据，不再保留供人浏览的多套目录。

Gate：real本轮结构验收PASS后已停止，未删除旧源或开训。未来synthetic也要先确认结构/迁移清单，再做全量，独立验收后结束本整理子任务并移交已授权的后续准备流程；不能跳过EXPERIMENTS的实际训练Gate或自动转换所有任务标签。初检加一次针对修订仍不通过则FAIL_STOP；真实容量/身份/文件缺失则给具体BLOCKED，不用重复归档或重跑源代替解决问题。

## 划分与真实标签边界

a04已冻结的3/1/2与6/1/2及其原UID保持。本轮结构整理不再重算N或重抽test；只有未来新增数据版本才需重新讨论划分。源内同比赛的半场/clip/上传副本同split，观察史不清零。

当前跨库隔离未认证，因此各来源只允许独立实验，不混合R、不借另一来源val或训练过的辅助/ReID模型，不称跨库独立外部测试。已确认跨库重叠如需处理，另定版本，不原地改变已交付清单。

真实检测/MOT采用来源规定的目标人物和框语义，记录ignore局限，不凭空补画遮挡人体或裁判框。MOT保留完整原FPS、序列边界和ID命名空间；检测视图不拼回连续GT。

分割、Pose、号码的真实监督仍未就绪，不阻止检测/MOT。旧1000分割图、8000Pose人物帧、2500号码轨迹只是待重定的工作量参考，未取得便不能当作训练配额。号码已核小标签数不等于拥有裁剪媒体、逐帧可读性或比赛映射。WorldPose的SMPL投影不等于人工COCO17；SAM未经独立人工核验不当真值。

合成COCO17按实际17点与0/1/2操作定义使用，旧body12/unknown不转换。号码readable未认证不能全当不可读负例；号码像素非零不等于人类可读。多机位及所有派生窗口继承seed split，不能把同运动不同视角跨split。

## 存储纪律

真实/合成媒体和原GT在G；研究代码、隔离环境、公开基座权重、训练日志/检查点在研究项目D盘按需创建的目录。UE临时SSD工作集由生产工程管理，不因本方案另造副本。

真实准备沿用既有50GiB累计新增G峰值预算，包括先前归集与临时包；不是每次任务再加50GiB。正式媒体重组前需按当时账本另核实际空间、硬链接去重和生产预留，不把逻辑文件大小当物理重复占用。训练产物预算另定，不借用真实或生成媒体额度。
