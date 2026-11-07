# 实验日志

唯一累计实验日志，存放在docs/exp。按轮次追加，已有事实、失败和负结果不覆盖；同系列汇总可随新增结果更新并注明依据。此日志记录实际执行与分析，不代替或绕过已封存的实验/论文方案。

## 记录规则

每轮完成、失败、中断或取消后及时追加，至少包括：

- 编号、日期、系列、任务类型与状态：准备/诊断/预算校准/正式训练/推理评价分开。
- 目的与对照；实际训练的模型、公开初始化及可训练层；未训练明确写“无”，不得把测试张量当模型实验。
- 数据来源/版本/划分/实际数量、UID清单与观察史引用；不把待生产数量当实际输入。
- 实际参数：软件/代码绑定、输入、增强、batch/累积、精度、优化器/LR/损失、seed、每阶段与累计R/S曝光、更新终点、阈值、评价范围和末步检查点；未知/未执行明确标记，不用计划值冒充实测。
- 执行结果：完成更新、耗时/资源、loss和完整指标、失败原因及证据路径；评价先报绝对值，再给同系列配对差及适当区间。
- 同系列比较与结论：哪些对照已完成、哪些仍缺失、结论允许覆盖的任务/场景和局限；不挑best、不删低分、不把单次波动写成贡献。
- 后续动作与Gate：按既有方案继续、具体阻塞或需要用户讨论的方案变更；未经允许不改封存方案。

正式/预算校准/推理评价的逐步原始日志、预测、完整检查点、优化器/RNG等仍在实际run中完整保留，本MD只做可读索引与分析，不能替代原始证据。非主线测试/安装验证的临时日志、诊断权重/报告，在会话展示后删除；此处仅保留简短事实摘要。测试源码、依赖锁、公开基座和必要来源清单保留。

## 方案封存（2026-11-01用户指令）

以下现行文件原地保留，不建立archive/版本副本。自动化、heartbeat及子代理禁止编辑、格式化、改时间戳或更新摘要；只有用户在直接对话讨论中明确允许修改后，才执行获准范围并重新登记摘要。不能把新实验设计伪装成日志/配置来绕过封存。

| 封存文件 | SHA256基线 |
| --- | --- |
| docs/plan/EXPERIMENTS.md | 613495b089280e72b2ea503ec50d0c1ea97b87cc8dc53380e4e3e868cdec5ac2 |
| docs/plan/PAPER.md | d48cb4160a513773cde0e910fe1c45c7eb5a63bf058871eafd38e9ad27b71c17 |

自动流程可在既有授权与方案范围内推进准备、实现、执行和记录。若需要改变研究主张、实验策略/范围、评价定义或资源/证据规则，先在本日志记录问题/候选及原因，待直接对话讨论并获允许；不得自行回写上述方案或以配置变更偷换方案。发现摘要变化时先核实用户授权，不自动覆盖恢复用户文件。

## 当前系列总览

| 系列 | 已执行 | 模型训练/真实评价 | 目前可得结论 |
| --- | --- | --- | --- |
| PREP 环境与实现准备 | 固定依赖、BBox接口、完整检查点/RNG、布局dry-run、ignore criterion与给定框AP helper | 0轮真实模型训练；0轮真实网络推理/评价 | 工程组件经CPU夹具核验；尚不能证明合成数据迁移贡献 |
| DET-CAL 检测预算校准 | 未执行 | 未执行 | 无成绩或预算选择结论 |
| DET 正式检测与尺寸对照 | 未执行 | 未执行 | 无R/S→R或多尺寸效果结论 |
| MOT 检测到追踪 | 未执行 | 未执行 | 无HOTA/AssA/IDF1结果 |
| Pose / Seg / 号码 | 仅规划 | 未执行，真实标注仍缺 | 不能主张这些任务的真实迁移贡献 |

旧训练仍作为预实验保留在原工程/研究证据中，不复制或改写为本轮训练。本日志不补造旧轮次缺失数据。

## DATA-001：合成canonical整理计划通过，正式方法定义待直接确认

状态：真实数据整理阶段进行中，尚未建立目标或完成链接。用户直接要求渲染后按计划推进；本轮不训练、不改封存方案。

- 源G5、根PASS、128机位/130 owned退出及无生产消费者已复核；原发布后cleanup FAIL_STOP与原证据保留，交付data_affected=False，用户最新生产AGENTS明确成品有效且不得重复发布/恢复控制器。本研究不修复生产、不清旧staging链接。
- 精确只读plan_synthetic_layout(...writers_closed=True,enumerate_files=True)实际PLANNED_DRY_RUN，耗时5.4秒；完整plan紧凑排序JSON SHA=57e707673581c4865299c9c69ff590bce2bebe38fd86a9c6dc1f26746b766dc3。G:/FutsalMOT_Dataset/synthetic_research_v1/v4_coco17_20seed_long_active→G:/FutsalMOT_Dataset/synthetic；目标不存在、同卷，按原train/dev与seed相对路径一一硬链接90853个seed文件（547338754331逻辑bytes），媒体copy0；源root/schema/manifest/验证收据继续原位引用。一个train seed pilot通过后全量，不删原发布、不改seed内部。RELOCATION只保留一次精确表及身份/状态，dataset.json完整闭合前不得READY。
- 首pilot为train/seed_00007546（30022），7机位×153帧，原seed版本v3_coco17_20seed_clearance_max300f保留；其余19个seed原v4不改写。所有UID、split、trajectory/COCO17、号码visible/readable语义原值保留。coco_export_sha256是生产coco_export_complete.json收据SHA，不是COCO标注文件内容SHA；source_frames_sha也不冒充导出trajectory SHA。
- 结构预算：新增媒体0；目录项/精确JSON metadata上限512MiB、G最低余量100GiB，原逻辑509.749GiB不是新复制空间；只读G空余1208075640832字节。真实物理元数据开销仍在apply时观察，不提前称已实测。尚未创建目标；工具代码用临时夹具核验后，再refresh无写入者证据，pilot→其余全量→READY。
- 实际训练池元数据：合成train16seed/103机位=24149图，dev4seed/25机位=6030图；全部train RGB含合法空目标，不按困难程度或面积删图。SoccerNet train3比赛/14clip=2100图，val1比赛/5clip=750图；根独立从索引核train/val计数，未打开test GT/媒体。SoccerNet索引SHA=8dbf20aaa643c97f898905530ecc784468934f27d352e58c6d106257a0bde9e8。本数目不是训练已执行或正式协议已冻结。
- 原预算数学：S20 U_S=30187、482992S曝光（补齐12张）；R50 U_R=6563、105008R曝光，R100=13125、210000R曝光。独立s/seed17 R50/R100校准397份完整checkpoint。39次候选若R50合计960756更新/19308份完整checkpoint，若R100合计1196988更新/24024份；只为容量计数，实际完整state大小/时间/峰值未测，不签资源PASS或启动39次。

待用户直接讨论的四项定义（尚未正式启用，也未改EXPERIMENTS/PAPER）：

1. 训练ignore候选：仅来源已明确标注的非目标/ignore框，随target同几何变换；TAL正匹配优先，落在半开ignore框内的未匹配anchor中心只屏蔽全80类负BCE，box/DFL和正目标归一化保持原实现。不凭空为未标人群造ignore。
2. 主AP候选：现有custom_player_ignore_AP@300，正目标一对一优先，未匹配预测对ignore的IoA≥.5忽略；明确非COCO crowd、非官方排行榜，重叠ignore可能吸收重复预测。全部本论文对照必须同一口径，不能拿该绝对分直接与官方论文排行榜比较。正式采用须用户明确确认，不能因已有代码就默认主指标。
3. 增强删框候选：仿射/裁剪后仅零面积框退出监督，保留非零小框；不隐式采用原生w/h>2、aspect<100、面积保留率>.1删框规则。target/ignore同矩阵/裁剪/翻转，ignore不套小框过滤，记录所有派生排除原因，原GT不改。
4. resize候选：RGB直接960等比例LetterBox，float ratio、round目标尺寸、居中整数pad、INTER_LINEAR/pad114，不加原生BaseDataset先ceil resize，不声称两阶段像素相同。

这四项会改变监督/评价或预处理定义，来自用户封存规则的待决范围；如明确允许，只更新EXPERIMENTS对应条款及授权后的SHA基线，PAPER仍封存。已有S20、独立R50/R100选择、SGD/LR、有效16、增强概率、完整0/50/末步与终点EMA不重复索求许可。RGB/BGR转换、矩阵同步/日志接线是机械正确性，可按已授权范围实施。结构整理不依赖上述确认，继续独立完成。


## DATA-002：合成canonical真实整理与实际BBox池接线完成

状态：B1–B5 PASS，canonical READY；DATA-001的待执行目录步骤已实际闭合。该轮为数据准备，模型训练/校准/推理/评价与optimizer更新仍为0，不产生数据集性能结论。

- 源与范围：正式v4/G5/根PASS和128机位闭合/130 owned normal_exit已绑定；现场Win32_Process无UE、生产或其他研究writer。原a14发布后清理FAIL_STOP保留，不重启/修复生产或删旧staging。目标G:/FutsalMOT_Dataset/synthetic，原发布仍在synthetic_research_v1/v4_coco17_20seed_long_active。
- 工具验证：子代理实现layout_apply.py及小夹具，根独立全177项CPU检查38.164秒PASS/0跳过，CUDA未初始化且SGD.step拦截；TemporaryDirectory全部删除、无持久诊断报告。精确plan SHA仍为57e707673581c4865299c9c69ff590bce2bebe38fd86a9c6dc1f26746b766dc3。
- 真实pilot：train/seed_00007546，3230文件同fileID/size/mtime，原v3版本/trajectory UID/split保持；三根标注流式hash后只解码1张C5 4K RGB，媒体hash0。pilot实际PASS后由根放行其余19seed；异常即留FAIL_STOP部分目标、无自动覆盖/接续/清源。
- 全量：90853个同卷硬链接、547338754331逻辑bytes、媒体copy0，20seed/16train4dev、128机位、30179 RGB及各双mask。全文件与目录集合、fileID/size/mtime和原stat fingerprint均核对，原seed内部字节/COCO17/号码语义不改，root schemas按原路径与SHA引用。总wall322.172秒（含pilot审阅停点）。全卷空余差观察176099328字节，非排他物理成本归因；JSON原子写峰值受512MiB约束且G余量始终高于100GiB。
- 产物：dataset.json=READY，65292字节/SHA8aff1d86371211e16fd493f5dbc0ddb5a09820a12190dd58d6f6d86d7416787a；唯一RELOCATION.json=PASS，21075150字节/SHAcfa0fc5013e8c303c94fe9a45322ddb95ff6ea44e7e16e9814e1e20f726afe5f，完整一次旧新表与身份/源绑定；不另造迁移报告树。原发布source manifest/validation/G5的SHA保持前值。
- canonical读取机械接线：catalogues.py仅读有界metadata，核READY/RELOCATION完整PASS、原发布指纹及20份seed manifest；按索引seed→原camera_ids→升序帧，复用原派生UID规则。根独立6项新fixture测试1.749秒PASS/0跳过，含>4MiB receipt一次读取、512MiB拒绝、v3/路径/重复/状态及Dataset构造边界；临时夹具已清。实际train/dev池与RequestBBoxDataset构造均通过，未调用getitem、未读GT/图像/JSONL/COCO，根另核60文件源目标stat身份，未重复媒体解码。
- 实际池：S train16seed/103机位/24149图，S dev4seed/25机位/6030图。S有序pool SHA分别7e66aca666e0e46ff74ad24b5bedaaf8eb70acd70e51d24347cdb99bbb0b29db、c3f7e06bddcbfbd15c9d5b6e4c2aafb363b22f54639091276d6546188636e5ea。SoccerNet原索引train2100/val750，仅metadata；其原顺序pool SHA分别2c8142b605349f7061bc33fb36358976171662a11dd722b8aff6f90ada28b5d0、0cffac2b599794acd4357caf99dab893e2e6eb899cb4abe00b505343c379bbeb。这些是CycleSampler精确有序池摘要，不与DATA-001的排序集合摘要混淆，尚非正式run冻结。
- 配置只更新已验证事实：canonical路径、N_S/N_R、已有公式得到S20/R50/R100预算、源/池摘要及生产释放事实；protocol/data/resource/numeric/test Gate仍未通过，不以新能力布尔值启动。EXPERIMENTS/PAPER封存SHA未变，四项正式定义的对话审阅仍待用户答复，不因等待超时视作允许。真实训练循环/完整资源和数值预检仍待完成；Pose/Seg/号码缺真实GT分支维持BLOCKED。

## PROTOCOL-001：用户直接批准4项修订及正式开训

用户明确“按照你的建议，正式开始训练”，按刚才逐项推荐修改EXPERIMENTS对应BBox条款；PAPER原SHA不变。新EXPERIMENTS SHA=8e15dd43be6532f2b2b653b205d77bd7f3d754131ec90f4a07a199e0883a2ec4，原SHA=613495b089280e72b2ea503ec50d0c1ea97b87cc8dc53380e4e3e868cdec5ac2，AGENTS仅登记本次明确允许的新基线。采用role_policy_v2已知非球员背景、真实invalid/unknown才ignore且正匹配优先；标准COCO主AP100/自定义ignore AP300辅助；默认训练删框；直接960。其余预算/完整记录/停止与资源规则不改。pycocotools2.0.10已从官方固定SHA wheel安装，pip check PASS，锁新增实际依赖。先接实际代码与一次有界预检，再按实际完整状态与预测空间/耗时准入独立s/seed17 R50和R100；完整s15仍不默认启动，D容量问题未绕过。本条不是训练已开始，真实更新仍0。

## PRECHECK-001：真实GPU初检触发显存资源保护

CPU完整227项/47.668s PASS、0跳过，非主线随机CPU fixture实际6次SGD更新，正式0。GPU a01在已核生产释放后于08:04:53Z开始；大micro/FP32参考期间显存长期仅49–138MiB，Windows可能共享显存回退，违反1GiB余量。为保护资源，核PID30496/创建1793520290.3639088/精确命令后只终止该诊断；首次Stop-Process后PID尚在退出故未删锁，后续实际exit1且PID消失、显存释放后才退休owned锁。无UE/其它应用被操作、无正式run/模型参数被承接。诊断更新数因终止前未返回收据标unknown，不造0；正式更新0。a01 admission FAIL_STOP与源码身份永久保留，不伪造正常退出。唯一针对修订：在新a02独立尝试中把CUDA allocator分配限为min(总显存80%,实际空余减1.5GiB)，避免Windows共享显存慢回退；候选不足由OOM明确拒绝，不降输入/有效batch/更新/记录或放宽数值阈值；正式run继承实测同一limit。增加逐候选进度，无热改活动a01。若修订预检仍失败则具体BLOCKED，不继续无限尝试。

## RUN：R50_seed17 STARTED

UTC 2026-11-01T08:14:17.376133+00:00。用途：SoccerNet来源内R预算校准，不是合成贡献结果。公开yolo11s起点、seed17，train2100图/val750图；目标6563更新（名义50轮），实际已提交0更新/0R曝光/0S。输入960、有效batch16、micro8、fp32、workers2。SGD Nesterov/momentum.937/weight decay .0005（bias/norm0），LR终阶段peak.001→.00001、warmup100，clip10，既定HSV/仿射/翻转及默认删框、role_policy_v2；每50更新完整raw/EMA/optimizer/RNG/sampler、PT/每1000/末步val全部原始预测保留。

状态与实际结果：{"state": "initialized_from_public_not_yet_updated", "stage0_and_periodic_evaluation_retained": true}。证据：D:/projects/FutsalMOT_SyntheticDatasetResearch/runs/bbox_s_seed17_calibration_20261101_a02/R50_seed17/run.json、steps.jsonl、events.jsonl、checkpoints、evaluation。只用本次末步EMA做预算选择，不取best；R50/R100两独立结果未全闭合前不能选预算。

## PRECHECK-002：真实GPU修订预检通过，冻结有限R校准并启动

UTC 2026-11-01T08:18:46.392667+00:00登记。a02实际85.25秒PASS，38次诊断optimizer更新、24次诊断backward batch；仅真实train图用于数值/吞吐/无损输出验证，未读取test、未据val成绩选择精度。诊断权重/NPZ/临时目录已清，保留主线资源准入收据admission.json（SHA41bce6a72dff142c11dda6c6a7f6ce1bc345fcf70f193c64dafcd46e9283ee09）；a01 FAIL_STOP原证据保留。

- 数值与速度：固定原阈值loss相对误差≤.05、梯度cos≥.98、梯度norm相对误差≤.15；BF16多个批次不满足梯度一致性，依既定规则回退FP32，不降低阈值。FP32 micro16在受限allocator下OOM，安全候选中选micro8；有效batch16、workers2、pin_memory/persistent_workers、prefetch1，单线程worker。workers2/4稳态均约.453秒/更新，2的首批开销更低；有限短测不当全部训练耗时保证。
- 显存：CUDA allocator占比.8、上限13736765030B，micro8 FP32峰值allocated8391700480B/reserved8971616256B，当时显存free6722420736B；不以Windows共享显存回退扩充batch。后续各run同一数值/微批/资源规则。
- 完整恢复：实际更新后full raw/EMA/SGD/CPU与CUDA RNG/已提交采样游标/loader Generator及下一请求序列逐项exact_equal；worker吞吐另测2/4，恢复对照本身workers0。完整检查点更新前76284590B、更新后114192047B；不能只按公开19MB权重估训练证据容量。每图原始84×18900 FP32输出加过滤标志上界6650344B，实测NPZ无损读回。
- 容量与冻结：预检R50保守66272036893B（61.72GiB）；随后冻结执行公式以两份checkpoint作为瞬态预算，R50最终新增上界65426679162B（60.93GiB）/最多4小时，133份checkpoint与8次750图val；R100候选125385993726B（116.77GiB）/最多8小时，264份与15次val，每次派发前重新检查D剩余100GiB。两个预算独立公开起点，R100尚未准入；两者均完整有效才依末步主COCO AP择预算。15策略块仍容量BLOCKED，不改G路径/证据频率，不宣称校准已完成。
- 代码：根独立227项CPU检查47.668秒PASS/0跳过，6次非主线随机fixture CPU更新；allocator修订后7项针对检查PASS。冻结32份Python及requirements.lock到本组code，逐文件摘要与实际预检完全相同，未热改；runtime backend SHA6c290562944eb99a40f885e48fefd5124a18471334bf7c58402daf227aac0547，runner SHAed83809f417abd7ae08ab1bcc3f808732d4e74760d5d4e29ae49909d7fdf715e。实际入口为python -m futsalmot.runner，旧train/evaluate骨架仍拒绝，不冒称旧入口可训练。
- 启动：2026-11-01 16:14:07北京时间隐藏窗口启动唯一group/launch.py；wrapper29644、控制器13836/create_time1793520847.8217845，命令/PID/锁已核。R50 run已保留阶段0完整checkpoint并进行公开起点750图PT基线评价；登记时status=RUNNING、已提交更新0，正式参数更新以steps.jsonl为准，不能将启动或38次诊断算正式成绩。主线路径runs/bbox_s_seed17_calibration_20261101_a02/R50_seed17，run.json/steps/events/checkpoints/evaluation全部留存。
- 原生跟进：旧automation实际不存在，未误称已更新；新heartbeat bbox已ACTIVE、每20分钟，限本次R50/R100校准和证据收口；最多504轮且不晚于2026-11-07T23:55Z。正常中间态安静，不自动扩到s15/其它尺寸/test，不操作g或生产；完成或需用户决定且独立准备完成后停止该跟进。

本系列结论：真实网络、增强、完整状态与有界资源预检已经得到实际证据；尚无合成迁移贡献结果。当前仅真实R预算校准，R100及完整正式策略块的存储条件仍须各自准入。

## STORAGE-001：用户允许检查点降频与G盘完整留存

UTC 2026-11-01T08:26:40.113061+00:00。用户直接允许降低保存频率并在实验结束后迁至G专门区域；仅修改EXPERIMENTS对应保存/存储范围，旧SHA 8e15dd43be6532f2b2b653b205d77bd7f3d754131ec90f4a07a199e0883a2ec4 → c7d0d54b9993d1b975332d3f19625c8f670b704a056c1ebba47a42c5d9e288c6，AGENTS登记新基线，PAPER SHA保持d48cb4160a513773cde0e910fe1c45c7eb5a63bf058871eafd38e9ad27b71c17。未来独立run阶段0/每500更新/最终及停止失败边界保存完整状态，逐步loss/LR/梯度/UID与PT/每1000/末步val原始输出全部保留。当前实际a02/R50已提交100以上更新、PT val AP50:95=.38598711247659906/AP50=.796586201048111，仅公开起点真实验证，不是合成贡献；该run保持旧50直到结束，已产生检查点不删除。新G区域G:/FutsalMOT_Training/runs，闭合后精确SHA跨卷复制、目标读回/无写者复核、G发布后才释放D；旧路径绑定不改，D RELOCATION指向G。失败与负结果同样归集，不冒充完成。完整s15更新不变，checkpoint 8856→924（按R50），约941.83→98.27GiB粗估；但原始预测累计保守上界约2229.75GiB，未自动解除正式15/39累计资源Gate。新R100只在旧控制器退出且确认没有已执行R100后独立准入，不重复训练。

## STORAGE-002：归集工具与500频率执行/衔接验证完成

UTC 2026-11-01T08:39:49.558023+00:00。G:/FutsalMOT_Training已实际建立，一个README说明用途；尚无已迁移真实run，因为a02/R50仍活动（最近status提交1700更新）。子代理只实现run_archive.py及其测试，初检与一次针对修订后19项通过；根审阅精确路径/活动进程/清单/复制/目标SHA/源复核/同卷发布/逐文件释放代码，独立首批30项1.031秒PASS，后补真实生成冻结代码/引用/launch的机械衔接夹具并针对9项.172秒PASS，累计31项不同检查/0跳过，CUDA未初始化。所有TemporaryDirectory自动清理，无诊断权重、缓存helper或持久测试日志；只留测试源码。

- 迁移机制：archive_group(source, G:/FutsalMOT_Training/runs, research_root=...)只接受已终止且无相关GPU锁、PID/创建时间/命令绑定写者的直属组；Windows直属venv shim仅在完全同argv且身份可证时豁免，不能迁移调用者自己代码。单writer锁、4MiB流式拷贝、G最低100GiB；原文件/原失败status bytes保持，目标全SHA PASS后发布，再逐文件释放D。G唯一ARCHIVE_MANIFEST、D RELOCATION/status指向目标；已有PASS可续接未完成释放，未知文件/冲突/活动组/内容改变拒绝。
- 新执行规则：future runner按实际config checkpoint_every=500调度与计数，R50=15份、R100=28份，评价频率/日志/原始密集输出不变。R100新工作集上界95741738325B（89.17GiB），D与G各保留100GiB。当前R50仍原133份，已冻结代码不热改。新prepare_R100_after_archive只在旧R50完整6563、原控制器仅因D容量未派发R100且已归集PASS时，核原admission/3份R50小证据/manifest并准备新独立R100；已存在R100或人工取消/其它失败拒绝自动重复。新run引用旧有效R50终点比较预算，不重训R50、不承接其权重。
- 数值证据：根逐文件比较原32模块，只有未来runner.py改变，另增run_archive.py；其余31份原数据/增强/loss/模型/优化器/评价/精度组件摘要完全相同。既有admission原bytes/SHA不改，新的storage_revision侧车绑定当前源码和直接授权，只重做机械存储验证，不冒称做过第二次数值预检。当前活动group/code的32份摘要仍全部匹配原冻结。
- 源码摘要：runner.py=82bc78c251399382deecc6ca8df10aa63601f3b5eedae0fc34e7db1e037daf90；run_archive.py=5fb562b2db098cd1172581337b1db1183d6985d7973f71e129b38a2d6de0554d；test_runner_storage.py=85226b2f6f518f864f7fc78a5c37347e3f6f6c5b51d4a5446c0179a62df3d418。
- 原生heartbeat bbox已更新ACTIVE，保留20分钟/原硬截止，知道最新方案SHA和G区域，当前只观察活动组；闭合后先G归集、D释放，再最多独立R100。健康中间态安静，迁移活锁存在不重复启动；两校准和归集闭合即停止本有限跟进，不自动扩到15/39/test，不影响g/生产。

该轮结论仅为存储/执行衔接PASS，不是模型贡献结论；实际R50结果由主线run原日志记载。移至G尚未执行，待当前控制器/worker/writer真正退出，不能把准备通过当真实迁移完成。完整策略块仍受累计预测容量/实际S读取吞吐和作业时间Gate约束。

时间估计补记（2026-11-01T08:42:56.068178+00:00）：应用户询问列明BBox2次已启用校准、正式15+24=39次条件候选，共41含校准；不新增实际作业。当前已存日志1911次正式更新时，最近200步mean .48559秒/median .484秒，750图PT验证333.14秒。由已观测速度估s R50约1.5–2小时、R100约3–4小时（含既定验证，不含G归集）；S/混合及其它规格未实测I/O/吞吐，计算参照与可冻结预算分列在TODO，不能据此启动矩阵。现有硬上限4/8h、绝对截止和存储Gate不变。

## RUN：R50_seed17 COMPLETE

UTC 2026-11-01T09:51:25.938447+00:00。用途：SoccerNet来源内R预算校准，不是合成贡献结果。公开yolo11s起点、seed17，train2100图/val750图；目标6563更新（名义50轮），实际已提交6563更新/105008R曝光/0S。输入960、有效batch16、micro8、fp32、workers2。SGD Nesterov/momentum.937/weight decay .0005（bias/norm0），LR终阶段peak.001→.00001、warmup100，clip10，既定HSV/仿射/翻转及默认删框、role_policy_v2；每50更新完整raw/EMA/optimizer/RNG/sampler、PT/每1000/末步val全部原始预测保留。

状态与实际结果：{"tag": "step006563", "update": 6563, "main_metrics": {"AP50_95": 0.5165115374980012, "AP50": 0.941099523964852, "AP75": 0.5056873088257927, "AP_small": 0.33257477789442114, "AP_medium": 0.5057760772141966, "AP_large": 0.5835367728536506, "AR_max1": 0.05302695092518102, "AR_max10": 0.45075422365245377, "AR_max100": 0.6037510056315366, "AR_small": 0.38661417322834646, "AR_medium": 0.6047788697788699, "AR_large": 0.6348387096774194}, "raw_prediction_bytes": 4246207173, "images": 750, "wall_seconds": 320.07800000000134, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}。证据：D:/projects/FutsalMOT_SyntheticDatasetResearch/runs/bbox_s_seed17_calibration_20261101_a02/R50_seed17/run.json、steps.jsonl、events.jsonl、checkpoints、evaluation。只用本次末步EMA做预算选择，不取best；R50/R100两独立结果未全闭合前不能选预算。

## R50-001：末步结果与G盘归集移交

UTC 2026-11-01T09:55:41.653316+00:00。用户通知R50完成后，根独立核对完整6563条有序已提交更新（每次16图/optimizer成功、UID16个），累计105008 R曝光/0 S；阶段0/50倍数/末步共133份检查点size与原收据一致，PT/1000/2000/3000/4000/5000/6000/6563共8次750图验证、6000份原生NPZ均存在，末步summary与COMPLETE状态一致。没有额外推理、训练、读取test/GT或重哈媒体。原控制器13836及wrapper29644/其spawn worker均退出，research_gpu锁正常退休、旧R100目录不存在。

- 末步EMA原始主指标：AP50:95=0.5165115374980012、AP50=0.941099523964852、AP75=0.5056873088257927；小/中/大AP=.33257477789442114/.5057760772141966/.5835367728536506。公开起点AP50:95=.38598711247659906、AP50=.796586201048111，因此末步主AP提高13.05244250214021个百分点。只取规定末步，不拿中途更高验证点代替；该轮仅真实R训练，不证明合成贡献。R100未完成前不选择预算。
- 主线耗时：run STARTED UTC08:14:17至COMPLETE09:51:25，约1小时37分钟；包含PT/定期/末步验证，G归集另计。实际参数/完整结果已由driver写RUN COMPLETE，原记录不改。
- 组状态保留FAIL_STOP：旧控制器准备独立R100时D空余205563019264B不足其125385993726B新增预算+107374182400B预留，故仅派发前容量拒绝，不是R50训练失败；旧traceback原bytes留存。按用户已批准的新500频率和G留存规则作合法移交，不重放旧控制器或重复R50。
- 归集已经开始：仅将已闭合a02组迁至G:/FutsalMOT_Training/runs/bbox_s_seed17_calibration_20261101_a02，archive PID=28832/create_time=1793526878.645933、exec session81496，单一临时.archive_followup.json用于接续；源文件在G完整SHA读回/发布前保持。尚未声称归集完成，R100也尚未启动。

R50同系列验证轨迹（均为原始末点EMA观测，不据此挑best）：

| 更新 | AP50:95（%） | AP50（%） | 本次验证秒数 |
| ---: | ---: | ---: | ---: |
| 0 | 38.5987 | 79.6586 | 333.140 |
| 1000 | 53.3906 | 95.1889 | 311.578 |
| 2000 | 54.1082 | 95.0491 | 310.328 |
| 3000 | 53.2027 | 94.9085 | 304.813 |
| 4000 | 52.4677 | 94.2539 | 302.797 |
| 5000 | 51.9166 | 94.1170 | 303.609 |
| 6000 | 51.5125 | 94.0567 | 308.953 |
| 6563 | 51.6512 | 94.1100 | 320.078 |

这一轨迹表明本次真实训练在该验证面板上较公开起点有改善，后段验证AP较部分中间点低；原因尚未作因果诊断。预算仍严格待独立R100末步结果后按原规则选择，不新增early stopping/挑best或重复R50。val仅1场/5clip，不能将本次幅度外推为跨来源普遍收益或合成贡献。

## STORAGE-003：R50归集闭合、未来证据直接写G

UTC 2026-11-01T10:24:03.094150+00:00。用户明确避免搬运、检查点直接设G；仅获准存储条款原地更新，EXPERIMENTS SHA c7d0d54b9993d1b975332d3f19625c8f670b704a056c1ebba47a42c5d9e288c6 →46aa2fe83daf99bc0a2e1880f593eb36b2c11297b95f8eaaf479de19c4401446，AGENTS已登记；PAPER仍d48cb4160a513773cde0e910fe1c45c7eb5a63bf058871eafd38e9ad27b71c17。R50归集6213文件/49874030816B PASS，G唯一ARCHIVE_MANIFEST SHA69443abad2acd960145bf9ccb735dfce8f067ea5ce971f8a7f3841edb99102ba；D逐文件释放同量、只留RELOCATION/status，原FAIL_STOP与R50全部结果bytes保留，临时.archive_followup已精确删除。不再重复搬运/重哈本组。

- 未来R100：D控制D:/projects/FutsalMOT_SyntheticDatasetResearch/runs/bbox_s_seed17_calibration_20261101_a03_storage500，G实际输出G:/FutsalMOT_Training/runs/bbox_s_seed17_calibration_20261101_a03_storage500。检查点/steps/events/预测/评价/controller日志直接G；双端RUN_LOCATION同bytes严格绑定，无junction、不覆盖已有组。D代码/环境/公开基座不搬。开始前42份小控制/冻结源码共543060B同步G control；结束且worker/writer与GPU锁闭合后只补小控制并封闭唯一CONTROL_MANIFEST，不搬完整run或扫描全量模型/NPZ。
- 验证：根独立33项存储/衔接CPU检查3.136s及40项监控CPU/短时HTTP检查.875s全PASS/0跳过，TemporaryDirectory夹具自动清理，无诊断模型/报告留存。原32预检模块仅runner的存储/派发改变，其余31数值/模型/增强/损失/评价模块SHA完全相同，新增run_archive/output_storage；不重复GPU数值筛选或训练。实际34份源码/锁/新计划/旧admission与R50三个小终点证据冻结，sidecar记录存储授权；旧活动/归集源码均不改。
- 资源：R100独立公开初始化，seed17、2100R/750val、960、FP32 micro8/effective16/workers2、13125更新/210000R曝光目标；checkpoint0/每500/末步共28，15次val保留原生密集预测，完整新增上界95741738325B/89.17GiB；G完整新增+100GiB、D仅32MiB小控制+100GiB。准备时D237.9GiB/G1078.5GiB，容量PASS；max8h及绝对截止不变，不等于15/39块准入。
- 实际派发：已隐藏启动唯一launch，wrapper PID23604/create_time1793528642.976556，stdout/stderr直接G。启动不等于已有参数更新；阶段0/PT与首步由实际run/status/checkpoints/steps核实后补记。两个预算均只按独立末步EMA主COCO AP比较，100严格更高否则50，不取best、不读test、不自动扩展。

## RUN：R100_seed17 STARTED

UTC 2026-11-01T10:24:14.789437+00:00。用途：SoccerNet来源内R预算校准，不是合成贡献结果。公开yolo11s起点、seed17，train2100图/val750图；目标13125更新（名义100轮），实际已提交0更新/0R曝光/0S。输入960、有效batch16、micro8、fp32、workers2。SGD Nesterov/momentum.937/weight decay .0005（bias/norm0），LR终阶段peak.001→.00001、warmup100，clip10，既定HSV/仿射/翻转及默认删框、role_policy_v2；每500更新完整raw/EMA/optimizer/RNG/sampler、PT/每1000/末步val全部原始预测保留。

状态与实际结果：{"state": "initialized_from_public_not_yet_updated", "stage0_and_periodic_evaluation_retained": true}。证据：G:/FutsalMOT_Training/runs/bbox_s_seed17_calibration_20261101_a03_storage500/R100_seed17/run.json、steps.jsonl、events.jsonl、checkpoints、evaluation。只用本次末步EMA做预算选择，不取best；R50/R100两独立结果未全闭合前不能选预算。

实际启动补记（2026-11-01T10:27:08.860487+00:00）：真实controller31060/create_time1793528643.0162673，GPU锁精确绑定D控制组；G阶段0完整checkpoint76289902B已保存（原收据SHA04d420a55f94caad7ec0d4669ebc52810017eddda8b0c7fb80484e27516f4caf），run/status RUNNING/0，先PT评价，不将启动当更新。只读观察台已换v3并实测自动选择G R100，direct_g不标归档；只重启监控，训练未重启。原生bbox已更新直接G路径/新方案SHA、仍60分钟及原硬截止，R100结束不再全组归集。README/TODO/G入口事实已原地更新。

监控收口补记（2026-11-01T10:28:43.712444+00:00）：v4 root独立40项0.969s PASS/0跳过；实查原“日志暂不可读”只是0更新首次steps.jsonl尚未产生(WinError2)，stdout/stderr/checkpoints共享读取正常。仅初始missing按等待显示，真实读取错误仍保留。实际同URL健康version4、PID28156/create1793528904.072339，自动选中G R100/direct_g，phase={"kind": "validation", "label": "正在验证（验证期间训练步数保持不变）", "tag": "step000000_PT", "images": 400, "total": 750}、update=0，errors空；ETA明确直接G无结束整组搬运。只重启观察台，训练31060未动，34冻结模块仍一致，非主线小夹具已清，无临时权重/报告。

## RESUME-001：用户重启Codex后的R100显式恢复

UTC 2026-11-01T10:47:47.546308+00:00。用户报告重启Codex后训练退出并明确要求重启；只读确认旧controller31060/create1793528643.0162673及wrapper/worker均已退出，无UE/训练写者，stderr为空，原状态停RUNNING（不伪造正常退出）。最后1000条逐步记录严格连续并与1000完整检查点一致，累计16000R/0S；可信收据SHA17828b4582a5638c3bd907dbe874350327b5ece4cf9e30bfa394de82341014a4、114197359B，CPU只读加载raw/EMA/SGD/RNG含CUDA/采样及loader状态通过，34份冻结源码与双端绑定未变。没有未知更新/回滚/重训。

恢复G/recovery_r01引用原run/协议，保留旧group/run状态和原锁bytes，确认无写者后精确退休旧锁。原PT验证已完成；step001000验证尚未闭合，原预测全保留，既有冻结evaluate将另建step001000_retry1重新评价，不覆盖或合并原结果。完整新增按原89.17GiB加一次750图原生预测上界重新准入，D仍100GiB+32MiB、G100GiB余量。13125总更新/500检查点/数值/数据/增强/末步选择不变；恢复仅12125剩余更新，不从公开基座重训。八小时原绝对界限2026-11-01T18:24:06.023648+00:00保留，外层有界守护到界仅写受控STOP，不给恢复重置无限预算。独立恢复launcher不改原冻结训练代码，执行/日志继续原G组。直接恢复授权不扩展15/39/test。派发及实际恢复状态随后补记。

恢复派发核验（2026-11-01T10:49:05.283457+00:00）：recovery_r01/launch.py SHA830e1528f0cafd74566d2695f6bc1b1cc4105253099404d2b11fabd86b6b2a23；DETACHED_PROCESS+NEW_PROCESS_GROUP+BREAKAWAY_FROM_JOB隐藏派发wrapper11148/create1793530083.941226，实际controller33184/create1793530083.953157已持唯一GPU锁，实际Windows IsProcessInJob确认不属父进程作业。run/status已在完整raw/EMA/SGD/RNG/采样恢复后写为新PID、update1000；新step001000_retry1已创建，原partial原样保留，恢复不重复1000次更新。冻结34模块与EXPERIMENTS/PAPER均未改。原stdout/stderr只追加，stderr仍0，剩余12125更新，原8h绝对界限2026-11-01T18:24:06.023648Z不延长。

进程核验更正：上一段将DETACHED/BREAKAWAY启动成功写成“实际controller不属父作业”不准确。实际新核验wrapper11148及controller33184的IsProcessInJob均为true；启动标志已生效到CreateProcess成功，但未验证所有外层作业归属，因此不能保证再次重启Codex不会受影响。明细见recovery_r01/process_lifetime_verification.json。第1000步恢复与验证重试已成功运行，不为进程隔离验证再次停训练或更改冻结实现；以真实状态/后续更新为准。

恢复最终小收据（2026-11-01T10:50:53.916314+00:00）：恢复PID33184身份/锁/命令确认，run更新1000，stderr0；34冻结源码与两封存方案SHA保持。只读观察台已恢复PID24512，同原URL实际识别新训练身份/direct_g，phase={"kind": "validation", "label": "正在验证（验证期间训练步数保持不变）", "tag": "step001000", "images": 200, "total": 750}。原bbox跟进已更新本次显式恢复身份、partial retry规则、原8h界限和进程隔离核验限制，仍每60分钟/原全局截止；无重复训练或新矩阵。

## RUN：R100_seed17 COMPLETE

UTC 2026-11-01T14:00:54.604078+00:00。用途：SoccerNet来源内R预算校准，不是合成贡献结果。公开yolo11s起点、seed17，train2100图/val750图；目标13125更新（名义100轮），实际已提交13125更新/210000R曝光/0S。输入960、有效batch16、micro8、fp32、workers2。SGD Nesterov/momentum.937/weight decay .0005（bias/norm0），LR终阶段peak.001→.00001、warmup100，clip10，既定HSV/仿射/翻转及默认删框、role_policy_v2；每500更新完整raw/EMA/optimizer/RNG/sampler、PT/每1000/末步val全部原始预测保留。

状态与实际结果：{"tag": "step013125", "update": 13125, "main_metrics": {"AP50_95": 0.4957188790826453, "AP50": 0.9264702099429764, "AP75": 0.461668426939532, "AP_small": 0.3492485664667982, "AP_medium": 0.48508734055808056, "AP_large": 0.5577072589060098, "AR_max1": 0.051407884151246974, "AR_max10": 0.4401146419951729, "AR_max100": 0.5813757039420756, "AR_small": 0.3862204724409448, "AR_medium": 0.5813390663390664, "AR_large": 0.6149032258064516}, "raw_prediction_bytes": 4236809464, "images": 750, "wall_seconds": 358.5630000000019, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}。证据：G:/FutsalMOT_Training/runs/bbox_s_seed17_calibration_20261101_a03_storage500/R100_seed17/run.json、steps.jsonl、events.jsonl、checkpoints、evaluation。只用本次末步EMA做预算选择，不取best；R50/R100两独立结果未全闭合前不能选预算。


## BUDGET-001：R50/R100末步比较与有限校准闭合

UTC 2026-11-01T14:03:59.538774+00:00。用户通知R100已完成，要求比较并说明后续合成贡献实验。根只读复核R100于2026-11-01T14:00:54.508597Z（北京时间22:00:54）COMPLETE，13125更新/210000R曝光/0S；完整28份检查点大小与原收据一致（合计3159624867B），15次750图验证按0/1000倍数/13125闭合。step001000_retry1为成功验证，原step001000中断预测保留且不计成功次数。末步记录optimizer成功；未重跑训练/推理或读取test、权重/NPZ/媒体。恢复控制器33184、wrapper11148及原worker身份已退出，GPU锁已退休，无相关训练/UE写者，stderr0。R100仍是公开起点独立训练，经1000步完整状态恢复，未重复前1000更新。

| 指标 | R50末步EMA | R100末步EMA | R100减R50（百分点） |
| --- | ---: | ---: | ---: |
| 主COCO AP50:95（%） | 51.651154 | 49.571888 | -2.079266 |
| AP50（%） | 94.109952 | 92.647021 | -1.462931 |
| AP75（%） | 50.568731 | 46.166843 | -4.401888 |
| 小目标AP（%） | 33.257478 | 34.924857 | +1.667379 |
| 中目标AP（%） | 50.577608 | 48.508734 | -2.068874 |
| 大目标AP（%） | 58.353677 | 55.770726 | -2.582951 |

两组均YOLO11s公开起点、seed17、SoccerNet train2100/val750、960、FP32 micro8/有效16/workers2及既定SGD/增强/角色和评价。实际更新6563/13125、真实曝光105008/210000。R50端到端约1小时37分，R100约3小时37分；后者包含用户重启中断及一次验证重试，两组保存频率不同，时间只作运行成本描述，不能当严格吞吐比较。按预登记“100末步主AP严格更高才选100，否则50”，本任务选择E_R=50、U_R=6563；D/G budget_choice同bytes。小目标指标改善不改换选择依据，不取best、不追加轮次搜索。校准仅单seed/一个真实val比赛，不能断言100轮普遍更差或将下降因果归于过拟合；两组均未用合成数据，尚无合成贡献结果。校准权重不直接当正式三seed结果。

直接G输出已在最终位置，未搬运或全量hash。G control/CONTROL_MANIFEST.json=SEALED，46个小文件/549379B，逐文件SHA通过；manifest SHA b26a15c64f2899a8dedfbe6a9972e6efb47ead54ad75279b1aed5c40d4761d22。D小控制与G control一致；D/G status科学内容相同，分别落盘的updated_at_utc约差5ms，属于正常独立时间戳，未覆盖。R50旧归集和容量FAIL_STOP、R100中断/恢复全部保留。原生bbox有限跟进已删除，daily g未动，观察台仍可查历史。

下一阶段沿封存方案具体化：N_S=24149（16train seed/103机位），S20 U_S=30187/482992S曝光；R50为6563/105008R。正式R、S、S→R、R→R、R+S×seed17/29/43最多15次，未准入。建议先按小块执行核心R/S→R/R→R对照，再补齐S与R+S及其余seed，不根据涨分筛选保留方法；实际次序/作业绑定与资源仍需冻结。S→R对R控制真实标签和末阶段预算，R→R控制额外更新与图像数，R+S对顺序策略，S对直接迁移；不把同更新称同FLOPs，不把合成阶段20轮称已充分收敛。S→R从对应S末步raw接续并重建真实阶段优化器/EMA，不能从本次R50权重开始。正式分支尚需合成4K数据真实增强/吞吐/续接小样与完整存储、时限准入，不能因R校准完成直接认定通过。

当前D约237.89GiB、G约1012.75GiB空余；完整s15原始预测保守上界约2229.75GiB，仍不能整块准入。没有删原始输出/降低评价频率/改方案，本轮不启动新run。后续只按既有完整证据规则逐块核算；确需改存储或证据规则必须直接讨论获准。test继续封存，Pose/seg/号码缺真实GT的正式分支保持BLOCKED。EXPERIMENTS/PAPER SHA均未改。


## FORMAL：R_seed17 STARTED

UTC 2026-11-01T23:53:06.195324+00:00。目的：合成数据真实迁移五策略对照；公开YOLO11s独立起点、seed17，SoccerNet train2100/val750，synthetic train24149，策略R。实际已提交0/6563更新，R/S曝光0/0；960/FP32/micro8/有效16/workers2，SGD Nesterov momentum.937、WD.0005，clip10。第一阶段30187步峰LR.01，真实终阶段6563步峰LR.001，各自cosine到.01倍，阶段重建SGD/EMA并承接raw；独立R仅终阶段、S仅第一阶段。检查点0/每500/阶段末完整状态，PT/每1000/阶段末750图原生预测与全部负结果保留。不取best、不读test；本轮结果尚需与同seed/其余seed对照，不能单轮宣布贡献。

状态/原始结果：{"state": "public_initialized_zero_updates"}。证据：G:/FutsalMOT_Training/runs/bbox_s_soccernet_seed17_core_20261102_a01/R_seed17。


## FORMAL-PREP-001：正式R控制块启动与合成输入性能Gate

UTC 2026-11-01T23:55:09.001467+00:00。用户2026-11-02直接要求按原计划继续训练。新增formal.py/formal_preflight.py真实实现五策略、三流采样、分阶段raw承接/新SGD和EMA、全局与局部步数、完整恢复及直接G；原34个数值/数据/损失/增强/评价模块SHA无变化。根完整338项CPU检查最终PASS/0跳过/57.532s；首次338中1项因调用未设YOLO_OFFLINE被正确拒绝，按既定离线环境复验通过，未修改训练或放宽断言。新的loader边界夹具验证initial/midphase/newphase/iter后零更新恢复，不把initial误撤销一次gen推进。只读监控v5增加formal阶段/精确val计划ETA与有效batch兼容，43项PASS/0跳过/.965s；该非主线夹具均TemporaryDirectory清理，未留下玩具模型/报告。

实际S输入预检admission=PASS，115.765s，9种原机位4K/960训练输入、真实first-stage更新和全状态读回/恢复，raw/EMA/SGD/CPU与CUDA RNG/sampler/loadergen及下一loss/梯度全部精确一致；实际raw承接后新R优化器/EMA验证通过。固定FP32/micro8/有效16/workers2、offset cache7机位/3000帧。峰值CUDA allocated8392781824B/reserved8954839040B，余量6753878016B>1GiB。3份诊断完整权重304887308B已在会话展示且TemporaryDirectory精确删除；2个spawn临时helper本轮结束后精确删除。诊断原始preflight_steps.jsonl与warm日志/测量保持原bytes进入G正式输出组，不删失败或改旧收据。

首8更新窗口均值4.4751s仍含冷索引，为区分冷启动只再做一次固定64更新、最多300秒的持续诊断窗口，实际214.235s；最后16均值3.2070625s，累计GPU机械准备330s<600s。合计76诊断optimizer更新含恢复重放，只是准备，不能计入正式实验曝光或数据集成绩。按实测S20仅30187训练步约26.9h，尚不含真实微调和验证；S→R未准入当前块。已向用户提交约62.18GiB无损SSD LetterBox缓存选择，候选D/cache三文件、128GiB上限/保留D100GiB/最多4h、原像素/标注/UID/增强/曝光不变；封存方案写明当前不批量生成图像缓存，实际构建等待直接允许，仅代码/小夹具准备可继续。不自行延长为慢速S长run或改封存计划。

已冻结并真实启动有限正式两组（不取消其余五策略，只是逐块准入）：R_seed17=6563更新/105008R/0S、15checkpoint/8val/max4h；R_to_R_seed17=30187 R_extra+6563 R，总36750更新/588000R/0S、77checkpoint/40val/max16h。全由公开YOLO11s独立起点，SoccerNet2100train/750val、seed17/960/FP32/micro8/有效16/workers2、SGD/既有增强与COCO主评价，0/每500/阶段末完整状态，每步loss/LR/梯度/UID及全部val原生预测保留，不取best、不读test。整个块最多20h且仍受2026-11-07T23:55Z绝对界限。

完整新增保守369.2502245GiB（R65.47465、R→R303.77558），G另100GiB；D只32MiB小控制另100GiB。逐步2MiB/更新、预测JSON1MiB/图、native6650344B/图、检查点实测×1.1与一次失败partial/瞬态均独立预算，详见不可变storage_revision。D控制D:/projects/FutsalMOT_SyntheticDatasetResearch/runs/bbox_s_soccernet_seed17_core_20261102_a01，G输出G:/FutsalMOT_Training/runs/bbox_s_soccernet_seed17_core_20261102_a01；36个源码/依赖已冻结，job_table SHA b97371e2812779597db5bcb517ad14b10b191b8f9510ab75678bd04b71fefa68、initial control43文件。实际controller12244/create1793577174.1052206、wrapper26172/create1793577174.068056，launch SHA b47fddd9d1a7c16d02f25fa3cc5c9de4b03d648430366874141a73c9b1c11d29；23:52:54Z派发，run初始化RUNNING/0先公开起点评价，启动不等于已有更新。完全隔离Codex重启未验证，不编保证。

新原生跟进bbox-r-r-r每60分钟/原截止，有限两组及小control/owned退出闭合即删除；健康安静，失败/取消不自启，不自动扩S、其他seed/尺寸/test。观察台仍http://127.0.0.1:8765/，本次48h只读窗口。EXPERIMENTS/PAPER原SHA均保持，新的缓存代码不进入当前活动冻结包。


## CACHE-AUTH-001：用户批准无损SSD缓存

UTC 2026-11-01T23:57:28.414251+00:00。用户明确允许构建约62.2GiB D盘无损960输入缓存，全部24149S train、原像素/标注/UID/增强/曝光保持，128GiB上限/D余100GiB/4h，代码与一致性夹具正在完成，尚未冒称实际缓存READY。CPU缓存可与已冻结R/R→R并行但受物理4GiB/提交8GiB及有界2worker门禁；S→R仍须cache READY、输入数值/容量/GPU互斥与新run冻结后独立公开起点启动。当前活动private code不改，EXPERIMENTS/PAPER不改，仅登记最新直接执行许可。


## CACHE-001：已许可CPU缓存实际构建与未来cached S→R接线

UTC 2026-11-02T00:05:24.408024+00:00。用户明确批准后，独立无损输入缓存已于北京时间2026-11-02 08:00启动，writer9804/create1793577629.283057、wrapper14340/create1793577629.2464635；G核心组/cache_build.launch.py SHA9ed3963fde233fc24b69c794c782be1b939f7f6e3ef25e570becfde7e83fb6e3。独立37源码/依赖快照在G核心组/cache_builder_code，bindings SHAa03cfe775c78511ef909968c2400aed313245ea88b4c9479f1a020244f8afff2；活动GPU原36源码复核完全未变。CPU无CUDA初始化，2个spawn worker各1线程、父2线程、最多4个future、单writer，源读取与GPU独立，不中断/热改R/RR。

目标D/cache/synthetic_bbox_960_v1，先.partial；images.npy精确66767155328B（62.181759GiB）、元数据保守上限后完整64.244259GiB，硬上限128GiB，D保留100GiB，物理4GiB/commit8GiB、14400秒上限。当前实际1408/24149行已按原UID顺序写入，state BUILDING，尚未READY；D余约175.8GiB。原RGB/GT/source版本和split不动，缓存增强前960 RGB uint8与完整标注/geometry，随机增强保持在线原实现。全量逐行像素写回/metadata无损验证，五份预处理源码、原RGBstat、GT与元数据绑定；正式消费者只接精确manifest SHA/READY三文件。失败partial保留、不自动重建。日志/状态/派发在G核心组/cache_build及cache_build.*.log，实际缓存不作为临时测试删掉。

根独立缓存12项CPU检查4.961s PASS/0跳过，含真实2worker及像素/标签/geometry/不同增强seed bitwise、修改或损坏拒绝，TemporaryDirectory夹具已清。未来新run缓存接线和有界预检已完成，根28项formal系列11.531s PASS/0跳过：新formal.py SHA25e2eacd7ea4e4fc07c5981986692d0b3c3e1fc2e8c3db7d6e7cedfcd2d46e22，formal_cache_preflight.py SHA434593774c427e6753f3d3a7a3b7d703a7b7332bd6f8981365abad505fa88437。旧无cache协议与活动冻结版本完整相同；本轮仅验证CPU接线，尚未执行cached GPU预检或S训练。

用户许可下后续有限范围为seed17 R/R→R/cached S→R三核心组，不扩剩余seed/尺寸/S单独/R+S/test。原生bbox-r-r-r已更新完整后续：缓存READY/writer退出且当前R/RR两job COMPLETE、G小control SEALED/owned退出/GPU锁退休后，才调用preflight_cached_formal(...prior_admission=D核心组/admission,closed_training_group=D核心组,max_seconds=240)，严格绑定G原warm收据，累计330+最多240≤570≤600秒，不重复数值搜索。通过新cache输入/完整恢复/真实阶段转换/吞吐及完整空间/时限Gate后，唯一新组bbox_s_soccernet_seed17_S_to_R_cached_20261102_a01独立公开起点，30187S+6563R/总36750/482992S+105008R，77完整checkpoint/40val，最多16h/原总截止；所有原始证据继续直接G。不续R权重、不改原来活动job_table，不因缓存READY直接把GPU Gate标true。每60分钟只在实质阶段完成/失败/需处理时通知；三核心及缓存/小control证据闭合即停。


## CACHE-FAIL-001：物理内存保护停止，缓存未READY

UTC 2026-11-02T00:29:10.885362+00:00。用户询问缓存完成情况时只读发现：缓存于2026-11-02T00:18:00.276613Z（北京时间08:18）FAIL_STOP，实际5437/24149行（22.5%）、1009.906秒。GlobalMemoryStatusEx物理available4258082816B≈3.966GiB低于4GiB保护线；commit available17262096384B约16.08GiB，高于8GiB。writer9804实际身份已退出，最终目录不存在，partial/cache.json=FAIL_STOP；images.npy66767155328B为预分配大小，不能当作24149行已完成，samples.jsonl35798472B和原日志/源码/失败完整保留。未降低护栏、删除partial或自动重建，S→R保持BLOCKED。前述按健康速率估计的剩余约1小时已不适用。当前真实R仍RUNNING/1600更新、controller stderr0，R/R→R原队列继续，不因CPU缓存停止而重复训练。后续先核实际内存压力来源和有界安全恢复设计；若需重建/资源或协议选择须按最新直接用户授权处理，不把FAIL_STOP伪造为READY。


## CACHE-RESUME-001：单seed窗口缓存、100MB/s限速与Real优先续建

UTC 2026-11-02T01:05:49.513346+00:00。用户直接要求修复高内存并复用完成部分，随后提出每seed一个NPY、限制G源读取；在解释“约312GiB是剩余原4K输入，约62.2GiB是全量缩小后D缓存”后，最新明确先100MB/s观察Real影响，150或更多仅待评估。未批准或启用18小时。沿原4h总量，旧运行外层保守1051秒已扣，本次最多13349秒并更严限定原2026-11-02T04:00:32Z（北京时间12:00:32）。无训练参数/更新/曝光/评价/保存变更。

根因证据：旧整个62.18GiB memmap在异常前触及5437行约14.0GiB，flush只写回而没有unmap。独立64MiB临时夹具RSS从34459648升到101629952B，flush后101638144B，unmap后34533376B；8MiB窗口峰值42921984B。原future最多4帧约11MiB和约34MiB标注不能解释14GiB驻留。该夹具及其他非主线临时物已展示后精确清理，未读取原4K大媒体或运行GPU。缓存停止后同一Real R最近100步（结束2981）均值.48657秒；本次派发前3901–4000更新100步均值.48357、中位数.469秒，低于旧并行901–1000均值.91594秒。只支持共享资源竞争判断，不是隔离硬盘/CPU因果实验或数据集成绩。

新v2目标D:/projects/FutsalMOT_SyntheticDatasetResearch/cache/synthetic_bbox_960_v2，16train seed/24149图，arrays/<原seed>.npy16份+cache.json/samples.jsonl。总NPY66767157248B（约62.1818GiB），含元数据保守新增68981749760B（64.2443GiB），原v1.partial约62GiB保留，派发前D175.78GiB可满足另100GiB。每个消费者进程/常态writer最多1×32行84.375MiB view，只有D→D前缀复制可同时读写2×32行168.75MiB；窗口flush并close，不能把单seed2.76–5.53GiB整映射当有界工作集。旧5437行将按有序UID/完整JSONL及全部原RGB stat核验，再D→D精确复制；GT/meta SHA一次，只有3张旧原RGB重新LetterBox抽样。旧FAIL没有持久pixelSHA，新的前缀SHA是恢复基线，不冒称旧逐行SHA原已留存或已重验全部原RGB。原v1文件/失败/冻结代码保留原bytes，不自动删62GiB预分配尾部。

仅独立缓存进程对G:/FutsalMOT_Dataset内容施加100000000十进制B/s上限，read/readinto/Path/text/index/GT/PNG都经最多64KiB底层分块；单worker/线程1、无预取、BelowNormal CPU，物理4GiB/commit8GiB/D100GiB/缓存128GiB原门限不降。Real优先只读小状态和最多2MiB标量尾部：验证/启动/未知状态暂停；近期median超过.469×1.10退让30秒、每10秒观察恢复，限速/暂停每秒检查STOP与截止。存在最多10秒观测滞后，不能承诺零瞬态I/O干扰，也未自动提高150MB/s。v2再次失败保留完整数组/JSONL/完成数，当前没有再次resume入口，不宣称自动可续建。

实现代理22项CPU通过，根最终28项CPU检查PASS/0跳过/5.524秒，含窗口RSS、D前缀复用/旧bytes不动、原RGB stat变化拒绝、像素/metadata/增强exact、64KiB计费和暂停/STOP。根第一次用不存在的tests包名调用仅发生2个模块加载错误、没有运行测试；改用unittest discover后检查通过，不将调用错误当实现失败或隐瞒。最终input_cache.py SHA ab353daf5d5a0b37a626d1f8261979224f7df7f5cebd6a0a108ff84893206471。有界独立代码审阅发现的前缀RGB stat漏查及双view标签均已针对修订，没有修改活动GPU的36份私有包、job_table、封存方案或源数据。

实际唯一续建于2026-11-02T01:03:05Z启动，writer27536/create1793581385.3155513、wrapper10412/create1793581385.2740388，精确launcher G当前核心组/cache_resume_r01/launch.py，SHA b036b829c697b983485e585cbdbc41589bd4ba75d139f2e638b4f98e8dbd9d6e；41份源码/工具/依赖已冻结，code_bindings SHA becbc53d05c0f3f061007696fd8bbc70fb090d9be93c7c0f5620b379b10c2ae4，resume_plan SHA86d47cd13d22d2c48178a74a14a18f7c7c7cf6ada45302f4f003b2837e3220f3。初始状态PAUSED_REAL_PRIORITY，正确等待R4000验证（step004000_final_periodic），RSS440754176B、CUDA未初始化、stderr0；仅暂停态，实际稳定构建峰值与吞吐对照尚待观察。原GPU controller12244/创建时间/精确命令与锁均未变，Real原作业照常。原生bbox-r-r-r已原位更新为新v2事实及只读100步前后对照，仍每60分钟/原总截止/有限3核心，不新增监控。缓存尚未READY、S仍未训练；正式性能与合成贡献不由本CPU准备推导。


## CACHE-RETRY-001：限速微秒边界故障与唯一同协议重试

UTC 2026-11-02T01:11:54.685986+00:00。r01先正确暂停等待Real4000验证，在恢复小metadata读取时于2026-11-02T01:06:45.571676Z FAIL_STOP：_ReadBudget.charge在while判时后运行资源check，再次取clock已越过仅3279/100000000秒的短等待终点，给sleep传入负值。根用旧冻结代码/假时钟3279B并推进check1ms，确定复现ValueError及-0.00096721秒，无源数据或GPU。不是内存不足、数据验证失败或训练失败。r01主进程27536/wrapper10412已退出；未生成v2.partial、数组或迁移收据，旧三文件identity与manifest SHA均保持，全部r01计划/源码/状态/原stderr保留。

仅把check后等待改成单次计算remaining<=0即break、正值才sleep；增加确定性时钟跨界回归。实现代理23项CPU通过；根独立29项CPU PASS/0跳过/5.718秒，无临时产物遗留。新input_cache.py SHA266160960fcfab56f5ba730e3074cceca41bf471ae165879559b037207feb9e9；其他40份冻结工具/依赖/源码SHA均与r01相同，100000000B/s/64KiB/单worker/窗口/Real优先/数据与护栏无变化，不热改旧r01或GPU私有代码。

按既定最多一次基础设施修订，唯一r02于2026-11-02T01:09:57Z派发：writer28820/create1793581797.8625467、wrapper25232/create1793581797.8210754；G当前核心组/cache_resume_r02/launch.py SHA2c290153e605c6cb671e6f0237fe3da0a03279cf8fd1e84d3f485b32b611b04b，resume_plan SHAbe51af07b77f730edad18baf99073952eaa0d21fe28a1d3d5fa48cb94057412c，41代码/工具/依赖bindings SHAe1ff2e232a8de30f230cc7e744b138b01c0e8ccbe0e68634fae194e229ff24c9。旧外层1051秒+r01从创建到FAIL向上取整221秒均扣，最多13128秒且仍按原2026-11-02T04:00:32Z绝对截止，未延18h。r02再失败不自动r03。

实际r02已BUILDING；60秒时RSS461987840B，观测峰554708992B，CUDA未初始化、stderr0，Real R4513/最近小窗口median.469秒与缓存关闭基线一致。当前仍是catalogue/前缀核验阶段，尚未有新行，不能据此断言持续读4K原图无影响或正式缓存READY。第一轮持续读取的新行与100步比较随后记录，不启动额外benchmark或更改训练。原GPU锁12244不变，Real正常继续。原生跟进已原地登记r02及唯一重试上限。


## FORMAL：R_seed17 COMPLETE

UTC 2026-11-02T01:46:03.773432+00:00。目的：合成数据真实迁移五策略对照；公开YOLO11s独立起点、seed17，SoccerNet train2100/val750，synthetic train24149，策略R。实际已提交6563/6563更新，R/S曝光105008/0；960/FP32/micro8/有效16/workers2，SGD Nesterov momentum.937、WD.0005，clip10。第一阶段30187步峰LR.01，真实终阶段6563步峰LR.001，各自cosine到.01倍，阶段重建SGD/EMA并承接raw；独立R仅终阶段、S仅第一阶段。检查点0/每500/阶段末完整状态，PT/每1000/阶段末750图原生预测与全部负结果保留。不取best、不读test；本轮结果尚需与同seed/其余seed对照，不能单轮宣布贡献。

状态/原始结果：{"state": "COMPLETE", "directory": "G:\\FutsalMOT_Training\\runs\\bbox_s_soccernet_seed17_core_20261102_a01\\R_seed17", "update": 6563, "exposures": {"R": 105008, "S": 0}, "terminal": {"tag": "step006563_final_final", "update": 6563, "main_metrics": {"AP50_95": 0.5165115374980012, "AP50": 0.941099523964852, "AP75": 0.5056873088257927, "AP_small": 0.33257477789442114, "AP_medium": 0.5057760772141966, "AP_large": 0.5835367728536506, "AR_max1": 0.05302695092518102, "AR_max10": 0.45075422365245377, "AR_max100": 0.6037510056315366, "AR_small": 0.38661417322834646, "AR_medium": 0.6047788697788699, "AR_large": 0.6348387096774194}, "raw_prediction_bytes": 4246207173, "images": 750, "wall_seconds": 330.6399999999994, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "phase": "final"}。证据：G:/FutsalMOT_Training/runs/bbox_s_soccernet_seed17_core_20261102_a01/R_seed17。


## FORMAL：R_to_R_seed17 STARTED

UTC 2026-11-02T01:46:10.954270+00:00。目的：合成数据真实迁移五策略对照；公开YOLO11s独立起点、seed17，SoccerNet train2100/val750，synthetic train24149，策略R->R。实际已提交0/36750更新，R/S曝光0/0；960/FP32/micro8/有效16/workers2，SGD Nesterov momentum.937、WD.0005，clip10。第一阶段30187步峰LR.01，真实终阶段6563步峰LR.001，各自cosine到.01倍，阶段重建SGD/EMA并承接raw；独立R仅终阶段、S仅第一阶段。检查点0/每500/阶段末完整状态，PT/每1000/阶段末750图原生预测与全部负结果保留。不取best、不读test；本轮结果尚需与同seed/其余seed对照，不能单轮宣布贡献。

状态/原始结果：{"state": "public_initialized_zero_updates"}。证据：G:/FutsalMOT_Training/runs/bbox_s_soccernet_seed17_core_20261102_a01/R_to_R_seed17。


## NAMING-APPLY-001：用户批准统一命名并实际应用

UTC 2026-11-02T02:00:01.569879+00:00。用户明确“命名设计写入到设计文档中去，然后应用”。仅在EXPERIMENTS新增命名节，反向移除该节后原文SHA精确还原46aa2fe83daf99bc0a2e1880f593eb36b2c11297b95f8eaaf479de19c4401446，证明其他原文未改；本次新封存SHA62d25394f570517caa79b9eaeba510ba06683604de86d6dbeaf3b4234860abf6，AGENTS登记新基线，PAPER仍d48cb4160a513773cde0e910fe1c45c7eb5a63bf058871eafd38e9ad27b71c17。没有改变训练预算/数据划分/损失增强/评价/原始证据/资源与停止Gate。旧已经启动run继续保留旧计划摘要及所有路径/ID，属于合法历史引用。

监控v6统一主标题、选择器、对比图例与CSV名称为阶段｜任务｜模型｜真实面板｜策略｜训练seed，例如“正式｜BBox｜YOLO11s｜SoccerNet｜R→R｜seed17”；校准用校准R50/R100，未知明确未登记。来源只从源绑定确定，不靠组名猜。纯函数canonical_group_name生成main_bbox_yolo11s_soccernet_S2R_seed17_YYYYMMDD_a01（aware时间转北京时间），拒绝危险slug/naive时间/bool种子，内部detect、S->R等采样token及稳定选择ID仍原值。未来新单作业外层D/G同名组采用规范；既有核心组内部R_seed17/R_to_R_seed17及历史目录不重命名。原ID和路径继续放监控详情，未创建空的新训练目录。

根独立48项CPU/临时HTTP检查PASS/0跳过/1.344s，内嵌HTML JavaScript Node语法PASS，临时夹具已清理。只重启已按登记PID/创建时间/完整命令/健康接口核实的只读监控6096；新服务PID10016/create1793584544.0099916，URL仍http://127.0.0.1:8765，截止2026-11-03T23:48:11.085495Z未超过原48h剩余界限，无临时监控日志。实际HTTP验证四个训练均新标签，R50/R100/正式R/正式R→R的原稳定选择ID全部一致。训练12244及缓存28820真实身份/锁均未改变，没有重启训练或缓存。刷新已打开页面可加载新HTML。

为未来新正式实验绑定此次获准新计划，只在future src/formal.py增加FORMAL_PLAN_SHA并用于protocol/prepare/sourcebindings三处，SHA13227250935c04ef2eb1facc159e600c3b7be2905ef6454b18640f2d0ae9743c。原runner SHA仍1f54a69303195fd7d924611394c5ddda47780d9ac80864908f05c5c6ff7ff6a9、当前private formal仍8c853d142efe6b083999f0ef573bea4eb47d8c2be3eca254f10534d87783ccc9，其他数值来源检查未改。根独立28项formal CPU PASS/0跳过/10.624s：五策略新旧完整字段唯一差异为批准的EXPERIMENTS_sha256；不是新的GPU数值预检或额外训练。旧活动RR已经通过原计划独立job入口，本次根文档更新不会在其内部阶段重复检查；若日后旧run中断并恢复，原入口可能因根新SHA拒绝，须单独明确兼容审阅，不允许热改旧包绕过。

R已完整结束并自动接续R→R，仍由原冻结控制器/作业表执行。原生bbox-r-r-r已更新新根计划基线、旧run历史绑定说明及未来canonical S2R路径生成；仍须cache READY/旧GPU块闭合/原cached预检与完整资源Gate，不扩大三核心范围，不因命名另建并行训练。


## CACHE-TIMEOUT-001：r02到原绝对截止安全停止，S→R输入仍阻塞

UTC 2026-11-02T04:18:27.491937+00:00。唯一r02于2026-11-02T04:00:32.527201Z（北京时间12:00:32）FAIL_STOP，TimeoutError: original cache absolute deadline reached。冻结launch.py的stop_check在为Real9000验证暂停、读取下一张RGB期间命中原2026-11-02T04:00:32Z截止；没有放宽原4h总量，也没有批准/启用18h。此为预算到期，不是再次内存保护、负sleep或数据不一致。原writer28820/create1793581797.8625467及wrapper25232/create1793581797.8210754均已核实际退出，未启动r03、未清旧失败或partial。

对象为16个train合成seed/24149张的无损960 RGB LetterBox输入缓存，来源UID/split/标注和随机增强协议不变；CPU单低优先级线程/worker、100000000B/s源读取上限、64KiB分块、32行映射、Real优先，训练/optimizer更新均0。v2.partial/cache.json实际持久记录6419/24149行（26.58%）：复用旧5437行，加本次新982行；其余17730行未完成，不能把16个预分配NPY或文件逻辑长度当成READY。最终v2目录不存在，不执行validate_ready_cache或cached GPU预检。原v1三文件/失败、r01失败、r02冻结计划/源码/原stderr全部保留。v2失败manifest SHA8af63689de69895ef07a19154c374a890c8c36503d3dc580678b4765ae5a652b，r02/status SHA1c0f887ee70096075a95e465ee7e7fcf42f40b41e0cd83779cf1397b9d251de2，per_seed_migration_receipt SHA8aacf3ef89b62408395f0a057c899592126a97fba9ab17d4b851db35705d5b9e；仅核小收据，没有读大NPY/原图或全量hash。

r02外层elapsed10231.828秒（包含检查/前缀迁移/Real退让），缓存manifest内部write stage9303.265秒；不能混成同一计时范围。实际G累计读取21964712468B，暂停计数4370.761秒，物理RSS观测峰665272320B约0.62GiB，映射峰88473600B，CUDA未初始化。与旧v1触页14GiB驻留相比，新结构在本次实测中控制住内存，但Real退让/低优先级/逐图处理及其他开销下仍未在时限内完成；没有把100MB/s上限当实际吞吐，不能凭本结果批准150或保证延时长度。旧100步对照只支持该窗口未见明显Real变慢，不能推出无干扰。

GPU独立工作继续：controller12244/create1793577174.1052206/精确原launch与GPU锁仍一致，R→R当前10000/36750、first阶段、160000R/0S、既定验证中，训练stderr0；原s/seed17/960/FP32/micro8有效16/workers2、SoccerNet train2100/val750和完整记录/16h预算不变。R已COMPLETE/6563/105008R，末步主AP.5165115374980012，驱动已有记录。本次缓存失败不是R→R训练失败，不重启/改参/读test。观察时D余121923280896B、G余1002233679872B，100GiB保留满足。

cached S→R仍BLOCKED于输入未READY；不以缺失合成结果宣布贡献。唯一基础设施重试已用，当前v2没有再次恢复入口，不能自动覆盖或重建；若要继续缓存，需用户另行批准新的有限续建预算，并先明确失败前缀复用/恢复实现与独立冻结，不能借延时热改旧任务。当前还有已授权R→R在运行，继续原60分钟观察；待其完整终点/小控制SEALED/owned退出闭合后，如无用户新许可，报告三核心缺S的具体阻塞并停止本跟进，不因缓存结束漏观察R→R，也不重复通知本次已知停止。方案/PAPER摘要均保持。


## CACHE-EXTEND-001：用户新授权6419行原位续建，独立66小时有限预算

UTC 2026-11-02T04:31:54.292173+00:00。用户明确“允许延长缓存构建时间，复用已完成的6419行继续构建，仍以不影响Real训练为前提。根据实际速度设置合理的有限时长，不要再用原来的4小时截止”。这覆盖旧r02预算到期后的禁止自动续建条款，属于新用户预算许可；原r02及r01/v1的FAIL/冻结/日志不改，不能冒称原失败已恢复成功或基础设施自动r03。

依据原r02外层10231.828秒仅新增982行（不是把复用5437当新生成），剩17730行线性保守推51.31543小时，含25%余量为64.14429小时，上取本次66小时/237600秒。包含本次来源与前缀核验、Real暂停让路，不从旧4h再扣；实际启动2026-11-02T04:30:21Z（北京时间12:30:21），冻结新截止2026-11-04T22:30:21.146105Z（北京时间11月5日06:30:21），仍早于研究全局截止。此为硬上限，非保证完成时间；Real结束后可能提早完成，未知变慢也不自动再延时。

新独立src/futsalmot/cache_resume_v2.py只实现v2.partial原位续建，SHA481b54d63eca9d399a7cb790cc53466ef03db9a4831bdb122695bdd4feeb5c1f；原input_cache.py仍266160960fcfab56f5ba730e3074cceca41bf471ae165879559b037207feb9e9，不修改活动GPU/旧冻结。D余约113.55GiB，不复制另一套62GiB。先确认旧writer28820/wrapper25232退出、6419完整JSONL末row6418与newline，再全核现存D前缀像素及JSONL/UID顺序/GT绑定/RGB stat；仅3张原RGB抽样，不扫全4K。旧FAIL未保存逐row pixel SHA，新核的是本次D前缀恢复基线，receipt明确old_pixel_sha_durable=false，不能伪造旧SHA。完整核验后才保存旧FAIL原bytes与逐行新SHA/数组身份receipt，再修改manifest和未提交尾，原JSONL前缀bytes不变；未知多行/半行尾拒绝、不猜截断，16个数组原位使用。最终新行逐像素write/read exact，pixel flush/close先于JSONL fsync/manifest；前缀读回与全部结果IO在READY同卷发布前闭合，发布后仅可选显示回调被隔离，不能把已发布READY反记FAIL。

实现代理26项CPU通过；根第一次stdin测试入口引发旧Windows spawn夹具加载错误（51项已过、1个入口错误），没有真实数据写入；改为-m unittest可导入入口，52项独立检查PASS。随后对同一READY发布路径的必要读回/统计收尾再加3项故障夹具，根最终55项相关CPU PASS/0跳过/7.171秒，涵盖原位前缀/未知尾/数据绑定/失败边界/Real优先/窗口/发布回调。TemporaryDirectory自动清理，测试源码保留，无诊断模型/报告/缓存夹具遗留；没有新增GPU数值搜索或optimizer更新。独立只读审阅确认需闭合的发布后异常及v1+v2合计门限已针对修订。

实际唯一新CPU组G:/FutsalMOT_Training/runs/bbox_s_soccernet_seed17_core_20261102_a01/cache_extend_c01。42份小代码/工具/依赖冻结（旧r02原41份逐文件SHA相同，新增resume模块1份）；bindings SHA9537a56480b8496aca0aff366b5a9b880b14488231da32f8dbc540fc3cd75995，resume_plan SHA8359706d9c53519e813411bf7a696ec755d31809224b61dab019e75db6fab61b，launch SHA3a553eab6755395379d64822bd290c86d5cdb3d19f7fb5da2e4ea8a3ae794f79。wrapper13940/create1793593821.220506，实际writer8952/create1793593821.227307，完整命令新G组launch.py，runtime_identity与实际进程一致；研究根.synthetic_cache.lock与cache/.synthetic_bbox_960_v2.resume.lock均属于同一writer，未知锁不清除。结束仅精确退休自有锁并保留小证据。

初始状态PAUSED_REAL_PRIORITY，正在Real11000验证step011000_first_periodic期间退让；progress PREPARING_SOURCE_AND_PREFIX/completed_rows6419/prefix_verification_complete=false，RSS440233984B、CUDA未初始化、stderr0。此时D manifest仍旧FAIL是预期，因为核验完成前不修改它，不因其旧FAIL重复派发；后续只观察新c01/status与真实身份。不能把初始6419当本次新增行或已完成核验，不能称缓存READY。单CPU低优先级、线程1/worker1、100000000B/s/64KiB源上限保持；Real验证/启动/未知状态暂停，近期median>.469秒×1.10退让30秒，10秒检查，不能保证零瞬态影响，不自动150。物理4GiB/提交8GiB、D保留100GiB、旧v1+v2已知文件合计128GiB门限保持。

原GPU12244/创建时间/完整命令/锁未变，R→R仍按原job_table在first/11000/36750、176000R/0S训练验证，s/seed17/960/FP32/micro8有效16/workers2/SoccerNet train2100val750不变。本次前缓存关闭100步10359–10458 mean.48276/median.484秒为观察值，仍使用原.469阈值，不重设宽松基线。S→R未启动：须本次cache READY24149/16arrays+writer退出、当前R/RR终点/SEALED/owned闭合及原最多240秒cached预检和完整资源Gate。66h许可不扩大三核心/多seed/任务/test，不修改封存EXPERIMENTS/PAPER。原生同一bbox-r-r-r按原60分钟继续监测新c01和旧GPU，正常中间态安静；失败/取消/本次新截止到期保留证据，不自动追加预算或新续建。


## CACHE-EXTEND-VERIFY-001：6419行前缀核验闭合并实际追加新行

UTC 2026-11-02T05:21:30.731733+00:00。仅核小收据/进程/标量，未重复运行检查或扫描原图/NPY。cache_extend_c01实际writer8952/create1793593821.227307与精确命令仍匹配，prefix_verification_complete=true；prefix_resume_receipt SHA01dfb25f4377cad07f55cbc6029c823654d32da03846f28150803b4db42ea211，保存的original_v2_FAIL_manifest SHA8af63689de69895ef07a19154c374a890c8c36503d3dc580678b4765ae5a652b精确匹配旧失败。6419行JSONL前缀SHA692190547616dafec7a44aaf467e21423f06d4e3cf2237f041a217b7b62a463f，D像素新恢复基线SHAd633ba675f2cd4283c809242de308cb9b0bc893f088d580c04087814f425b262，明确old_pixel_sha_durable=false。抽样row0/3209/6418原RGB LetterBox逐像素exact；前缀bytes保留，未把新基线说成旧持久像素SHA。

本轮小状态BUILDING/6656/24149（6419原位复用后已新增237行），源读取累计6293287961B，RSS495792128B、观测峰566263808B，映射峰88473600B，CUDA未初始化，stderr0；属于正常构建中，尚未READY。66h新截止北京时间11月5日06:30:21及100MB/s/Real退让/资源门限保持，不提速或延时。新receipt只是恢复准备闭合，不是正式S训练/数据集成绩。

原R→R控制器12244/CT/命令/锁保持，最近100步14334–14433 mean.48974秒/median.484秒，相比本次续建前10359–10458的mean.48276/median.484秒，mean约+1.45%、median相同；与原.469秒退让阈值相比median约+3.20%低于10%，未见该窗口明显变慢。不同样本和自动暂停使该对比不是隔离I/O因果实验，不保证零瞬态影响。D余121887412224B/G余983376142336B/物理19199594496B，原保留满足。后续只接续观察，不重建或重复核前缀；Real与缓存均未结束，不启动S→R、不通知重复健康进度。


## CACHE-STATUS-001：用户要求缓存慢速说明与停止例行训练巡查

UTC 2026-11-02T06:19:37.244136+00:00。用户已有训练观察台，要求训练人工监视、agent无需再考虑训练，并询问缓存推进缓慢。仅核缓存status/进程/既有实现，未读取训练steps/val/日志、未修改活动缓存/训练，未新增GPU或benchmark。2026-11-02T06:16:27.478075Z（北京时间14:16）缓存BUILDING/7072/24149=29.2849%，复用6419后本次新653，余17077。外层6362.703秒含来源/前缀核验与Real退让；累计source_bytes_read13697772177B，含准备与暂停均速约2.1528MB/s，此非独立HDD读速。小进度快照已记paused_seconds2651.651，约44.194分钟/外层41.675%；进度每32行更新，暂停计数有快照滞后，不能精确拆分所有耗时。单worker/单线程依次读取4K PNG、解码/LetterBox、像素读回与metadata检查、D写入，且64KiB粒度限速/资源检查存在额外开销；本轮没有分项profiler证据，不能断言硬盘/解码/检查各占多少或已证明某项为唯一瓶颈。

本轮全程约369.466新行/小时，按此含准备/暂停的平均保守推余46.221小时，非完成承诺。100MB/s为上限，当前整体明显未达上限，直接改150不能据此预期提速；未授权/未实际提高。RSS504053760B约.469GiB、观测峰574877696B约.535GiB，CUDA未初始化、stderr0，D余113.500GiB，现证据不指向内存再次不足。实际writer8952/CT1793593821.227307/精确命令核实仍匹配。缓存当前正常构建，完整READY仍未到，不把预分配62GiB称完成。

已将同一原生跟进改为“SSD缓存构建跟进”，只看缓存，训练进度/AP人工观察，不再根例行100步巡查；不关闭冻结缓存内部RealTrainingPriority自动退让，也不改变原先用户Real优先要求。未来S→R只在缓存READY后执行必要的一次旧组闭合/GPU互斥准入，不因缩小巡查而绕科学资源Gate。66h/北京时间11月5日06:30:21截止、原数据与证据、内存/磁盘余量保持；没有授权热改/重启/新加任务。


## PLAN-SCOPE-001：用户决定单seed探索，扩展留方案不列TODO

UTC 2026-11-02T06:56:12.016591+00:00。用户明确从TODO删除扩展训练内容、方案保留不同模型规格但等待s训练完成查看效果再决定；多seed是否执行依后续训练进度决定。用户同时明确Pose与Seg是项目论文核心、尚无成熟方案、暂不加入TODO、后续要详细规划；本次不是永久删去这些研究目标。对“未来新实验只阶段0和阶段末验证”只允许讨论，没有批准实际改频率。

已删除TODO展开的多seed/多规格/Pose/Seg单项矩阵，以及旧A3其他任务/D1–D4/多规格C5等未执行待办，保留当前BBox s/seed17的R、R→R、cached S→R三核心及两次已完成校准共5条训练记录（3条已完成、1条已启动待终验、1条条件待启动）。首次编文校验发现旧A3未执行任务仍残留，校验拒绝写TODO，补齐清理后读回通过；不是训练或缓存失败。旧实验/失败/原始模型预测/日志无删除；PLAN-REDUCE-001中的9正式/11总只是建议，现被用户最新3正式/5记录范围取代，不自动执行后续六次。S/R+S未列当前TODO，原候选方法留方案。

用户直接允许的EXPERIMENTS范围条款已原地更新并重新封存，旧SHA62d25394f570517caa79b9eaeba510ba06683604de86d6dbeaf3b4234860abf6，新SHAd17e46ea0075b3a6882e4b413df35455f1175395e5ece45b1e464d5e69375f6b。新首节、自动顺序/候选矩阵/seed候选说明及Pose/Seg成熟度说明均与上述新范围一致；n/m/l/x不是删除设计，而是待s结果/资源后直接讨论。已启用规格与seed结果全报，不因低分重训/只保留上涨项。PAPER仍d48cb4160a513773cde0e910fe1c45c7eb5a63bf058871eafd38e9ad27b71c17且文件bytes/mtime未动；AGENTS新基线、README实际范围、config执行范围及同一原生缓存跟进已同步。当前训练/缓存冻结包仍引用各自合法历史计划46aa.../62d...，不回写原run/plan/代码；本次不巡查训练，不改变Real退让或缓存66h/全局截止。

为未来未冻结S→R只机械更新src/formal.py的FORMAL_PLAN_SHA，新源码SHA27a548dbf57c19f61f0c50ce16d9b6d666a4764f80b69611642518762a964833；既有CPU fixture只更新新摘要期望，根28项CPU检查PASS/0跳过15.707秒，五策略字段除获准计划摘要外与旧冻结版相同；未执行新GPU数值搜索或optimizer更新。runner仍1f54a693...、formal_cache_preflight仍434593...、input_cache/cache_resume均未改。原GPU job_table SHA仍b97371e2812779597db5bcb517ad14b10b191b8f9510ab75678bd04b71fefa68、private formal仍8c853d14...；CPU c01 bindings与launch小文件SHA保持。临时CPU夹具自动清理，无新测试报告/模型遗留。

验证降频讨论：严格保留每阶段0及阶段末，独立R候选由8次val到2次，两阶段S→R/R→R由40次到4次；此前PLAN-REDUCE-001写成3次是计数错误，此处更正。第二阶段从前阶段raw重建EMA，阶段0并不等于前阶段最终EMA，不能无说明合并。该候选可减少长任务验证/密集预测这一部分约90%，不代表总训练/存储都减90%；仍保留逐步loss/LR/梯度/UID、0/每500/末完整checkpoint及实际执行的全部预测与负项，代价为失去中途AP曲线，异常更多依赖逐步数值监测。若用户采纳，须明确实际变更许可、仅新run生效、核RNG/执行语义与完整资源预算后冻结；当前原val/checkpoint规则完整未改。

多seed讨论依据：初始化、采样和增强波动会影响方法比较，重复训练是稳定性证据的重要组成，并非所有研究统一规定的必做次数，也不只是表格装饰。单seed可以支持探索、排查流程和报告该次绝对效果，不能直接推为稳定平均优势；多seed不能补足真实GT、合理对照或独立比赛覆盖。若后续仅在同一seed数据/评价/预算下做配对重复，应全部记录并给出变异范围；三seed仍只粗描训练波动、不等于三场比赛。参考Bouthillier等Accounting for Variance in Machine Learning Benchmarks（https://arxiv.org/abs/2103.03098）及NeurIPS Paper Checklist（https://neurips.cc/public/guides/PaperChecklist），后者允许如实说明算力不足的局限，不将固定三seed当硬门槛。本次不自动采纳新的seed数或追加比赛/test。


## VAL-ENDPOINT-001：用户批准未来每阶段0/末验证并实现实际调度

UTC 2026-11-02T07:11:59.517477+00:00。用户明确“同意未来新实验只在每个阶段0和阶段末验证，保留逐步日志和每500步完整检查点。更新对应方案和代码，当前正在进行的训练不改。”仅对尚未冻结的新实验生效；当前R→R、CPU cache_extend_c01及所有历史run/代码/计划/原始预测/失败不回写、不删、不重启。

EXPERIMENTS获准验证条款原地更新并重新封存，旧SHAd17e46ea0075b3a6882e4b413df35455f1175395e5ece45b1e464d5e69375f6b，新SHA3814e51504548a39e6b88fdf073a4377ba22b6ddf7b1607330bfb1664c6eac67；PAPER仍d48cb4160a513773cde0e910fe1c45c7eb5a63bf058871eafd38e9ad27b71c17，bytes/mtime未改。AGENTS新基线、TODO/README实际规则、config未来策略及同一原生缓存跟进已同步。当前仍BBox/YOLO11s/seed17三核心，S→R条件待启动，型号/多seed/Pose/Seg范围不扩，原全局截止与缓存66h截止不延。

未来src/formal.py SHA d0173fe14d3a0072add7da62d18cdc13e51ff57fb352254c223600b06dbbdcfb。evaluation_policy=phase_endpoints_v1、evaluation_every=null、精确evaluation_schedule：R/S各2次val，两阶段S→R/R→R/R+S各4次。first阶段末EMA与final阶段0由raw重建EMA即便global update均30187，仍保留step030187_first_final和step030187_final_initial两条。实际run_job新鲜训练、每步循环及显式恢复共用_evaluate_if_due/evaluation_due；已删除两处%1000实际调用判断，非端点500/1000等恢复不补val，绝非只改返回schedule表或能力布尔值。未来_journal同步端点文案。主指标/后处理阈值/val750图/末步EMA/不best/test保持，每次实际执行val的全部原生密集预测/标准与辅助AP/失败仍留。

数据及训练预算保持：SoccerNet原train2100/val750、synthetic全部16train seed/24149图；s/seed17/960/FP32/micro8有效16/workers2，公开YOLO11s独立起点，SGD/增强/criterion/采样/阶段重置/更新与R/S曝光不改。R6563/105008R、S30187/482992S、长任务36750及对应588000总图曝光目标不变，checkpoint仍R15份/S62份/长77份，每阶段0/每500/阶段末及受控失败完整raw/EMA/SGD/RNG/采样/loader状态，逐步loss/LR/梯度/UID全部保留。

原runner.evaluate已在正常评价finally恢复训练RNG并使用独立eval loader generator；其初始化在try之前，未来formal._evaluate外围另以finally覆盖初始化失败/成功返回，恢复Python/NumPy/Torch CPU及已初始化CUDA RNG、训练loader generator和原model各子模块training模式。runner.py仍1f54a69303195fd7d924611394c5ddda47780d9ac80864908f05c5c6ff7ff6a9，不改历史入口/私有包。训练微型CPU夹具在不评价、较密评价和端点评价三种情况下，6步loss及最终参数逐值相同，属于准备证据而非数据集成绩/正式训练；主线optimizer更新0，未运行真实推理、GPU数值测试或读取test。

根31项formal/预检相关CPU检查PASS/0跳过9.554秒，含实际调用/恢复中间点跳过、交界两模型评价、成功失败RNG+模式保护、微型训练数值一致及新旧协议逐字段差异核对。新旧协议仅获准EXPERIMENTS_sha256、evaluation_every、evaluation_schedule与新增evaluation_policy改变；checkpoint/图像预算等全同。原始两处%1000条件静态已无，实际调用均使用共享执行器。一次子代理工具容量不足未执行任何代码，root完成实现/检查，没有未结束的代码任务；TemporaryDirectory夹具自动清理，测试源码保留，无诊断权重/临时报表遗留，不重复无变化全套/GPU预检。

未来formal_cache_preflight.py SHA1be92becbf3be51abafa7e649460ac676e789c99c14b87a00b45d95a4576241d。完整freeze会包含此前用户已批准的CPU缓存续建helper，新门禁仅对cache_resume_v2.py精确SHA481b54d63eca9d399a7cb790cc53466ef03db9a4831bdb122695bdd4feeb5c1f允许CPU-only新增文件；错误SHA或其他未知新增模块仍拒绝，原网络/loss/optimizer/数据/增强模块不能改。真实旧admission的executed_code集合与当前源集合已实际通过该门禁，没有以扩大机械模块名单绕数值Gate；cached真实GPU预检仍须旧R/RR完全闭合/GPU释放+完整cache READY后最多240秒，原330秒已花不重置，总≤570≤600秒。

预算重算口径：保持旧36750步日志2MiB/步、77完整cp实测×1.1、每图native6650344B及JSON1MiB、一次750图失败partial和瞬态；只改val40→4，旧长任务326176543074B/303.77558GiB减去36×750×(6650344+1048576)B，候选110.18077GiB。另G100GiB、D32MiB小控制+100GiB保持，最终须cached实际checkpoint/吞吐/剩余容量及硬截止重新冻结，不直接把候选资源Gate写PASS。长任务验证/预测部分减少90%，不是总存储或训练更新减少90%。未启用s15理论val480→48、native保守2229.75→222.97GiB，checkpoint与约861.33GiB逐步日志上界仍在，整个矩阵仍不准入/不回TODO。

最终只读小绑定确认旧GPU private formal仍8c853d142efe6b083999f0ef573bea4eb47d8c2be3eca254f10534d87783ccc9、job_table仍b97371e2812779597db5bcb517ad14b10b191b8f9510ab75678bd04b71fefa68；CPU c01的42冻结文件逐一SHA、code_bindings9537a564...、launch3a553eab...全部原bytes匹配。未读取实时训练status/steps/AP或按旧PID操作。新方案摘要不要求旧run恢复覆盖，旧46aa/62d/d17引用继续合法；若未来旧run需用户显式恢复，单独处理绑定兼容，不热改现运行代码。

## SEG-GT-001：公开足球实例Mask核验与待审最小实验候选

本轮UTC 2026-11-02T10:30:07.2963606+00:00。用户明确先着眼Seg，继续核验足球公开GT的格式、划分和可用性，暂不训练。仅进行公开元数据/标注的有限CPU解析、样图及短片抽样；主线训练更新0，未加载模型/初始化CUDA/推理，不巡查或改动当前训练、缓存与自动化。EXPERIMENTS/PAPER与TODO/README均不改，不建立新方案、报告树或训练组。以下均为待直接审阅的候选，不构成训练准入。

ihelon/football-player-segmentation，Kaggle版本1（平台登记2023-04-11初版），原始annotations/instances_default.json实际18691334B，SHA256=9db8c37770afbaa3a9cba0d41437e6ca1cb6a77fcf0eca3e5ca115ebb97e995b。匿名单文件访问HTTP200，标注ZIP5846296B；原始文件清单3页完整查询共513文件：512张JPG共327587594B、1份COCO JSON，图像与标注文件名双向无缺失；不是凭文件最大编号推算图数。512图全部1920x1080，7686个唯一annotation id，类别仅person；每图4–31实例，每条为一个非空polygon，结构偶数坐标且至少3点，未见multipart，iscrowd全0、attributes.occluded全false。后者不能用作可信遮挡分层真值。COCO bbox为原图绝对xywh，segmentation为绝对像素多边形，area单独保存；不应当作YOLO归一化polygon原样读。

Voxel51镜像固定commit=4fd647b30d7504007544b9a2e007c0822a0d6e91。metadata.json实际7404B/SHA=c762652bbe3c3ba5884a6541b7e478d0ec361fb121755fc02a993a66e6d0187e；samples.json实际8110085B/SHA=1eddb4f5eb7ac0303a34e9bd0de2608be12a74777089721cd4fb0974c63a81f7。实读512图/7686 detections与7686 segmentations；前者无mask，后者全有mask，单类person。归一化bbox+框内二值mask，mask以base64/zlib包装NPY保存；3个实例实际解码为bool二维数组。镜像data_0至data_5只是存储分片，没有比赛、seed或split语义，不得按这些目录划分。原始COCO宜作为正式派生/评价的源，镜像只辅助核验，不强制安装FiftyOne。

来源/划分核验：原始COCO info的contributor/date/url/version/year为空，所有date_captured=0，license对象为空；Kaggle数据卡/API及HF卡另登记CC0。技术上有现成提供的实例标注，但原始标注生成/人工复核过程未披露，不能直接称逐像素人工精修GT；元数据空白也不能推断无许可或判为SAM伪标签。未找到官方train/val/test、比赛id、轨迹id、来源时间或原视频链接。按数值文件序均匀取9张：0/64/130/197/266/330/398/462/527.jpg，另查看34/99.jpg；场地、红/浅蓝球衣及纽约红牛场边文字一致，支持单场来源的保守解释，不代替全部512图比赛身份证明。0/34/99原图与COCO GT叠加共60个实例实际经pycocotools解码（17+30+13），用于小样格式/对齐目视核验，不是全库GT质量PASS或模型成绩。现有标注包含门将/裁判等，不能自动偷换为players-only，也不擅自增加观众ignore规则。

第二足球候选quantigoai/soccer-dataset，Kaggle版本1，平台登记CC BY-SA4.0；上传方称3段YouTube来源短片，未提供可核的完整原视频身份链，不把其MIT来源描述当已核实的全部媒体许可。完整文件清单12项，原始总38403014B。3份ann/instances.json实际为COCO polygons，并非只根据Panoptic名称推定逐人格式。Batch1=227图/679条各类标注/423 Player+14 Referee，JSON676428B/SHA=93d46e35e23624ef38ca2c045f6593865d755e7fea8e2264c99c77e96937eb84；Batch2=195图/1660条/927 Player+115 Referee，2080349B/SHA=ff668481f2b00a0c9b2d6cb15ce806a2d33e6ed6973cdc4fe66d10ba63afd740；Batch3=425图/7128条/3877 Player+969 Referee，5867055B/SHA=d50519a9c04b219be458419db27e11d05d70911ddebfa903999467d696f79a95。合计847图/9467各类annotation，其中5227 Player、1098 Referee；候选统一person评价应为6325条，不能把所有9467都报为人物实例。数据卡844/9455与实际文件不一致，按版本1实文件另记，不改来源数据。

3个原生MP4合计29249206B，临时读取CPU视频头为227/195/425帧、均30FPS，与3份JSON图数一致；每片仅取首末两帧，未全量解码/媒体hash。抽帧显示JUV/RM比分牌、同组球衣/场景与统一视频水印，按同一场比赛的3个剪辑保守处理，不宣称3场独立比赛；含近景/全景与镜头切换，无官方split。两来源抽样场景明显不同，但正式冻结前仍需登记身份/排除重复血缘，不能凭不同下载站点自动认定完全独立。

第三候选sadhliroomyprime/football-semantic-segmentation，平台登记CC BY4.0，原作者描述100张、2017皇马对曼联超级杯集锦、每12帧取样且COCO fused labels；11语义类别。其主要定位是语义分割，未读取153300437B大COCO文件，不能当已通过逐人实例GT的候选，不用连通域或SAM补成真实instance GT。本轮优先ihelon，Quantigo外部足球评价候选，不默认转篮球。

待审最小方案：ihelon全512图作为独立Seg真实训练池候选；Quantigo整库847帧作为已观察样图的外部足球val候选，不切其3个同场片段冒充比赛级train/val/test，不创建随机逐帧80/20。此为新的跨数据集探索面板，尚无官方独立test，不改当前SoccerNet BBox的split或GT，不称全新盲测。训练/评价统一保留原person语义；Quantigo在独立派生视图把Player与Referee映到person，原类别/id/bytes保留，需明确批准这一评价映射。类内未标注看台人、源标注方法不明和单外部比赛相关帧均列局限，不用新增ignore规则或人工修GT暗补。

计算预算候选仅YOLO11s-seg/训练seed17，一个任务三组，不扩Pose/多规格/多seed：R=500R更新/8000R曝光；R→R=500R_extra+500R/1000更新/16000R；S→R=500S+500R/1000更新/8000S+8000R。三组共2500更新/32000R+8000S，每组均独立公开预训练起点，不能续当前BBox或相互借已训权重。S候选为16个原train seed各100整图共1600，先按事前机位/时间/UID规则冻结，不按成绩选图，原BBox全24149池不变；500步有效16对应S5等效轮、R15.625等效轮，RR31.25等效轮，绝非BBox R50已证明该Seg预算充分收敛。保留零更新公开起点并报告S→R相对R及同更新RR的绝对Mask AP和配对差；无test、无best或低分重训。固定预算前不跑模型评价。

资源/实现候选：960整图、FP32、micro1累积到有效16、workers1，须GPU释放后每任务型号至多10分钟实测准入，当前不运行。按获准规则每阶段0/末val、0/每500/末完整raw+EMA+optimizer+RNG+采样检查点及逐步loss/LR/梯度/UID记录，候选R2份cp/2val，两阶段各4份cp/4val（阶段间两份状态不合并），三组10份cp/10val事件；若每次全847图即8470验证图像事件，raw/EMA前向与实际密集预测额外据实现/实测预算，不把事件数当实际网络前向数。缩短阶段后warmup与学习率须给绝对调度表另审，不静默套1000步warmup或复制BBox已冻结数值。候选沿现有Seg mask_ratio2做资源检验，原始polygon/整数实例mask须保留；训练适配不能用轮廓桥接抹掉合成mask孔洞/断开部分。现有BBox Trainer不是已完成的Seg Trainer，Mask loss、真实增强、完整恢复和评价仍需实际实现/有界验证，不能仅设Gate true。

输入预计仅数百MiB级公开图片/标注和约36.6MiB的Quantigo原始包，不新建62GiB RGB缓存；若S缓存READY可复用RGB引用，但新增Mask标签另接。全部未来主线结果直接G，D/G各100GiB保留不变；精确总预测/检查点/时间需实测后逐块冻结，当前不承诺耗时、容量PASS或新有限作业许可，不延既有全局/缓存截止。正式评价主指标候选标准COCO Mask AP50:95/AP50，并按3个clip和原图Mask面积辅助拆分；3片不当3比赛，未披露occluded全false不作分层，单seed/单外部比赛不构造虚假独立样本CI。进一步真正跨比赛test仍需新的公开来源，不默认人工补标。

来源：https://www.kaggle.com/datasets/ihelon/football-player-segmentation （v1）；https://huggingface.co/datasets/Voxel51/Football-Player-Segmentation/tree/4fd647b30d7504007544b9a2e007c0822a0d6e91；https://www.kaggle.com/datasets/quantigoai/soccer-dataset （v1）；https://www.kaggle.com/datasets/sadhliroomyprime/football-semantic-segmentation （v1）。JSON及文件清单通过公开Kaggle API匿名实际取得，记录上述精确SHA；没有把API内容失读/页面空行误称数据不可用。样图、叠加和3短片仅本轮临时诊断，展示核验后按精确名单清理；唯一实验日志保留本轮事实与候选。正式方案/代码/TODO均待直接讨论允许，不从本条启动训练。
收口核验：本轮19个临时诊断文件共37585170B已在展示后按绝对路径/精确名单删除，临时目录确认不存在；未递归清理或触及正式源/模型/缓存。唯一日志SEG-GT-001位于686行，EXPERIMENTS/PAPER末次SHA仍3814e515.../d48cb416...；没有新增训练待办或派发。


## FORMAL：R_to_R_seed17 COMPLETE

UTC 2026-11-02T10:59:40.912229+00:00。目的：合成数据真实迁移五策略对照；公开YOLO11s独立起点、seed17，SoccerNet train2100/val750，synthetic train24149，策略R->R。实际已提交36750/36750更新，R/S曝光588000/0；960/FP32/micro8/有效16/workers2，SGD Nesterov momentum.937、WD.0005，clip10。第一阶段30187步峰LR.01，真实终阶段6563步峰LR.001，各自cosine到.01倍，阶段重建SGD/EMA并承接raw；独立R仅终阶段、S仅第一阶段。检查点0/每500/阶段末完整状态，PT/每1000/阶段末750图原生预测与全部负结果保留。不取best、不读test；本轮结果尚需与同seed/其余seed对照，不能单轮宣布贡献。

状态/原始结果：{"state": "COMPLETE", "directory": "G:\\FutsalMOT_Training\\runs\\bbox_s_soccernet_seed17_core_20261102_a01\\R_to_R_seed17", "update": 36750, "exposures": {"R": 588000, "S": 0}, "terminal": {"tag": "step036750_final_final", "update": 6563, "main_metrics": {"AP50_95": 0.4665311479396385, "AP50": 0.899551370752138, "AP75": 0.42668455855582854, "AP_small": 0.26395451240516327, "AP_medium": 0.45783805925092397, "AP_large": 0.5448367483423882, "AR_max1": 0.051739742558326626, "AR_max10": 0.43326629123089305, "AR_max100": 0.5537208366854385, "AR_small": 0.29606299212598425, "AR_medium": 0.5532186732186732, "AR_large": 0.5994193548387097}, "raw_prediction_bytes": 4155373916, "images": 750, "wall_seconds": 328.54699999999866, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "phase": "final"}。证据：G:/FutsalMOT_Training/runs/bbox_s_soccernet_seed17_core_20261102_a01/R_to_R_seed17。


## R2R-ANALYSIS-001：真实控制块闭合分析与先缓存后S→R跟进更新

UTC 2026-11-02T11:10:34.247337+00:00。用户直接告知R→R完成，要求分析是否调整训练策略、更新定时任务，先完成缓存再继续训练。本轮为既有正式作业核验/分析，不是新增训练或重评；driver已有R→R COMPLETE记录不重写。原公开YOLO11s独立初始化、seed17，SoccerNet原train2100/val750，原冻结46aa...协议不变，test未读。960/FP32/micro8有效16/workers2，SGD Nesterov momentum.937、WD.0005、clip10；第一阶段30187步峰LR.01、终阶段6563步峰LR.001、阶段raw承接/重建SGD和EMA。R=6563更新/105008R/0S；R→R=36750/588000R/0S，额外真实曝光482992。R→R约为R的5.599573倍图像曝光，不能称曝光相同。

同scope/同750图末步EMA主标准COCO结果：R AP50:95=0.5165115374980012，AP50=0.9410995239648520，AP75=0.5056873088257927；RR AP50:95=0.4665311479396385，AP50=0.8995513707521380，AP75=0.4266845585558285。RR-R百分点差={"AP50_95": -4.998038955836265, "AP50": -4.154815321271399, "AP75": -7.900275026996417, "AP_small": -6.862026548925787, "AP_medium": -4.793801796327263, "AP_large": -3.870002451126242}。small/medium/large绝对AP：R=0.3325747778944211/0.5057760772141966/0.5835367728536506；RR=0.2639545124051633/0.4578380592509240/0.5448367483423882。公开起点主AP.38598711247659906；RR仍好于公开起点，但弱于较短R，负结果完整保留，不用中途高点替换末步。该750图是既有来源内val范围，非官方完整榜单，不读test，不给单seed/单比赛伪CI。

只读事前选定阶段小summary看趋势：RR first5000=.4911943687472756、first15000=.4949248693206473、first30187末=.4761025564896045；终阶段0=.46773971413431004、局部3000=.46688825703049386、局部6563末=.4665311479396385。首阶段末EMA到下一阶段0由raw重建EMA不是同一模型，.8362842355pp差不能误报状态损坏。终阶段从0到末约-.1208566195pp，未明显恢复。长时重复同一真实池未改善本seed验证，与过拟合/特定优化路径不利的解释一致，但未额外实验证明是过拟合、高LR还是EMA/阶段重建的独立因果；不能因RR下降推定S一定有效。

策略决定保持原已批准探索：不再追加R→R、不扩多seed/规格、不因为负分重训、早停、选best或事后缩短S预算。R50已选的终阶段6563步维持，唯一待做S→R仍30187S+6563R；这样才可同时给出SR-R（是否胜过简单真实基线）和SR-RR（同更新/总曝光额外阶段来源的对照）。若SR仅胜RR但未胜R，只能说明优于该长真实对照，不能宣称总体超过R；真正需改阶段/优化器/策略时先记候选，待用户明确允许新协议，不改封存方案。原新增Seg/Pose500步/公开足球GT仅待审规划，本轮不批准或派发。

闭合证据：D/G core/status均FORMAL_BLOCK_COMPLETE，R末6563、RR末36750，test_enabled=false。R15份完整checkpoint/8个成功val小收据，RR77份完整checkpoint/40个成功val小收据，末步checkpoint收据bytes=114295215、SHA9d241f80fa25c2c311ada05e2cd47388589272101dd33d2e71f788ee962bb66e（只读已有收据，未读取或重哈pt）。G control/CONTROL_MANIFEST=SEALED，scope control_snapshot_only、45文件623213B逐个小SHA通过，manifest SHAe670db1c27611b5ea8237924435efc2362600e0344541450fef0e1c06d77f866。实际进程创建时间/完整命令扫描确认原controller12244/wrapper26172和训练worker已退出，.research_gpu.lock不存在，无UE；当前同父组仅有可信cache_extend_c01 writer8952/create1793593821.227307及wrapper13940/create1793593821.220506，不能把合法CPU缓存写者误认为原训练未闭合。不搬G原生输出或全组hash，无新推理。

当前缓存小状态：2026-11-02T11:10:23.664336+00:00 BUILDING，9280/24149，复用6419，prefix核验完成；Real_priority={"paused": false, "reason": "real_block_complete", "state": "FORMAL_BLOCK_COMPLETE"}，Real闭合后原实现已自动停止退让，无需重启/热改缓存。原100000000B/s上限、单CPU低优先级/线程1/worker1、物理4GiB/commit8GiB、D100GiB/缓存128GiB与66h至2026-11-04T22:30:21.146105Z保持。原错误/partial/冻结证据不删，不自动提速150、多worker或另建续建。缓存不等于已READY；只有24149/16arrays与精确manifestSHA、writer/wrapper退出及两锁退休后才validate_ready_cache。当前未启动S预检/GPU训练。

已用原生automation_update更新同一bbox-r-r-r，名称“SSD缓存完成后继续S→R”，ACTIVE，每60分钟/原UNTIL2026-11-07T23:55Z/原目标聊天不变。以凝练现行提示取代旧重复历史，明确先缓存READY和退出/资源验收，再一次复核旧R/RR闭合、小控制和GPU互斥；使用现源formal_cache_preflight最多240秒，原115.765+214.235=330秒不重置，总≤570≤600。cached与direct像素/标签/增强、完整恢复/下一步数值、R阶段转换、实际吞吐与checkpoint大小必须实际通过。未来按已批准每阶段0/末4val、77cp重新核完整资源（候选110.18077GiB只是旧测量估计）、G另100GiB、D32MiB另100GiB、单新run≤16h且在全局截止前闭合。canonical新目录须查旧候选和同名任务避免重复，独立公开起点，34等必要源码/依赖/数据/权重/cache manifest实际冻结；全部证据直接G并封小控制，人工取消不自动恢复，错误有界不无限重试。S启动后用户人工观察训练，自动流程只做必要生命周期闭合；无变化安静，缓存完成/S启用/新失败或最终闭合才通知。完成当前三核心与所有证据闭合即删本跟进，不碰每日g/UE生产。

EXPERIMENTS/PAPER保持原SHA3814e515.../d48cb416...与原bytes；本轮仅追加本日志、更新TODO/README和AGENTS实际状态及同一原生定时任务，不修改数值源码/config或已冻run，不创建新训练组。原始日志、预测、全部模型和负结果继续原位保留。


## CACHE-ACCEL-001：直接授权的硬件优先加速续建，9501完整行原位保留

UTC 2026-11-02T11:37:33.130055+00:00。用户在真实R/RR闭合后明确要求“立刻开始投入所有硬件资源完成缓存任务，尽可能快地进入下一批训练”。这是新独立加速执行许可，覆盖旧c01单低优先级CPU/100MB/s/Real退让资源限制；不变更封存科学方案、GT、UID、split、LetterBox或曝光。目的仍为16 train seed/24149张S的960 uint8无损输入缓存约62.18GiB，不是训练、合成贡献结果或新数据版本。

诊断：本机14物理/20逻辑CPU，RAM约31.84GiB，检查时可用约19.18GiB、D约113.49GiB/G约812.19GiB。旧SourceReadThrottle在D临时16MiB CPU文件夹具实测无guard3.999277秒/4.195MB/s，有guard3.999590秒/4.195MB/s（4072次guard），显示主要耗时在每小块额外sleep而非资源查询；名义100MB/s不是实际吞吐。请求sleep81.92微秒实际mean515.82微秒，655.36微秒实际1034.23微秒。新串行大缓冲无sleep通道同16MiB夹具0.0059602秒，返回字节精确、peak open1；这是OS/SSD缓存夹具，不声称HDD达2.8GB/s或端到端加速数百倍。临时文件已自动清理。

旧c01在核真实PID/CT/cmd与两锁后，仅写本组STOP_REQUEST，正常落FAIL_STOP/error explicit cache-only STOP request，9501行完整保留；writer8952/create1793593821.227307及wrapper13940/create1793593821.220506均已实际退出，两锁退休证据保留。canonical v2.partial旧FAIL manifest SHAba08d581d70bf561cb2dd89520c9a6a981ea808cae48de21e1cb36f6847b9bdb，原数组/JSONL前缀不删、不复制62GiB。用户移交停止是已知受控原因，不伪称c01 COMPLETE或未知基础设施故障。

实现与核验：tools/cache_acceleration.py SHAba139b1c62633d8a1b1a2bbdd35713f6b1332579297e2470a1de8c28a67fc982，tests/test_cache_acceleration.py SHA818f24289810dc52a9727564695b343812453823c2c3175723f68b32a16fe4b3；根独立18项CPU PASS/0跳过0.110秒，覆盖有序有界预取、原序返回、异常传播与恢复open/资源检查。新8准备线程/max_pending8/各Torch和OpenCV线程1/NORMAL优先级、源open到close串行1MiB缓冲不限速，解码/原LetterBox并行，SSD单writer原32行映射flush-close-unmap及逐行write/read像素exact、metadata roundtrip/JSONL fsync均保留。每秒重guard，physical4GiB/commit8GiB、进程private8GiB上限、D100GiB与v1+v2合计128GiB不降，不初始化CUDA。根src完全不改；私有cache_resume_v2仅writer元数据CPU_workers由1改8并登记scheduler/max_pending，不修改像素/来源数值文件。

独立G同一父组/cache_accelerate_a01于2026-11-02T11:35:29Z隐藏派发，wrapper32536/create1793619329.479354，实际writer7504/create1793619329.517253，命令本组launch.py。launchSHAf94aba8bac28af007431c7606040c0627a81dd5e3d58d837514efdb82b6d5cd4，planSHA1cd5a198a1a097d5fb8a96b1d1f78e860b2fa16395a9a409d42ead8a56f65dc8，43文件code_bindings SHA94ed3fb4e0c5d3c7c9591ee616139aa8a910e9567078595fb5a49052ec2b4ad1已逐小文件SHA核验，launch compile通过。原截止2026-11-04T22:30:21.146105Z不延长/不重新66h；全局2026-11-07T23:55Z保持。原位9501前缀全核与3RGB抽样后才append剩14648行，未知尾拒绝，不重放。初始PREPARING_SOURCE_AND_PREFIX/9501、CUDAfalse、stderr0；此时未READY、未开始cached GPU预检/S正式训练。新任务失败/取消/到期仅具体报告，不自动开下一attempt。

已原生更新同一bbox-r-r-r为“SSD缓存加速完成后继续S→R”，ACTIVE/每60分钟/原全局截止/目标聊天保持；以新身份和资源策略替换旧c01活动及限速条款。未来仍仅原BBox YOLO11s/SoccerNet/seed17的S30187+R6563、每阶段0/末4val、77完整checkpoint、逐步证据保留；READY24149/16arrays、writer退出/两锁退休后才验收和最多240秒cached准入/完整存储时限Gate。R/RR不重跑；Seg/Pose未准入。EXPERIMENTS SHA3814e515...和PAPER d48cb416...前后保持，仅更新AGENTS/README/TODO事实并追加本唯一实验日志，不改原冻结数据或现行GPU数值协议。

本次启用后实测核验补充（同CACHE-ACCEL-001）：前缀核验通过，实际进入新增写入；2026-11-02T11:39:00Z BUILDING10240/24149，44.14秒新增352行=7.9746行/秒，源增量134.3665MB/s（十进制），进程private约1.014GiB/RSS约.653GiB、D余113.48GiB/stderr0，两锁与实际PID身份吻合。短窗预计余29.1分钟仅估计，不承诺整个余程。随后小状态测量已写本组initial_throughput_receipt.json：{"observed_at_utc": "2026-11-02T11:40:08.258682+00:00", "status_at_utc": "2026-11-02T11:40:04.469738+00:00", "state": "BUILDING", "rows": 10720, "interval_seconds": 107.89000000000402, "new_rows": 832, "rows_per_second": 7.711558068402716, "source_bytes_per_second": 135006132.72777325, "estimated_remaining_minutes_short_window": 29.02353385416775, "private_bytes": 1088507904, "RSS_bytes": 703291392, "stderr_bytes": 0}。独立只读审阅未见锁序死锁/有序单writer或像素标签语义阻断，8pending/源peak_open1/CUDAfalse实际吻合；单文件read及shutdown需等在途读返回，非即时中断。为响应尽快接续训练，同一原生定时任务缓存阶段暂改每5分钟静默检查，S实际启用后恢复原每60分钟；原UNTIL/总轮数/7天上界不变，不创建额外任务。尚未缓存READY，不启动GPU。


## CACHE-READY-001：24149行16seed无损SSD缓存完整完成与实际验收

UTC 2026-11-02T12:20:30.522366+00:00。目的仍是保持合成train24149输入/UID/GT/split/原960 LetterBox不变的I/O加速，不是训练或真实迁移评分。cache_accelerate_a01于2026-11-02T12:13:54.683505Z READY，原位复用9501行并新增14648行，共24149/24149、16NPY，路径D:/projects/FutsalMOT_SyntheticDatasetResearch/cache/synthetic_bbox_960_v2；总66958650006B，manifest SHA fce4e02611f198ba1e72fe1a5ae47ebf047f83a29b2738ef88a0d1429001b324。完整缓存画像NPY66767157248B，其他为索引/血缘/恢复收据。此次_resume_pack耗时2298.985秒/38.32分，源读取270602501876B/read_wall2069.604秒，源文件峰并发1，映射峰88473600B，进程观测RSS峰739983360B约.689GiB、CUDAfalse、post_publish_progress_error null、stderr0。8有界准备线程、无旧sleep限速、原单writer/逐像素write/read/标注roundtrip/JSONL前缀读回全部通过。与旧速度对照只能说明执行吞吐改善，不是模型效果。

root实际检查writer7504/create1793619329.517253与wrapper32536/create1793619329.479354已退出、全局/per-cache自有锁已正常退休，16个数组名与READY manifest/sourcebinding一致；已调用现行validate_ready_cache通过count24149/完整sourcebinding，不重复4K解码或全NPY hash。原r01/r02/c01/v1失败/STOP/冻包/日志保留，不把旧失败改成成功。缓存预算提前闭合，无需延长11月4日22:30:21Z界限。

## CACHED-PRECHECK-001：首次零CUDA更新拒绝与一次JSON序列化比较修订

root使用本READY spec，仅在R/RR小控制SEALED、真实退出/GPU锁退休及生产closed后，启动main_bbox_yolo11s_soccernet_S2R_seed17_20261102_a01独立cached预检。首次于2026-11-02T12:16:41Z派发wrapper11176，7.203秒/diagnostic_updates0在setup_cuda前FAIL_STOP：与原S准入来源不一致 real_panel。实际根对比显示60处tuple/list类型差异，无数值/UID/文件身份/digest差异；runner.canonical(old)==canonical(current)精确true。原因旧JSON落盘将tuple转list，cached新检查用Python直接!=，而formal.prepare_group现有检查本来采用canonical。旧a01 admission SHA5f87654d35680f941d15c7d4c4f4d112bf43773c7f8a23d4d491bff4b8855c33、原错误/helper/stdout留存，GPU锁finally退休。wrapper已结束未能事后取得childCT，不编造其PID身份；原Start-Process真实11176/start12:16:41.3058005Z保留。

唯一有界机械修订：formal_cache_preflight.py将来源组与public起点绑定统一用现有runner.canonical比较，保持所有键/值/身份/digest/UID顺序精确；未忽略字段、未放宽任何数值门槛。新增JSON roundtrip及真实digest/identity/UID顺序/字段缺失拒绝回归，根13项CPU PASS/0跳过2.387秒。新文件SHA3efdfac4103eb55540b732d887856e546500950ea2390359c3344b0c9941c8d3；formal.py/runner.py及网络/损失/优化器/数据/增强/精度模块不改，不改EXPERIMENTS/PAPER。新a02仅由于a01明确失败的一次修订，不是增加正式实验或猜重放。a02实际于2026-11-02T12:18:45.3464873Z派发，wrapper7752/create1793621925.3464873，actual7140/create1793621925.3735356，helper cached_S2R_preflight_r01.py SHA9c4330e13d8fbfa70dc1c3630906102e3c32466fb08565daf338aa27a642bda8，准确GPU锁属于本a02。旧330秒不重置，首次7.203秒计入，修订预检最多232秒，合计569.203≤570≤600秒。诊断checkpoint展示后由现TemporaryDirectory精确清理；a02二次若失败保留全部证据报告，不自动第三次预检/a03。此时尚未PASS或正式S更新，不冒称已训练。原生同一跟进已保存新READY/失败/修订活身份，暂5分钟静默，正式启用后恢复60分钟，原总截止保持。


## FORMAL：S_to_R_seed17 STARTED

UTC 2026-11-02T12:22:16.502290+00:00。目的：合成数据真实迁移五策略对照；公开YOLO11s独立起点、seed17，SoccerNet train2100/val750，synthetic train24149，策略S->R。实际已提交0/36750更新，R/S曝光0/0；960/FP32/micro8/有效16/workers2，SGD Nesterov momentum.937、WD.0005，clip10。第一阶段30187步峰LR.01，真实终阶段6563步峰LR.001，各自cosine到.01倍，阶段重建SGD/EMA并承接raw；独立R仅终阶段、S仅第一阶段。检查点0/每500/阶段末完整状态，每阶段0/阶段末750图原生预测与全部负结果保留。不取best、不读test；本轮结果尚需与同seed/其余seed对照，不能单轮宣布贡献。

状态/原始结果：{"state": "public_initialized_zero_updates"}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_S2R_seed17_20261102_a02/S_to_R_seed17。


## CACHED-PRECHECK-PASS-001：限定修订通过，唯一正式S→R a02准入并启用

UTC 2026-11-02T12:25:00.471618+00:00。原a01序列化误拒绝的FAIL_STOP与7.203秒/零更新证据保留；唯一a02修订预检PASS，实耗52.266秒/8诊断更新，原330+7.203+52.266=389.469秒总准备，不重置600秒/限定上界。9个C5/C1/C2/C3/C4/P02/P04/P03/P01代表请求cached/direct像素/标签/几何/增强血缘逐值一致；完整raw/EMA/SGD/CPU CUDA RNG/sampler/loader generator/创建边界恢复全部exact，下一更新loss与clip前梯度exact；S→R raw承接/新optimizer空state/EMA0等全部通过。S四个warm步.453/.531/.469/.469秒，均值.4805秒（包括缓存读取/增强/传输/反传/optimizer/sync，workers2/prefetch1），此前直读3.2070625秒是不同短窗诊断对照，不保证正式全程倍速。GPU peakallocated8392781824B/peakreserved8954839040B/free6753878016B，均保留1GiB要求。4个诊断临时完整pt共419233723B已显示、TemporaryDirectory精确清理并根确认无.cached-preflight目录，未删除正式/失败模型。actual7140/wrapper7752已退出、锁退休后才prepare/正式派发。

实际资源以新checkpoint最大114285487B重新计算，完整新增110.91485648974776GiB（非旧候选110.18077），storage_revision.json保留各绝对部分：77cp按cp*1.1上取、36750步日志2MiB上界、4*750图native6650344B+JSON1MiB/图、一次750图partial、瞬态3GiB+2cp；G另100GiB，D控制32MiB另100GiB通过，cache占用和旧v1失败照算。S.4805/R旧mean.48357推36750训练17678.52341秒，以2倍35357.04682+2400验证+770checkpoint+600启动/边界余量=39127.04682秒，冻结max_run_seconds43200=12小时，不随恢复重置，仍受原全局2026-11-07T23:55Z。未减少更新/日志/完整cp、未删除旧预测或失败绕Gate。

使用canonical_group_name实际at与attempt2生成main_bbox_yolo11s_soccernet_S2R_seed17_20261102_a02；a01有明确零正式更新失败，a02是唯一有界机械修订，不增加第二个正式job。cached admission/executed_code/requirements/公开YOLO11s权重/真实与合成面板/cachemanifest均实际绑定冻结。D预检原preflight_steps.jsonl先保留根单一精确helper，prepare_group创建双端RUN_LOCATION与G初始control绑定后，将原bytes写G/preflight_steps.jsonl且SHA读回，再精确释放D原文件与helper，准入admission不改。code_bindings根实际核全部39模块SHA，源formal.py仍d0173fe14d3a0072add7da62d18cdc13e51ff57fb352254c223600b06dbbdcfb、cached_preflight3efdfac4103eb55540b732d887856e546500950ea2390359c3344b0c9941c8d3，其他网络/损失/优化/数据/增强/runner数值原bytes不改。

唯一正式训练于2026-11-02T12:22:04.1431321Z隐藏启动，wrapper32292/create1793622124.143132，controller25232/create1793622124.17863，完整命令D本a02组launch.py；launchSHAb47fddd9d1a7c16d02f25fa3cc5c9de4b03d648430366874141a73c9b1c11d29，job_tableSHA40a4a3dedb5588773c8e171cb27a92a536f073c9b4fd67b9908eec1813dcdbdf，GPU锁和真实CT+cmd吻合，stderr0。初始RUNNING/first/update0是阶段0公开起点评价，不是已有参数更新；driver实际FORMAL STARTED见其原条目，不重复造正式RUN。原独立公开YOLO11s/seed17/SoccerNet2100train750val/S24149、960FP32 micro8有效16workers2，S30187/482992S后R6563/105008R，总36750/588000曝光，SGD/loss/增强/阶段raw+新SGD/EMA不变，阶段0/每500/末77完整cp、每步证据、阶段0/末共4val与原生密集预测保留。全部大证据直接G同名输出，不搬整个run；结束须G末步/4val/77cp/小control SEALED/hash与owned退出/GPU锁退休闭合后才报告SR-R与SR-RR，不提前宣称合成贡献。

原生同一bbox-r-r-r已恢复每60分钟并更新活身份/唯一a02/12h/缓存与准入完成事实，原全局UNTIL/目标聊天/总轮数与静默规则保持；训练由用户观察，agent只必要生命周期闭合。人工取消不恢复、失败不自动新job或第三预检；Seg/Pose/其他seed规格/test未准入。EXPERIMENTS/PAPER前后仍3814e515.../d48cb416...封存，无新方案副本；当前仅更新事实AGENTS/README/TODO与本唯一实验日志。


## USER-PAUSE-001 PREP：用户要求22:00受控退出，明天通知后再恢复

UTC 2026-11-02T12:50:25.719671+00:00。用户直接要求本轮今天22:00退出、明天继续，并在时间澄清中明确选择“明天由我通知再恢复”。本轮只安排2026-11-02T14:00:00Z（北京时间22:00）暂停当前唯一S→R，不自动明天恢复、不新增训练。训练原独立公开YOLO11s/seed17/960FP32微8有效16workers2，30187S+6563R/总36750更新与482992S+105008R、4阶段端点val/0每500末完整cp和每步证据均保持；暂停额外边界cp保留，不减原定训练预算或原始模型/预测。

已读冻结执行器：STOP.json由每步开始与验证每批检查；训练更新边界_save_boundary(user_stop)完整保存raw/EMA/SGD/RNG/采样/loader，验证中STOP走受控恢复RNG后保存controlled_failure并保留原partial，不强杀。原controller finally在run_job worker退出与GPU锁退休后封G唯一control小清单。没有热改活动code/config/job_table/run或原runtime_limit，D控制STOP尚未写入（等待指定时间）。新有界stop_at.py位于G本a02/pause_20261102_2200，--check实际controller/wrapper alive PASS、target正确、automatic_resume=false；helper仅CPU读取小状态、到时核原PIDCT完整cmd/精确GPU锁/launch与job_table SHA，才写一次自有D STOP。等待与验收上界14:10Z，核末step/末cp/update/tag/bytes、已知owned子进程/controller/wrapper退出、GPU锁退休与G control SEALED逐小hash；不读模型/NPZ全哈、强杀或清未知锁。完成final.json为PAUSED_VERIFIED或ATTENTION；守护会追加本唯一日志FINAL，根不重复写。

实际隐藏派发2026-11-02T12:47:58.9313983Z，wrapper3308/create1793623678.9313984，watcher31880/create1793623678.9703915，ARMED at12:47:59.078195Z，stderr0；完整命令本pause目录stop_at.py。实际文件helperSHA95b4c5b6ea34179c86788a664840fd4afa569c4b2c4c95ec49358180958070f5，planSHA668a10d68f54e6585bfd0fe64841d60de754143b0c11ed19c54e6e46adf27034，runtime_identity/dispatch/armed保留，不重复派发。它不持GPU锁/不初始化CUDA，关闭Codex后独立进程寿命仍未验证，未编造保证。同一原生bbox-r-r-r改为整点小时跟进、22:00可核收口补位；若守护缺失且到时同一训练身份仍活动，可按本授权写STOP，不能误操作新PID。确认暂停后将本跟进PAUSED，等待明天用户通知，不自动恢复；每日g不动。

原runtime_limit固定wall截止2026-11-03T00:22:13.211953Z（北京时间11月3日08:22:13）当前保留。暂停跨夜后不能原封重派已封controller或改旧deadline冒充预算未过；明天收到直接恢复指令再核可信停点与原剩余预算，独立审阅有界恢复兼容并冻结证据，保持原更新/曝光/数值/全局Nov7截止，不自动重置12h。当前训练仍活，本文PREP不冒充已退出/已完整保存停点；真正停点曝光由22:00实际收据/driver USER_STOP登记。EXPERIMENTS/PAPER仍3814.../d48cb...封存。


## FORMAL：S_to_R_seed17 USER_STOP

UTC 2026-11-02T14:00:01.449306+00:00。目的：合成数据真实迁移五策略对照；公开YOLO11s独立起点、seed17，SoccerNet train2100/val750，synthetic train24149，策略S->R。实际已提交9774/36750更新，R/S曝光0/156384；960/FP32/micro8/有效16/workers2，SGD Nesterov momentum.937、WD.0005，clip10。第一阶段30187步峰LR.01，真实终阶段6563步峰LR.001，各自cosine到.01倍，阶段重建SGD/EMA并承接raw；独立R仅终阶段、S仅第一阶段。检查点0/每500/阶段末完整状态，每阶段0/阶段末750图原生预测与全部负结果保留。不取best、不读test；本轮结果尚需与同seed/其余seed对照，不能单轮宣布贡献。

状态/原始结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_S2R_seed17_20261102_a02/S_to_R_seed17。


## USER-PAUSE-001 FINAL：22:00受控暂停验收

{"state": "PAUSED_VERIFIED", "observed_at_utc": "2026-11-02T14:00:04.047280+00:00", "update": 9774, "phase": "first", "R_exposures": 0, "S_exposures": 156384, "checkpoint": {"kind": "checkpoint", "update": 9774, "phase": "first", "phase_update": 9774, "tag": "user_stop", "receipt": {"path": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_S2R_seed17_20261102_a02\\S_to_R_seed17\\checkpoints\\009774_first_009774_user_stop.pt", "sha256": "9026ad28a134a56ee55da2c914dc9a98d4316a02a5e53740fcc88247ce24f17c", "bytes": 114292975, "source_bindings_sha256": "e50e439ee63445dac24bf28917c8f2a6f044c18fc74c184a070e6d93987bc52e", "schema_version": "futsalmot_checkpoint_v1", "producer": "futsalmot.records/v1"}, "saved_at_utc": "2026-11-02T14:00:01.361007+00:00"}, "control_manifest_sha256": "3d53cf75673331419298edf26fa3852c14057a3b2f7812e732a4d49463daace3", "all_known_owned_exited": true, "GPU_lock_retired": true, "automatic_resume": false, "runtime_limit_unchanged": true}。用户要求明天通知后才恢复；非训练失败，不重置预算，不删STOP/清单/模型。恢复须另审可信检查点与原wall deadline兼容，不自动派发。

USER-PAUSE-001 VERIFY（UTC 2026-11-02T14:02:37.824517+00:00）：守护已写FINAL与driver USER_STOP，根不重复造RUN记录。独立只读核末step/checkpoint/status精确9774、R0/S156384，cp tag user_stop与bytes114292975；六owned真实CT均退出、GPU锁不存在，D/G group USER_STOP；G control SEALED/scope control_snapshot_only，49文件747946B逐小SHA通过，manifestSHA3d53cf75673331419298edf26fa3852c14057a3b2f7812e732a4d49463daace3。未读或重哈pt/NPZ。原剩余first20413+final6563=26976更新，完整暂停边界cp额外保留；非失败/非COMPLETE。原生同一bbox-r-r-r已实际设PAUSED，等待用户明天直接通知，不自动GPU预检/恢复/新缓存，不改每日g。README/TODO/AGENTS已改真实暂停事实；原fixed wall截止/源code/config/所有证据和封存EXPERIMENTS3814.../PAPERd48cb...不动。未来恢复必须先审可信9774检查点与全日志提交一致、原剩余预算和跨夜wall兼容，不原封重派已封controller或自动重置12h。


## S2R-RESUME-001：用户通知后从9774完整停点显式续训，夜间暂停兼容

UTC 2026-11-03T00:01:15.711100+00:00。用户2026-11-03直接“继续训练”，覆盖昨晚等待通知/PAUSED状态；本轮只恢复已授权唯一YOLO11s BBox/SoccerNet/seed17 S→R，不新增正式job或GPU数值预检。从公开起点独立开始的原策略保持S train24149、SoccerNet train2100/val750、960FP32 micro8有效16workers2，首阶段30187/482992S后终阶段6563/105008R，总36750/588000曝光，SGD Nesterov momentum.937/WD.0005/clip10、首峰LR.01与终.001/cosine至.01倍、阶段raw承接并重建SGD EMA不变。4阶段端点val和0/每500/末77计划完整cp、逐步loss/LR/梯度/UID/曝光/全部预测与失败仍保留，额外user_stop边界亦保留，完成预计78cp而非删边界凑77；不best/test。

恢复前根只读核旧control SEALED49文件747946B逐小SHA、manifest3d53cf75673331419298edf26fa3852c14057a3b2f7812e732a4d49463daace3及旧owned退出/GPU锁不存在、production_closed。实际按原可信CheckpointReceipt SHA9026ad28a134a56ee55da2c914dc9a98d4316a02a5e53740fcc88247ce24f17c加载唯一114292975B的009774_first_009774_user_stop.pt并核全部绑定/first/phase9774/完整raw EMA SGD RNG sampler loader generator与iterator边界。全steps497286337B已逐行核9774连续更新、每步S域/R0/S16*i，末156384S精确；文件dev/ino/size/mtime稳定，无未知半行/重放或回滚。第一次CPU审计把访问时间也纳入stat相等而触发断言，读取本身可更新atime；改以稳定身份字段后全提交审计PASS，非训练或数据内容故障，没有模型更新。checkpoint字段实际扁平extra状态已按真实schema核，不猜签名。39原模块SHA、原公开基座/GT/依赖/source/cachemanifest与封存计划原bytes保持。

独立恢复预算审阅：原runtime起2026-11-02T12:22:13.211953Z到暂停完全闭合14:00:04.047280Z，保守计已用5870.8353271484375秒，12h43200秒仅剩37329.16467285156秒。用户要求跨夜暂停后继续，所以夜间空闲排除活动时间，本次恢复绝对界限在计划准备时即冻结2026-11-03T10:19:31.342213Z（北京时间18:19:31），准备/启动也耗此剩余预算，不重新给满12h、不随后续恢复重置，原全局Nov7界限仍保持。旧runtime_limit.json原fixed wall截止08:22:13和SHA df4d0b5bba4d9c72760b43eb269425a96ea75496e8f51d7710b0fa419553a8e6不改；原D STOP SHA10eaba37df692af8edd128907be6b67b722cbe5f3898d838408a707554033826与旧SEALED49文件全部保留，不编造旧wall未过或覆盖清单。重核完整原110.91485649GiB新增保守上界+G100GiB、D32MiB+100GiB通过（D121743831040B/G864820039680B），不删除旧失败/日志/检查点绕Gate。

独立机械恢复实现位于G同组/recovery_r01：原futsalmot模块39份直接从D冻包导入、原源码/配置/run.json/依赖不改。run_job_resume.py SHAd359b1aa59099140c2f008bc258cc00a84d0e68df18a6aac6302f26f7431058a仅将原run_job恢复时读deadline一行替换为显式剩余预算函数；AST逆替换与原函数完全相同，原训练循环/restore/增强/优化/评价/记录和检查点计算不变，函数用原globals执行原_verify_bindings核原39模块。lifecycle.py SHAeb7c8ab55779d36bc2c9832c404e0b6b5ecade038e71ea2161452fb646bde7d3仅核原runtime、同可信收据及原活动预算，承认精确原bytes的旧暂停STOP，响应新的G recovery_r01/STOP.json或G输出根STOP；任何历史STOP缺失/变化、错误group/收据/过期预算都拒绝。9项独立CPU PASS/0skip0.053秒（含数学改动拒绝、原bytes保留、新STOP生效等），临时fixture已清，没有GPU玩具或增加预检389.469秒已用准备预算。

唯一恢复launch.py SHA85b73a91ca8bddb2f327a8cadd6e19fb12ea86f6b1226e22720988063203998f，resume_plan SHA9d89c581103b188a354107228a731a831643a57b8c361af0e669b9856e0dd995。2026-11-02T23:58:00.8832232Z（北京时间11月3日07:58）隐藏派发，wrapper25532/create1793663880.8832233，actual10160/create1793663880.9188263，命令G recovery_r01/launch.py，准确GPU锁仍group原D a02。原D dispatch/launch为历史不改，当前恢复以recovery dispatch/runtime_identity/CT+cmd为准。stdout/stderr新G recovery目录保留，旧controller日志不覆盖；process_lifetime_isolation_verified=false，不保证Codex重启完全隔离。实际恢复后RUNNING/first从9775首次接续，随后9800/9825/9850已提交、stderr0，不把初始化旧USER_STOP9774状态误判失败；原9774更新不重复。{"verified_status": {"state": "RUNNING", "pid": 10160, "create_time": 1793663880.9188263, "update": 10000, "updates": 36750, "phase": "first", "phase_update": 10000, "loss": 1.3151705265045166, "updated_at_utc": "2026-11-03T00:01:12.348944+00:00"}}

结束后原run_job finally退出worker、GPU锁退休，再仅将当前D小控制与独立恢复plan/code/dispatch/status/result封G recovery_r01/control/CONTROL_MANIFEST（scope control_snapshot_only，≤32MiB，旧manifest链）；原G根control暂停清单原bytes不动，不再调用旧capture_control覆盖。全部模型/预测/逐步记录直接原G位置，不搬整个组/全哈大证据。若恢复失败/取消/到界/控制seal失败，保留具体证据，不自动r02或重训；最终必须末36750/R105008/S482992、4val/全部cp小收据、新恢复control小hash/旧清单SHA不变、owned真实退出/锁退休后才闭合并对比R/RR/SR。原生同一bbox-r-r-r已恢复ACTIVE/整点每60分钟、原总轮数/7天/Nov7 UNTIL不变，正常中间态静默，用户人工观察训练，仅必要生命周期跟进，不重启旧22:00停止守护。EXPERIMENTS3814.../PAPERd48cb...前后不改，只更新事实AGENTS/README/TODO并追加本唯一实验日志。


## FORMAL：S_to_R_seed17 COMPLETE

UTC 2026-11-03T04:37:51.744740+00:00。目的：合成数据真实迁移五策略对照；公开YOLO11s独立起点、seed17，SoccerNet train2100/val750，synthetic train24149，策略S->R。实际已提交36750/36750更新，R/S曝光105008/482992；960/FP32/micro8/有效16/workers2，SGD Nesterov momentum.937、WD.0005，clip10。第一阶段30187步峰LR.01，真实终阶段6563步峰LR.001，各自cosine到.01倍，阶段重建SGD/EMA并承接raw；独立R仅终阶段、S仅第一阶段。检查点0/每500/阶段末完整状态，每阶段0/阶段末750图原生预测与全部负结果保留。不取best、不读test；本轮结果尚需与同seed/其余seed对照，不能单轮宣布贡献。

状态/原始结果：{"state": "COMPLETE", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_S2R_seed17_20261102_a02\\S_to_R_seed17", "update": 36750, "exposures": {"R": 105008, "S": 482992}, "terminal": {"tag": "step036750_final_final", "update": 6563, "main_metrics": {"AP50_95": 0.485407098793604, "AP50": 0.9154250233531986, "AP75": 0.4531668334274502, "AP_small": 0.3091567701594757, "AP_medium": 0.4747594983177036, "AP_large": 0.5675745270598338, "AR_max1": 0.05189058728881737, "AR_max10": 0.4310740144810941, "AR_max100": 0.5773028962188255, "AR_small": 0.35669291338582676, "AR_medium": 0.575921375921376, "AR_large": 0.6218709677419355}, "raw_prediction_bytes": 4157525361, "images": 750, "wall_seconds": 347.25, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "phase": "final"}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_S2R_seed17_20261102_a02/S_to_R_seed17。


## S2R-RESULT-001：三核心末步配对比较与有限流程终验

UTC 2026-11-03T04:42:32.639101+00:00。用户要求查看已完全完成的训练结果。本轮为既有正式run终验/结果分析，没有新增训练、推理、模型选择或test读取。driver已有FORMAL S_to_R_seed17 COMPLETE记录，不重复造RUN。正式S→R于2026-11-03T04:37:51.670324Z（北京时间12:37:51）COMPLETE/36750更新，原first30187/482992S后final6563/105008R；独立公开YOLO11s/seed17、SoccerNet train2100/val750、S24149、960FP32 micro8有效16workers2，原SGD Nesterov momentum.937/WD.0005/clip10、首.01/终.001 LRpeak各cosine至.01倍、raw承接/阶段新SGD EMA保持。9774受控暂停完整恢复后从9775继续，无重复旧更新；原4端点val/每500完整cp/逐步loss LR梯度UID曝光/完整raw EMA SGD RNG sampler loader与所有预测/失败全保留。

仅比较相同有序750张真实SoccerNet验证集、同seed的末步EMA标准COCO AP50:95（不是全官方数据集benchmark、不是best）：

| 策略 | 总更新 | 实际R曝光 | 实际S曝光 | AP50:95 | AP50 | AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R | 6563 | 105008 | 0 | 0.5165115374980012 | 0.941099523964852 | 0.5056873088257927 |
| R→R | 36750 | 588000 | 0 | 0.4665311479396385 | 0.899551370752138 | 0.42668455855582854 |
| S→R | 36750 | 105008 | 482992 | 0.485407098793604 | 0.9154250233531986 | 0.4531668334274502 |

SR-R主AP=-3.1104438704397164个百分点，SR-RR=+1.8875950853965484个百分点。当前额外482992合成曝光并未超越同真实曝光的直接R；与等总36750步的RR相比，S首阶段后微调高1.8876pp，最多支持本配对下相对长真实对照更好，不能忽略更强R基线或据此宣称普适合成收益。R仍是本轮最高结果。所有尺寸AP也均SR<R：small .3091567701594757对R .33257477789442114，medium .4747594983177036对.5057760772141966，large .5675745270598338对.5835367728536506；SR均高于RR对应指标。没有基于验证结果挑best、删负结果或自动改预算追分。

阶段证据（均同750真实验证）：公开PT0 EMA主AP.38598711247659906；S末step030187_first_final EMA .1911377373350205；R阶段0 step030187_final_initial从raw重建EMA .12135544109479035；最终step036750_final_final阶段EMA .485407098793604。S末EMA与R阶段0raw重建EMA是不同模型状态，不将两值差称新增更新的退化。合成训练loss较低而真实验证AP明显下降，提示源域拟合不能代表真实迁移；域差异/长预训练遗忘等仅候选解释，当前数据不足以定位因果。真实微调恢复到48.54%，仍低于直接R51.65%。若后续继续，应另讨论有限更短S阶段或数据/增强匹配方案；本轮只分析，不修改封存方案或自动启动搜索。

可比性与局限：根核三策略real_panel绑定canonical相同，AP scope/750有序UID/真实GT一致。RR和SR同总更新及总曝光，R和SR同终阶段真实曝光，但R总计算更少；源数据分布不同，不能由训练loss高低跨域选优。原R/RR保留旧8/40val频率，SR按用户新批准4端点val，评价不更新模型、RNG保护和缓存输入等价均已有准入证据；仍需如实保留执行版本差异。仅seed17、单验证比赛面板，验证集已用于预算校准与当前分析，不是独立盲test；没有多seed方差/CI或其他真实来源，不能据本单点证明稳定增益或否定所有合成训练策略。当前n/m/l/x、多seed、Seg/Pose候选均未自动启用。

终验已通过：G run/status COMPLETE36750与R105008/S482992；4个精确冻结tag各summary750图/阶段EMA，末val原生预测4157525361B、wall347.25秒，全部评价原生预测/JSON保留。checkpoints.jsonl共78个唯一完整状态（77原计划+9774用户暂停边界），逐个只stat既有收据bytes，末global36750/phasefinal/local6563/tagfinal，路径checkpoints/036750_final_006563_final.pt、114292975B，已有收据SHAef4c92fbee2823965f12ddb7de7794293dd36293042e146bfdcbb80c4cf06489；不读取/重哈大pt或NPZ。D/G group与recovery均FORMAL_BLOCK_COMPLETE，recovery failure/seal_failure不存在；actual10160/create1793663880.9188263与wrapper25532/create1793663880.8832233真实退出，扫描未发现对应launcher或parent10160的残留worker，GPU锁不存在、unknown writers=[]。

原暂停G control SEALED49文件747946B逐小SHA通过，manifestSHA3d53cf75673331419298edf26fa3852c14057a3b2f7812e732a4d49463daace3未变；恢复世代G recovery_r01/control SEALED59文件781422B逐小SHA通过，manifestSHA1b654b55613d10919bb9a350cf62cf498da0cd6da077dd707e6f4abdd50fae47，scope control_snapshot_only，旧manifest链保留。不调用旧capture_control覆盖原已封清单，不搬整个G原生输出、不作全组hash、不把原位证据叫迁移发布。缓存READY及全部旧失败/冻结/日志/模型/预测仍原位留存。

至此用户授权seed17 R、RR、cached SR三核心+缓存+小控制/owned退出全部闭合；更新TODO完成与README/AGENTS真实结果，并删除同一bbox-r-r-r跟进，保留每日g，不再自动开新任务。EXPERIMENTS/PAPER仍3814e515.../d48cb416...封存，前后摘要不变，未修改科学方案/摘要或增报告树。


## SYNTH-QUANT-001：合成数据效用的量化框架（研究建议，未新增训练许可）

用户要求调研可量化标准。没有发现可跨任务套用的“合成数据质量合格分数”或固定AP涨幅。建议把监督正确性、任务迁移效用、标注效率、泛化和成本分开报告，不合成一个任意加权总分。以下是本项目建议，不冒称统一国际标准，不新设统计显著/非劣效通过线。

1. 核心任务指标沿官方COCO：BBox用bbox IoU AP50:.05:.95，Seg用segm mask IoU AP，Pose仅在真实关节顺序/可见性/面积/sigma契约匹配COCO17时用OKS AP；AP50/AP75/面积分层辅助。先列每臂绝对值，再给Δ_R=100*(AP_SR-AP_R)和Δ_matched=100*(AP_SR-AP_RR)百分点。两者分别回答同真实标注池是否获益、同额外更新预算是否优于继续真实训练。单独S后真实val性能可作直接迁移诊断，但公开模型本身已真实预训练，不能称从零无真实监督。
2. 当前实际R51.6511537498%、长RR46.65311479396%、长SR48.5407098794%，Δ_R=-3.11044387044pp、Δ_matched=+1.88759508540pp。因此现有结果只支持“同长预算下比RR好”，不支持“加入合成比强R更好”。当前A检验缩短合成前缀，若触发B才有短预算匹配归因；复用旧3000步是开发分支，后续独立确认不能用它替代新公开起点重复。
3. 标注效率候选：固定真实val/任务/训练选择规则，以唯一区别的真实已标注图像或比赛组数N为横轴，建立预先固定、嵌套训练子集上的R与SR学习曲线。在预先选定目标AP q（或用户另定非劣效容差）且实际达到时，报告L(q)=1-N_SR(q)/N_R(q)，同时给真实人物实例/标注关节数及各子集来源。不能用重复曝光105008或48000代替独立标注量；当前A/R都用2100真实图，尚无节省标注证据。不外推未测区间，不因未达q改q；这些曲线是后续待审实验，不本轮自动扩矩阵。
4. 可信度与泛化候选：所有臂同一真实评价集、同训练seed配对，后续若获准多seed报告绝对值/均值/标准差/逐seed差；新增未见比赛只在方法冻结后做一次独立test。样本区间按比赛成簇配对重抽并重算AP；只有单场时可做连续片段条件区间但不能称比赛级泛化CI，不能把750相邻帧当750独立比赛。建议报告95%区间及实际簇数，不擅自设p阈值或把开发A>R当显著性。训练seed不代表新比赛，多个clip也不自动独立。
5. 辅助诊断与成本：GT结构/像素对齐、小物体/人物密度/尺度/长宽比/视角/光照覆盖、失败率可以定位差异；FID/KID或视觉逼真度不替代真实任务收益。报告GPU训练与验证wall、继承前缀原始成本、缓存/数据生成wall、标注预算、存储与失败成本；复用开发新增R成本单列，不宣称继承S免费。当前不额外跑FID网络或GPU诊断，不把数据通过像素门禁等同任务贡献。

依据与适用限制：官方COCO API直接定义bbox/segm/keypoints和阈值（https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py）；Task2Sim CVPR2022以真实下游表现评价合成预训练，强调不同下游偏好不同模拟配置（https://arxiv.org/abs/2112.00054）；Ruis等2024针对目标检测讨论合成域过拟合与保留真实预训练特征（https://arxiv.org/abs/2405.19822）；Tang/Jia CVPR2023研究合成数据的监督训练/域适配效用（https://github.com/huitangtang/On_the_Utility_of_Synthetic_Data）。这些文献不是本项目因果诊断，不从其它数据集转抄性能门限。上述框架先供审阅，PAPER/EXPERIMENTS不因研究自动改动。

## SEG-POSE-PLAN-002：BBox A启用后继续细化足球Seg/Pose最小方案（待审，不训练）

Seg沿SEG-GT-001实文件核验：ihelon512图/7686person多边形COCO实例候选作真实训练池；Quantigo847帧/6325 Player+Referee人物实例作外部足球val候选。后者3片保守视为同场尤文–皇马，前者抽样支持另一比赛但完整比赛身份证据仍缺；先核源/文件名/场景重叠、许可链、实例语义与可见边界，再冻结来源和派生映射。两来源都没有官方split。不能随机逐帧80/20、用存储分片充当split，不能把未知标注生成过程称人工精修或把occluded全false当遮挡GT。原多边形保留，Quantigo Player/Referee→person须明确审定；未标看台人/角色语义是局限，不能临时补ignore让AP上涨。

Seg建议的小预算开发设计仍三臂：公开yolo11s-seg独立起点、seed17，R500（8000R）、RR500+500（16000R）、SR500S+500R（8000S+8000R），合计2500新更新/32000R+8000S；S候选为16train seed各100整帧=1600，按事前机位时间覆盖与原UID冻结，不从val挑seed，不把等效重复轮次叫新图。此预算只是查迁移方向，不声称500充分收敛。R512/ihelon与外部847/Quantigo面板先冻结，不动SoccerNet BBox或其test。

绝对数值候选补齐供审：960整图/FP32/micro1累积16/workers1，所有可训练层、保留公开COCO80分类/原mask头、target person class0；SGD Nesterov momentum.937/WD.0005/clip10，所有500步阶段LRpeak.001，前50步线性.0001→.001、随后450步cosine至.00001，bias不另设.1暖启动；阶段承接raw、新SGD/EMA。loss候选按锁定Ultralytics8.3.237源码box7.5/seg7.5/cls.5/dfl1.5；mask_ratio2、overlap_mask=False（独立实例二值mask保留断开/孔洞）。HSV .015/.4/.4，仿射translate.05/scale.25/degrees0/shear0/perspective0，fliplr.5/flipud0，mosaic/mixup/copy_paste全0；几何变换同步RGB/框/原mask，mask最近邻，消失或过滤实例逐步记原因，不用polygon桥接丢孔洞。实际Seg loss/增强/完整恢复/评价尚未接入，以上不是可运行config或默认值已生效。

Seg主指标标准Mask AP50:95，辅Mask AP50/AP75/size及BBox AP，外部3clip各列但不称3比赛。保存全图COCO RLE（原图坐标、不lossy polygon逆造）及分数/匹配/原生proto+coeff候选等足够复现的输出；真实GT未知遮挡不分层，合成分层只作诊断。PT0及每阶段0/末验证，R2val/2cp、RR/SR各4val/4cp共10val/10完整cp（0/每500/末，阶段末/下一阶段0两份独立状态），逐步记录及失败保留；原型密集输出与完整状态预算需实测，不凭BBox6650344B替Seg估算PASS。若后来增加val或中途cp须重新核完整资源，不删旧失败。当前只候选，待用户允许后CPU适配，再GPU释放时每模型至多600秒机械准入；不与A/B并行GPU。

Pose新检索重点：3DSP/AutoSoccerPose官方仓库说明200个射门×20张裁剪图有人工2D关节，另10个射门的test只有tracklet；标注格式是H3WB风格命名JSON、坐标相对crop，3D为MotionAGFormer提升估计，不能当实测3DGT。论文说明来自22场比赛、图约100x100，并以PDJ/AUC评2D。出处：https://github.com/calvinyeungck/3D-Shot-Posture-Dataset 及 https://arxiv.org/html/2405.12070v1 。本轮仅核官方README/论文，不下载整个zip，不冒称已逐文件核4000份GT或官方test可评关节；格式/实际图数/每关节名与可见性字段/媒体许可仍须下一步有限元数据实核。论文17点不等于COCO17，头部/躯干定义与缺失点不能凭数量对齐。

Pose建议先做足球单人crop的2D迁移探针，YOLO11s-pose公开17输出保留，不改生成COCO17管线。先核两源共同肩/肘/腕/髋/膝/踝12身体点的严格语义，仅共享点作为辅助监督/PDJ，未对应头部点mask掉监督与评价、不能填鼻眼耳/插值骨架或把v0改v2。若真实文件确有完整同定义COCO17/可见性/面积才另启标准OKS AP；否则共享点PDJ与坐标误差标为自定义共有点指标，不冒称COCO17 AP或照搬论文17点PDJ。Pose未标其它人的crop不能作完整多人检测AP；按事前中心射手选择规则输出一个人，漏检按失败关节计入，不只对成功裁剪例报告。合成按可见mask原框同一裁剪/letterbox规则造person crop，保留原帧/seed/机位/运动血缘，cropped图仍是引用派生，跨机位同运动不跨split。

划分候选：用3DSP有2D标签的原train池，先实读info.ini game/half/time映射；按比赛分组固定训练/开发/未来留出约70/15/15比例，整段20帧射门不得拆，确切比赛/射门/帧数和分组算法及UID须在看模型成绩前登记；同一比赛所有关联片段同split，作者原test无2DGT不能强行作评价。若game身份缺失则先BLOCKED，不能退化为随机帧。当前只读来源结构，不读取新留出GT或模型预测。4000是作者声明，不先假定可准确分3200/800或训练val独立。

Pose开发预算候选同样R500/RR500+500/SR500+500、有效16、seed17、独立公开起点，共2500更新；输入先候选256方形crop（32倍数、共享12点尺度适配），FP32/micro4累积16/workers1；真实train/val数量等比赛划分后写绝对数，S1600个crop须跨16train seed/机位/时间/人物事前均衡冻结，保留同帧相关性。每500阶段SGD/50步warmup/.001峰值→.00001/.937/.0005/clip10与Seg候选同，loss候选box7.5/pose12/kobj1/cls.5/dfl1.5，头部缺失mask与kobj语义须实核；相同HSV/轻仿射/水平翻转需同步交换左右关节，禁止物理翻转标签未换。数值/增强/关键点loss/恢复/记录仍需真实CPU与有界GPU准入，无已运行Pose backend。PDJ建议距离阈值.5，归一化为GT肩中心至髋中心距离；GT退化/缺失归一化样本先定排除并报告数量，预测漏检不排除；分关节/按比赛宏平均及总体微平均并报，AUC网格先冻结。共享点集合与归一化不同于原论文17点，不直接排名其表中分数。

其它足球候选：WorldPose有足球多人SMPL/相机与视频，但当前官方须机构邮箱申请，视频另需FIFA协议（https://eth-ait.github.io/WorldPoseDataset/ 与 https://worldpose.ait.ethz.ch/）；不是匿名即下COCO17、投影关节/可见性也不是人工2DGT。SportsPose有体育动态3D姿态但学术许可按邮件申请（https://christianingwersen.github.io/SportsPose/download.html），不是现成广播足球COCO17。SoccerNet-GSR官方提供场上位置/角色/队伍/背号而非人体17点（https://www.soccer-net.org/tasks/game-state-reconstruction）；SkillCorner BodyPose是带估计误差的模型产物，不用作人工GT。未代用户发邮件/接受许可/申请账号，也未下载大包或启用这些源。

推荐下一顺序：先把Seg的两个小足球实例源与mask适配闭合；Pose并行做3DSP真实关节/比赛元数据核验，在上述缺口闭合后再定最小作业。真实足球优先，保持数据任务明确；Seg/Pose都未启动训练、不新增自动训练TODO。即使短预算方向有利，也只支持该足球面板的开发信号，标注节省/跨比赛泛化依SYNTH-QUANT-001另审，不把单人射门姿态结果冒称整场多人Pose结论。


## QUANT-PLAN-APPLY-001：量化框架获准入方案，Seg/Pose仅CPU准备

用户本轮直接要求量化评估写入方案、优先Seg独立性/类别/质量核验及最小CPU适配、Pose核3DSP关节和比赛分组，暂不启动这两项训练。EXPERIMENTS仅新增该获准节，原全文其余bytes保留；新SHA77284d717d0bd1a6895be9f4ba2eda216709f611c857d0d1a80df14c95155c1c，PAPER仍d48cb4160a513773cde0e910fe1c45c7eb5a63bf058871eafd38e9ad27b71c17。正式框架写绝对bbox/Mask/兼容时OKS AP、相对强R/匹配RR的百分点差、唯一真实标注量效率曲线、配对seed/比赛簇统计单位、观察史/独立test与总成本；未自行设q、非劣效界限或扩训练矩阵。当前BBox A原3f2998冻结为合法历史，未改现运行包/配置/源码/进程；原生bbox-a-b提示已同步最新计划，未来B只使用A/code已冻40模块并显式绑定新方案，不将新增Seg/Pose CPU模块塞入数值全集或放宽原门禁。原每60分钟/Nov7截止/条件B规则和每日g保持。

## SEG-CPU-AUDIT-003：两足球源实标注全量光栅核验与CPU接线

准备范围是公开源GT/少量代表RGB、CPU几何/评价适配，不加载模型、不前向/反向/训练、不GPU准入。两源Kaggle原v1卡/文件清单重新实际读取；完整包URL404不当数据不可用，使用公开fileName+datasetVersionNumber=1取得原文件。一次元数据请求404、两次连接EOF均未改源或训练；在原8分钟/96MiB上限内只补缺文件、原字节相同文件复用，成功末段49.516秒、保守网络累计61110495B。G:/FutsalMOT_Dataset/real/football_seg_public/ihelon_v1保留原18,691,334B COCO（SHA9db8c37770afbaa3a9cba0d41437e6ca1cb6a77fcf0eca3e5ca115ebb97e995b）及16张按全512文件序均匀选取的JPG；共17源文件29,089,006B，余496图未下载，不冒称全图池可训练。quantigo_v1保留原全部12文件38,403,014B，三原MP4不另存逐帧副本；SOURCE_CARD/SOURCE_FILES/SOURCE_RECEIPT和小质量/接线收据留在各源目录。下载成功不是训练数据准入。

类别实际：ihelon仅id1 person；Quantigo三文档均id1 Player、2 Ball、3 Goal Line、4 Field、5 Background、6 Referee、7 Football Pitch Line。CPU派生view将Player+Referee（含源标为Player的门将）映为person/class0，另五类明确非目标排除并计数，未知类拒绝，不增加观众ignore/修GT。原类别/id/annotation完整副本、source SHA、image/frame身份保留；多文档可能同文件名，UID现显式source:原文档SHA:filename，避免跨Batch碰撞；同SHA换路径保持同UID。此为准备映射，不把未披露角色/看台范围称完整所有人物真值。

独立性：实际目视16张ihelon均匀样本及Quantigo三clip首中末9帧。前者可见NEW YORK RED BULLS/MINNESOTA UNITED FC标识、红/浅蓝队；后者JUV/RM比分和统一视频水印/紫色皇马与斑马尤文，三段保守同一比赛。16张ihelon解码RGB与全部847个Quantigo帧逐一比较无完全重复；64-bit DCT pHash最小Hamming14，pHash本身不构成独立性证明。Quantigo视频227/195/425帧均1920×1080、对应COCO文件名后缀0..N-1精确连续，内部也未发现解码RGB完全重复。只有抽样比赛不同的强证据，ihelon全512原视频/比赛ID/时间未披露，不能声称全库比赛独立或盲测PASS；三片不能当三场。源卡CC0与CC BY-SA4.0如实登记，不将Quantigo卡称YouTube MIT等同已核完整广播许可链。

标注质量逐图有界解码原polygon为官方COCO RLE/mask，全部512+847图/14,011人物annotation（7686+6325）均成功，未拒图，target非有限点/原图越界点/crowd均0。ihelon全7686条原area等于官方光栅像素面积；Quantigo多数面积为连续polygon几何面积，与光栅略异，原area保留用于原标准评价，不私自覆盖。光栅框与原连续bbox的边缘差另存，ihelon最大19.19px、Quantigo各Batch1/3/13px，并不将这些差统一解释为标注错误。ihelon有1条3像素Mask（ann5904，bbox1.95×5.2），Quantigo B3有1条2像素Mask（ann5895，sourcearea2.5）；训练缩放后消失须逐步记录，val不按训练阈值删GT。

**确定问题，未修补：** ihelon image_id120/209.jpg原21annotation中有7对polygon、bbox、area、类别及光栅完全相同：(1767,1773),(1766,1775),(1776,1777),(1768,1779),(1778,1782),(1771,1783),(1769,1784)。因此原7686是原annotation数，不是已去重人数，不能直接拿它宣称唯一实例。Quantigo Batch3 image_id16641148/Football batch 3.mp4_00224.jpg还有一对完全相同的ann3585/3586；原6325同样是未修复annotation数。另B3 ann4482（frame268）polygon存在严格自交边5/12，原area303.5但COCO raster613（相对差101.98%）；这是具体自交，不是只凭面积差猜GT错。另ann5895的2.5→2是极小三角光栅差。全部原byte、ID和失败/异常收据保留，没有去重、删帧、重画、clip多边形或增加ignore；训练前须明确审定可追溯派生去重/修订策略或换源，不能把当前源报整体质量PASS。

6图原Mask叠加实际目视：ihelon0/34/527共17/30/15实例，Quantigo B1frame226/B2frame0/B3frame0共16/4/16实例；主要可见轮廓能对应球员/裁判，包括门将，未标看台人物仍为范围局限。小样目视不是14,011实例人工边界质量验收。诊断联系图/叠加展示后精确清，原G来源GT/样本保留。未观察模型输出或test。

CPU实现为新src/futsalmot/segmentation_data.py SHA7467601148e57494f735102b0215ef4fdf23ca092eea71735d58c91ed331ac36与tests/test_segmentation_data.py SHA1431477476c46f7c8ebaec8cb47ec7431c0bf7758655b99140d88b2c1b045dc1。root独立重跑14项CPU PASS/0跳过2.519秒，含原COCO polygon/压缩与未压缩RLE、孔洞/断开/重叠保持、整数instance ID、类别/UID/跨split泄漏、同步960letterbox+轻仿射/HSV/左右翻转、ratio2最近邻/过滤原因、micro1空目标/多目标、标准pycocotools2.0.10 Mask AP完美/错误/空预测与完整precision/recall/scores数组。仅CPU fixture的完美AP1不是真实模型成绩。pycocotools有NumPy copy关键字弃用warning，检查未失败，未为warning改依赖或源码。

静态锁定v8SegmentationLoss batch_idx/cls/bboxes/masks字段接线，并发现上游overlap=False外层zip(...,masks)在多图少目标时有截断范围限制，本准备明确仅允许原候选micro1（未来累积16由Trainer负责）；不热改上游，也不宣称多图loss正确。真实CPU样本接线：ihelon0.jpg17/17保留，已发布S seed_00007546/C5/000001的uint8整数mask与原JSONL十人visible_pixel_count逐一相符，10/10保留，所有实例RLE往返exact；均构造img[1,3,960,960]和masks[17或10,480,480]，torch.cuda.is_initialized=False。没有模型loss执行、优化更新、完整Trainer/恢复或GPU性能验收。

小准入状态为CPU_ADAPTER_READY_SOURCE_QUALITY_REVIEW_REQUIRED，详见G football_seg_public/CPU_PREPARATION.json及原来源/ANNOTATION_AUDIT/AREA_OUTLIERS/CROSS_SOURCE_AUDIT/CPU_ADAPTER_SMOKE小收据。当前Seg CPU数据—增强—组批—标准评价基础适配完成，源独立性完整链、上述重复/自交处理、所有512 RGB与训练数值/资源/Trainer尚未闭合；Seg不训练、不自动预检或启动R/RR/SR。

## POSE-3DSP-AUDIT-003：真实关节/裁剪尺寸/比赛划分核验

官方calvinyeungck/3D-Shot-Posture-Dataset固定commit c34f46d2f5d42e996ee328a95ad49217206b5607（2024-06-18）。原3dsp.zip实际23,496,450B/SHA3696523b69dd3033435f76ab64410aa8e34172df32bb9673eca93b6609c87ee7，Git blobdeab0655c7414065afd9a71d915df703958e2f5f。CPU子任务实际一次仅读ZIP目录、全部JSON/info和JPG头，不提取媒体/不解码像素；9453成员、原始未压39,750,646B。4200JPG/4000JSON/210info/210gt.txt。原train200射门/4000有2D标签帧来自20场，原test10射门/200图、无JSON来自3场，合计22场。原train/test共有2016-01-03 - 16-30 Crystal Palace 0 - 3 Chelsea（train3shot/test2）；其余test各4shot来自2016-01-13 Chelsea2–2West Brom与2016-02-07 Chelsea1–1Manchester United，所以官方split不比赛完全独立，也不能当有2D关节GT的测试集。

实读4000份标签全为如下17个ID/name：0 Center of Hips；1 Left Hip；2 Left Knee；3 Left Ankle；4 Right Hip；5 Right Knee；6 Right Ankle；7 Center of Body；8 Center of Shoulder；9 Neck；10 Head；11 Right Shoulder；12 Right Elbow；13 Right Wrist；14 Left Shoulder；15 Left Elbow；16 Left Wrist。共68000个2D点name/x/y均finite，无逐点v/visibility；info的visible仅shot级，不能全补v2。严格共有身体点COCO→3DSP：5→14,6→11,7→15,8→12,9→16,10→13,11→1,12→4,13→2,14→5,15→3,16→6；鼻眼耳无同定义点，Head/Neck/中心点不能冒充。原合成/模型COCO17不改、不新建12点生成管线，共有点辅助指标明确区别标准COCO17 OKS。3D字段是作者lifting估计，不当人工或实测3DGT。

实际适配差异：4000JSON image_path均为未补零1.jpg式，ZIP真实001.jpg式，严格同shot/frame三位路径兼容后4000全对应，不任意搜索代替缺图。801份JSON bbox w/h与实际JPEG头尺寸不符；全图有27种尺寸，不全100²，未来用真实图头归一化。按bbox算78点出界，但按真实图半开尺寸仅1点：train/00189/posture/010.json joint13 Right Wrist=(100.0,32.78)、真实100×117，恰右边界；未clip或改GT。PIL头与独立SOF解析一致。210info game/half/time/id/shooter完整，previous_id及(game,half,annotated_position)不重复；但end-start=600ms、step40ms与20帧应760ms覆盖不符，保留原值、不用其证明精确序列同步。未解码像素检查左右人工标签/边界，所以不是全面人工质量PASS。

比赛级候选（未正式冻结训练）：只用原train20场，按sha256(UTF8('3dsp_match_split_v1|seed17|'+game))升序；14/3/3场分别141/33/26射门、2820/660/520帧。dev为2015-09-12 Crystal Palace0–1Manchester City（14shot）、2015-10-31 Chelsea1–3Liverpool（10）、2015-02-21 Chelsea1–1Burnley（9）；holdout为2015-02-21 Swansea2–1Manchester United（5）、2015-08-29 Chelsea1–2Crystal Palace（16）、2015-11-29 Tottenham0–0Chelsea（5）。其余14场141shot为train：ManchesterCity–Sunderland2015-12-26(6)、ManchesterCity–Watford2015-08-29(11)、Everton–Chelsea2015-09-12(12)、CrystalPalace–Chelsea2016-01-03(3)、Chelsea–Bournemouth2015-12-05(16)、Liverpool–AstonVilla2015-09-26(14)、ManchesterUnited–WestBrom2015-11-07(9)、Chelsea–Swansea2015-08-08(18)、Chelsea–AstonVilla2015-10-17(8)、ManchesterCity–Liverpool2015-11-21(12)、Southampton–Liverpool2015-02-22(3)、Chelsea–Sunderland2015-12-19(9)、CrystalPalace–Arsenal2015-02-21(3)、ManchesterCity–WestHam2015-09-19(17)。完整原game字符串和排序规则由只读模块复现。整shot20帧不拆、比赛不跨组；这是新项目候选而非官方test。所有4000标签已做结构核验，不再把候选holdout描述为从未访问GT的全新盲测；未观察其模型成绩，当前不训练。

原210info共65414B，按sorted member名+NUL+原bytes串联SHA559f9ee41605455496267c0b22915df77e4c9f6ab5e5d0c716fd1f9ec66a754a；4000JSON共19778685B同算法SHA9d357e9eee1904019a8434c1233a735cad3181ad173989b4b5532f0510e13100。仓库Apache2.0 LICENSE.md/blob6aba58abbf9cdc45decc7c9cca48c2b3cceeb3dc不自动覆盖上游SoccerNet视频权利链。来源：https://github.com/calvinyeungck/3D-Shot-Posture-Dataset/tree/c34f46d2f5d42e996ee328a95ad49217206b5607 与官方论文https://arxiv.org/html/2405.12070v1 。23.5MB临时原ZIP在展示审计后精确清理，无媒体解压/新报告树。

新独立src/futsalmot/pose_3dsp.py SHA2dc231272b664974349017b94f746b3558dd2836124648d517ce9a77b48f909f，tests/test_pose_3dsp.py SHA8d79e2ec7035da0ee3f77b5438cf723c9730b66964580737a0ffe91bd474c198；root独立9项CPU PASS/0skip0.001秒；实际全metadata/JPEG头子任务审计1.016秒。工具提供严格源关节/零填充/图头/分组核验，只返回有名语义共有点，不生成伪COCO17、visibility、训练或下载入口。当前Pose维持CPU核验准备，不启动模型/GPU预检或训练。

收口核验：root最终23项CPU PASS/0跳过2.404秒；四份原Seg COCO SHA、小证据绑定及A已冻40模块与根对应文件全部匹配，EXPERIMENTS/PAPER仍77284d71.../d48cb416...。原文档及源码未回写现运行包。两份临时helper共8145B、两张已展示诊断图共1674497B已按四个绝对文件名精确清理并确认不存在；G原始Seg GT/16JPG/3视频与小收据均保留，Pose临时ZIP由CPU子任务展示后已清。Seg/Pose训练/GPU初始化0。


## BBOX-INTERPRET-002：已完成结果的效用边界与固定总量混合问题

依用户要求读取四项既有COMPLETE末步小收据，不加载模型/重评价。正式R6563/105008R，AP51.6511537498%、AP50 94.1099523965%、AP75 50.5687308826%；长RR36750/588000R，AP46.65311479396%；长SR36750/105008R+482992S，AP48.5407098794%；短A9563/105008R+48000S（新增只有R6563），AP49.7264816863%。长SR比同更新长RR高1.8875950854pp，说明该有限长预算/两阶段策略中用S代替额外R的结果更好；但较强R低3.11044387044pp，短A仍低1.92467206348pp，未证明加入S提高最佳实际BBox检测能力。短A比长SR高1.18577180696pp是缩短前缀的开发信号，缺未执行B的短真实前缀对照，不能分离合成内容与缩短额外训练的作用。所有未超基线结果保留，不因负结果恢复B/追切点。

短A相对R的AP50=-1.36276524667pp、AP75=-3.65598599863pp；size small=+0.226940412377pp、medium=-1.93819903146pp、large=-1.37010241245pp。更严格定位指标下降较多、小目标仅微小正差可作诊断；单seed/单场已开发val/无CI及多项查看，不能挑small正差宣传独立收益，也不能据此唯一归因为定位/GT错误。S阶段在真实val低于公开起点，以及合成/真实人物密度/尺度差异，使域偏移/真实预训练特征遗忘成为合理假设，不是因果结论；长额外R同样下降说明预算/调度也可能重要。最终SR高于公开起点本身不能归因S，因为后接R训练。没有低标注嵌套子集、跨比赛test或配对seed，不能声称节省真实标注/普适有效或断言该数据集对所有BBox场景无用。Seg公开GT问题与SoccerNet BBox原GT为不同来源，不拿Seg标注瑕疵解释本轮BBox。

用户随后明确公平比较应固定SR/R总训练量。root此前保持R曝光而建议13126步，是额外S增量问题，已撤回该比较预算，不扩成新RR。新的问题是同105008次总曝光用一半S替换重复R；各仍使用同2100张R图池，不把重复R曝光减少称标注节省。

## SEG-DERIVED-PREP-004：可追溯候选、完整自交扫描与保留原GT

用户要求继续优先Seg，对重复/自交给处理方案并保留原始GT、暂不训练。新独立segmentation_views.py与tests/test_segmentation_views.py实现只读SHA绑定、按原所有非id字段完全相同才合并同图同类目标、固定保留最小annotation id、保全部alias与原payload SHA；语义属性不同/跨图/跨文档不合并，原polygon/area/bbox不改，所有数据派生不覆盖source。

此次完整float64严格proper segment crossing扫描全部14011原人物polygon，比前次仅针对面积异常定位更全面；ihelon1688条/479图、Quantigo B1 4条/4图、B2 56条/28图、B3 16条/15图触发。1024顶点上限，实际各库max420/272/528/605；NumPy包围框剪枝，实际四库4秒级，不重解码全RGB/Mask，不把未检出proper crossing称完整简单多边形或人工边界PASS。精确非id全字段去重数仍7+0+0+1=8。

严格整图隔离候选已生成G football_seg_public/derived/seg_gt_candidate_v1（CANDIDATE_MANIFEST SHA0f97a51c588e4e09710b36f6a15642e1623c18b241c1af9dd1d72aaec8bd0946），去重后隔离任何目标自交图及其全部类别标注，绝不只删人物令其变未标背景。结果ihelon仅33图469条，Quantigo各223/167/410图、person418/832/4645条，原全类数量守恒。这暴露“见自交就全图删”过于激进，不能自动把该候选当最小训练面板，更不能为了保512配额悄悄放宽门限。

推荐准备方向为另一个**原COCO光栅语义规范化候选**，不是自动人工修形：按锁定pycocotools2.0.10将原polygon解释为原尺寸RLE；原area/bbox和其余字段完全保留，仅旁路审计记录raster area/bbox、新旧segmentation/payload SHA。该步骤在已确定COCO解释下保持像素，不将“存储为有效RLE”冒称轮廓人工正确，不做buffer(0)/凸包/轮廓桥接/补点或按模型输出修GT。自交仍全部留人审findings，ann4482面积303.5→raster613只作审计，不自动覆盖源area。需要人工裁定的实例若最终排除应整图统一排除/另行明确合法ignore，所有模型同一冻结view；不为某模型单独改GT。当前未批准最终Seg实验GT，不读模型分数选择清洗策略。

新RLE候选位于G:/FutsalMOT_Dataset/real/football_seg_public/derived/seg_gt_rle_candidate_v1，CANDIDATE_MANIFEST SHA0253bdff744d9d3ccc013bc5e6afc5d4e035d0361c5a8731695b6cbc23e9ea69，4份派生document+ledger+manifest及总清单按SHA读回。所有1359图保留；ihelon原7686→7679个person、Quantigo原6325→6324个person，仅8精确重复有alias合并，非目标类别原样。原COCO四文件SHA未变；逐实例ledger保sourceSHA/image/annID/原payload、新RLE SHA、area/bbox保留值、仅审计raster信息。原COCO到RLE通过同一官方编码定义；另4库各3实例实际decode原/新Mask逐像素相同，共12抽样，不冒称再次全像素人工验收。二候选均CANDIDATE_NOT_ADMITTED、human_review_required/训练false，无训练入口。原source GT/样本/视频未删除，两个候选供比较审阅而非两个新训练数据集版本上线。

新模块SHA9068c30e4e99c62b135e1a91fc40f6aa133f743d32a62e838cc1a5bafb6423ed，测试SHA19a4c9bc8da67d1e5c9426cc375a9dbcfdea045749ce589d8ef40a7759ab4eee。root连同原Seg14项/Pose9项，38项CPU PASS/0skip2.382秒，覆盖别名、绑定篡改/语义差异、全图隔离、计数守恒、RLE孔洞/自交像素等价、原area/bbox/source不变。两候选文件链均实际核SHA。原有数据→增强→micro1→Mask AP准备继续复用，未来Trainer/模型loss/完整恢复与资源GPU准入仍未执行；完整来源身份/496张ihelon原RGB/疑难标注人审未闭合，Seg/Pose不训练。

## MIX-FIXED-APPROVAL-001：用户直接批准单项固定总预算SR混合

用户先要求设计与既有正式R比较，随后明确“SR混合训练正式加入方案，开始SR混合训练”。root已仅按范围向EXPERIMENTS新增获准节，原文其它bytes保持，新SHA dc4596e63565eeeb50a5ae65a2327f7fbc6211997ee532c1a2cec95a645afe82；PAPER仍d48cb4160a513773cde0e910fe1c45c7eb5a63bf058871eafd38e9ad27b71c17。只批准一个独立公开YOLO11s/seed17单阶段final6563混合作业，对照已完成R_seed17，不新增R/RR、预S或后R，不恢复旧A/B跟进。总batch16，两个micro8各4R4S交错，总R52504+S52504=105008；相同960FP32/workers2/SGD100步warmup到.001/.937、cosine到.00001/原EMA与增强/损失。两val0/末750真实、15完整cp/逐步全证据、末EMA不best/test。单项3h/完整资源实测Gate；新混合GPU机械预检最多120秒计入原422.829后≤542.829≤600，不重做旧预检/AMP模式搜索，真正基础设施有界修订也不重置预算。现仅CPU实现/审阅准备，尚未GPU预检或正式混合更新，不把用户许可当Gate PASS。

混合批数值loss仍原sum(native microbatch scaled vectors)/16，不增域loss权重，1:1是图数不意味着梯度相等。旧formal R+S是36750整步交错/两阶段，不能直接改几个预算字段当本新方案；新入口独立，41源码仅A已冻40模块+fixed_mix，不把Seg/Pose新增CPU代码混入。独立审阅发现旧snapshot验证每个流draws必须16整除，奇数混合步各8无法用旧validator；必须独立mixed完整快照/恢复契约各8*i、总16*i，真实奇数step1保存/恢复后step2 exact，不能只偶数保存/伪造游标/热改旧checkpoint_state。相应准备进行中，Gate实际通过后才root唯一派发。


## FIXED-MIX：R_plus_S_fixed_budget_seed17 STARTED

UTC 2026-11-03T07:20:49.947508+00:00。单阶段独立公开YOLO11s/seed17，0/6563更新；实际R/S各0，总曝光0/105008。每有效16含8R8S、两个micro8各交错4R4S；SoccerNet train2100/val750/S24149，960FP32/workers2，单SGD/EMA，原100步warmup与final .001 cosine至.00001，原损失累积、不另加域权。15计划完整cp、2次实际真实验证、逐步与原生预测及失败全部保留，不best/test。固定总曝光比较R，真实曝光减半；单seed有限开发证据。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed17_20261103_a01/R_plus_S_fixed_budget_seed17。


## MIX-FIXED-ADMIT-001：同105008总曝光的唯一8R8S正式训练已派发

用户直接授权正式加入方案并开始。新fixed_mix.py SHA4ae0f93a0c3fb460b2e816b50798de4ae63405645cbfee28da67b05ef07cac7d，独立静态审阅通过；root独立8项CPU PASS/0skip2.273秒。A已冻40数值模块逐项SHA保持，实际41仅加新mixed入口；协议strategy=R+S/schema=bbox_fixed_budget_mix_v1/job_id=R_plus_S_fixed_budget_seed17，不调用旧36750步formal R+S定义。只本一个新job，旧R/RR/SR/A原封保留。Seg/Pose不训练。

root实际有界GPU机械预检PASS52.547秒/3诊断更新，step1奇数R8S8完整写盘/按可信SHA读回→fresh公开模型恢复→step2 R16S16，raw/EMA/SGD/全部CPU CUDA RNG/sampler/loader_generator/iterator状态及loss全部指标、实际梯度张量逐值exact；两个micro各交错4R4S、双游标共同提交。实际代表首R/S cached/direct全部sample字段一致。2诊断临时pt各114289967B/合228579934B已显示路径/SHA并由TemporaryDirectory精确清理，owned33004/wrapper14208真实退出/锁退休后才准备正式；无额外真实val评分。原422.829+52.547=475.376秒累计，不重做原330/AMP或更多toy。准入G runs/bbox_fixed_mix_prepare_20261103_a01；helperSHA1b5b47b87154a6c36e2eac0b5481a7b05210e65b9201234e483c60cf5ec26364，D admissionSHA0e4d0d60438396b5442fe814c3551020cae8dda75a68b24a1adf52eca854f907。派发前进程扫描首次把自身-c命令文本误识别为launch字符串，已改核具体script argv，原未创建组/未启GPU，未消耗机械GPUattempt或重放任何训练。

实际GPU peak_reserved9093251072B、peak_allocated8482889216B，free6615465984B，余量>1GiB。step1冷loader3.718秒、同iterator step2 .546秒、fresh恢复冷loader3.297秒，不能把短窗当全程吞吐保证。时间核以2*.546*6563+1200验证+150cp+600头余量+2*max冷步=9124.232秒，冻结max10800秒=3h。checkpoint114289967B按1.1上取，15cp+6563步各2MiB+2*750(native6650344B+JSON1MiB)+一次750失败partial+3GiB瞬态+2cp，完整新增33.94170332327485GiB；准入D余113.273235GiB/G772.585762GiB，另D32MiB+100GiB/G100GiB不降。storage_revisionSHA9e8586dbd862e9c9ad8965721920a4c0e07545754e705a58e6c2bdd28a5d6218。原R的public/requirements/真实和合成完整来源（只排除reader_cache运行统计）实际一致，R status精确SHA引用已核。

唯一正式D runs/main_bbox_yolo11s_soccernet_RS_seed17_20261103_a01，G:/FutsalMOT_Training/runs/同名，2026-11-03T07:20:35.9425625Z（北京时间15:20:35）隐藏派发。wrapper31292/CT1793690435.9425623，实际controller28200/CT1793690435.9826293，完整D本组launch.py命令和自有GPU锁已核，dispatch.json D/G读回。launchSHA41310f950b0e805001f9059651c4910daa2e7f9d7708a2a2fc2e28239f357420；job_tableSHAedf94c68ae8990991a838402cb62d92c97a57850c28ac1ac18b177c6000466fb；code_bindingsSHAfb6cbdbacca07dfbef001028986733536837684f13389e8d31bb536664eea4b7。runtime自07:20:45.908557Z起，绝对截止10:20:45.908557Z（北京时间18:20:45），不因恢复重置/不延Nov7全局界限。单阶段6563步/52504R+52504S=105008，公开YOLO11s独立起点，原960FP32/SGD final .001/100warmup/micro8×2/workers2及原增强损失；15完整cp、PT0/末步两个真实750val、逐步/密集原生预测全部留存。当前实际RUNNING且PT0验证进行中，初始update0并非已提交训练步数，不冒称已完成任何R/S曝光。大证据全直接G。

观察台本机HTTP已实际显示“正式｜BBox｜YOLO11s｜SoccerNet｜R+S｜seed17”，稳定ID96ddd66688e31b2ebf08，新readonly PID33268；因相对服务时长启动存在0.27秒偏移，独立重载时保守扣30秒，实际截止北京时间20:17:50.964725早于原20:18:20.679743，不延预算。训练未受重载影响，不持GPU锁。有限heartbeat bbox-sr-mix已创建ACTIVE/每60分钟/原Nov7 UNTIL，仅必要生命周期和终验；健康静默，不扩任何其他job，终验后删除自身，g保留。用户人工观察训练，process_lifetime_isolation_verified=false，不承诺Codex重启隔离已验。


## MIX-FIXED-START-VERIFY-001：实际混合更新确认

起点真实750验证已完整生成summary后进入训练；有界131072B逐步尾部读取确认实际已提交166/6563，R=1328/S=1328，两个micro均严格交错4R4S，逐步记录实际domain=R+S，不把PT0当训练。准确controller28200/CT1793690435.9826293仍运行；D冻结41份源码逐SHA复核、EXPERIMENTS/PAPER封存保持。小收据G本组startup_verification.json。此后交用户观察，只做必要生命周期，不按此继续巡查AP/steps；单项预算和终点不变。


## BBOX-REDESIGN-APPROVAL-001 / CHECKPOINT-1000-001：用户直接批准设计与保存规则

北京时间2026-11-03，用户明确放弃S3000→R并允许删除记录，同意成对短两阶段、真实标注减少和开发/独立确认分离，询问并批准保存间隔500→1000。根将该开发作业研究用途置ABANDONED_DIAGNOSTIC并从未来主结果/候选选择排除；选择保留不可变原始证据/负结果和观察史，未物理删媒体/模型/日志、未回写原COMPLETE/SEALED或篡改原记录。此前提到A结果的文字只为历史，不继续作为方法选择证据。

获准短RR1500R+5063R、短SR1500S+5063R均6563步/105008总曝光，各独立公开起点、每阶段100warmup/.001完整cosine至.00001、raw承接/新SGD EMA、其他原数值不改；两项必须成对，无A赢才派B规则。25%目标真实N525配对及一个预先固定合成方法，群组/UID/实际N须事前审；同N增益不等于已节约标注，q/非劣效尚未设，不声称75%节省。既有750继续明确为开发范围，确认需方法冻结后独立比赛另审，不看test。以上均已批准设计/未执行，不将同意设计冒充GPU启动许可或Gate PASS。

未来未冻结新训练每阶段0/1000倍数/末完整cp；6563单阶段8份、1500+5063两阶段10份，受控边界额外保存，验证/逐步字段不变。当前root formal/fixed_mix与活动D/G私有代码全部保持500/现RS15cp；本次仅方案规则生效，未冒称执行器已接1000、未进行GPU数值验证。保存为只读快照/序列化原则上不改变连续训练loss/参数，I/O下降与硬中断可恢复间隔变大须同时说明，逐步日志不能反演模型或授权未知提交重放。未来源码接入需有意义CPU参数/loss/RNG/loader等价验证及对应实际准入。

EXPERIMENTS新SHA 9939468aff4bb0925e73c04a761da6f89f687b6fab022b3986287b39f2610910；PAPER原SHA保持。本次只向获准范围新增最新节，之前方案全部bytes保留，AGENTS/README/TODO登记；旧dc4596活动RS绑定合法，不改其冻结job_table/launch/代码。当前混合继续原有有限生命周期跟进，Seg/Pose仍CPU准备不训练。


## PLAN-STRUCTURE-001：实验方案按研究结构重写，结果与变更迁入日志

北京时间2026-11-03，按直接要求纠正方案被日期/对话条款堆叠的问题。本次EXPERIMENTS按11个主题重整：研究问题/数据监督/共用数值/BBox流程/低标注/Seg/Pose/评价/记录/资源/文档边界。正文不再保存对话经过、测量分数、PID、审批先后或反复覆盖的旧条款；重复和过期的停止/精度/验证/保存规则合并为单一现行规则。具体实验保留目的、对照、独立起点、绝对更新/曝光、LR、切阶段、评价和停止判据。旧全文原bytes仅迁入本日志下方历史区作变更证据，不另建方案备份或平行日志，不再作为执行指令。

旧EXPERIMENTS SHA 9939468aff4bb0925e73c04a761da6f89f687b6fab022b3986287b39f2610910，新SHA 68e6d9738322870a6cc23d877811c646c9ab55b9eb0a6ff8d0a7414db2e024be；PAPER原SHA保持。当前mixed旧冻结计划/源码/控制与所有历史run不改。原先S3000诊断退出主结果和选择的决定不变，执行历史保留。原方案中失效的“Seg真实mask未取得”“Pose只能整图/标准17点”“所有任务自动20S+50R/五策略矩阵”“优先重新扫AMP”“每500检查点/每1000验证”等叙述不再充当当前默认；相应源审计、CPU实现差距和历史协议仍按日志追溯。Seg/Pose细化只整理既有候选与数据契约，不新增训练授权。

### 已核验BBox结果索引（来源为既有COMPLETE收据，本次不重评价）

| 实验 | 成功更新 | 实际R/S曝光 | 末步COCO AP50:95 | 记录/范围 |
|---|---:|---|---:|---|
| 独立R50校准 | 6563 | 105008R/0S | 51.6511537498% | BUDGET-001；校准身份独立保留 |
| 独立R100校准 | 13125 | 210000R/0S | 49.5718879083% | BUDGET-001；预算选择不当合成贡献 |
| 正式R | 6563 | 105008R/0S | 51.6511537498% | R2R-ANALYSIS-001；主参照 |
| 长RR | 36750 | 588000R/0S | 46.65311479396% | R2R-ANALYSIS-001 |
| 长SR | 36750 | 105008R/482992S | 48.5407098794% | S2R-RESULT-001 |
| S3000→R开发诊断 | 9563轨迹/6563新增 | 105008R/48000S | 49.7264816863% | SHORT-PREFIX-RESULT-001；ABANDONED_DIAGNOSTIC，不入主结果/不作选择 |
| 固定总量R+S | 目标6563 | 目标52504R/52504S | 末步尚未在本次核验 | MIX-FIXED-ADMIT-001；不把初始验证当终结果 |

长SR−R=-3.11044387044pp，长SR−长RR=+1.88759508540pp。仅同seed/同750已开发验证比赛的有限结果，不据弱长RR对照声称超强R，不把重复曝光减少写为标注节省；训练变长、高LR、EMA/优化器重建及域偏移未被独立因果分离。完整AP50/AP75、大小分项、成本和收口SHA沿各结果条目，不从差值补造缺失指标。

### 当前直接后续执行安排

用户随后明确当前实验约25分钟后结束，要求定时任务改30分钟，并在当前混合完整终验后继续新S2R和R2R。已把同一bbox-sr-mix改为每30分钟、原Nov7 UNTIL不变；执行链为mixed→短SR1500S+5063R→短RR1500R+5063R，两项均独立公开起点且RR无条件执行。新短配对CPU实现准备中，GPU机械与实际资源准入未完成，不能把本设计或旧S3000准入当PASS；当前活动500间隔/15cp原样保持。新两阶段每阶段独立warm100/.001→.00001、每1000完整cp/每阶段0末验证，不扩低标注/SegPose或其它seed。

### 方案整理前原文迁入（历史证据，以下不作现行执行指令）

<details>
<summary>整理前EXPERIMENTS全文；SHA 9939468aff4bb0925e73c04a761da6f89f687b6fab022b3986287b39f2610910；原bytes保留</summary>

# 准备与实验方案

## 2026-11-03直接批准：放弃旧S3000诊断、短两阶段配对、低标注与独立评价、未来1000步检查点

用户直接决定放弃旧S3000→R记录，并同意短两阶段、节约标注和减少同一验证集反复选策略的设计；同时批准未来完整检查点由每500改为每1000成功优化更新。本节优先于下方相关旧候选和保存频率。此次为方案批准与准备范围，不新增GPU训练/预检/下载/数据重划/自动队列；当前已经启动的唯一固定总量R+S继续原冻结协议、15份计划检查点和既有有限跟进，不能热改。未来作业仍须完整实现、恢复/数值与资源Gate及明确执行准入，不能把本节当实际PASS。Seg优先和Seg/Pose不训练的边界保持。

### 旧开发诊断退出主结果

D/G main_bbox_yolo11s_soccernet_S2R_seed17_20261103_a01、short_S_to_R_seed17（开发诊断S3000→R）研究用途状态改为ABANDONED_DIAGNOSTIC / excluded_from_primary_results=true / selection_eligible=false；原执行状态COMPLETE与真实测量不可改造成失败、未执行或不存在。其继承的S3000为原30187步长调度前缀，3000处LR约.00982778，未完成短周期衰减；后接R6563已有.001→.00001完整微调，不能写成没有微调。它不代表合理设计的短S→R，退出后续主结果表、候选筛选与论文贡献证据，不复用其权重或据此选切点。用户允许删除，但本次选择保留原始日志/GT/模型/收据及旧叙事作可追溯历史，不物理删除、不回写已SEALED控制；从主结果排除不抹除负结果观察史。旧条件B仍未执行且不再恢复。

### 短两阶段成对设计（已批准设计，未执行）

| 方法 | 第一阶段 | 第二阶段 | 总成功更新 | 最终R/S图像曝光 |
|---|---|---|---:|---|
| 已完成强R参考 | 无 | 6563R | 6563 | 105008R/0S |
| 新短R→R | 1500R | 5063R | 6563 | 105008R/0S |
| 新短S→R | 1500S | 5063R | 6563 | 81008R/24000S |

两项成对设计均从同一已绑定公开YOLO11s权重独立起点开始，seed17、原R2100/S24149/真实开发val750、960FP32/micro8有效16/workers2、原网络/可训练层/损失/增强/角色/评价保持；不继承旧3000前缀或旧已训练权重。1500×16=24000约为合成池一遍，仅为事前预算依据，不宣称最优。两项必须同预算、同调度、同阶段转换完整执行并保留结果，不再采用“S先超过强R才运行匹配RR”的条件选择。当前混合训练结束前不启动它们。

两个阶段分别完整warmup100步，LR从0到.001、momentum从.8到.937；随后各按原离散cosine公式，以本阶段U=1500/5063衰减至.00001，不截取30187步旧调度。SGD Nesterov/WD weight组.0005、bias和norm0/clip10及原EMA规则保持。阶段间raw承接、新SGD和EMA重建；第一阶段R使用独立R_extra流（SR使用S流），两组终阶段使用同一事前固定的R流与原采样规则、各0起81008次draws；增强RNG、loader状态和阶段转换须显式实现/绑定，不能声称跨域梯度或所有RNG轨迹自动相同。第一阶段末EMA与第二阶段0重建EMA独立tag，不合并。

每项共4次真实开发验证：两阶段0和末步；主结果仅最后EMA，不best/看分数早停。按新1000间隔，first检查点local0/1000/1500共3，final local0/1000/2000/3000/4000/5000/5063共7，共10份计划完整cp，受控退出额外保留。全部逐步loss/LR/梯度/R/S UID/来源/曝光与原始预测仍保留。比较短SR−短RR分离前阶段数据来源；短RR−强R反映阶段拆分/调度效应；短SR−强R检验固定总预算实际价值。短SR真实曝光较少，此设计不同时保证真实曝光相等，不称标注节省。学习率/检查点变化写为新协议；正式执行前须实测完整恢复、阶段转换和RNG隔离，实算完整存储/吞吐/有限时间，不复用旧准入结果冒称已核。

### 低真实标注配对（已批准设计，未执行）

先形成唯一真实标注池25%候选N=525（全量2100），同一525给纯R和一个事前选定的合成方法，独立公开YOLO11s/配对seed17、同总6563更新×16=105008曝光。当前混合和必要短两阶段开发收口后，基于开发证据固定一种合成方法；不在每个标注比例重搜方法、LR、比例或切点。若选1:1混合，则纯R105008R、混合52504R+52504S；若选短SR，则按本节81008R+24000S，均只从同一N真实图中draw，另完整报告曝光与重复次数。方法选择、预算/调度及全部UID/hash在训练前封存；不能将候选方法写成已决定或已训练。

从原train候选按比赛/clip/时间段/人物尺度事前分层，保留连续帧和同运动群组关系，严格不借val/test挑易例；N=525只有在既定群组约束可满足时取精确值，否则报告实际N并先明确冻结，不拆群组凑数。未选真实图不用于该臂训练、伪标签或超参选择；旧全量真实数据及使用史不删除。横轴记录唯一图数、人物实例数、比赛/clip数和可测标注成本，公开预训练曾用真实图的事实保持，所以结论仅针对新增目标足球标注量。

固定计算下小R池重复更多，须保留低标注R的完整学习曲线与同等开发调参机会，不用明显未收敛/过拟合的弱R作对照；若要调整共同调度须在独立开发规则下成对冻结，不事后择best或让某臂额外调参。先完成一个N的配对；有信息价值后再讨论嵌套50%等曲线/配对seed，当前不自动扩矩阵。低标注SR优于同N的R只说明该标注条件下有效；要支持节约标注，还需达到全量R的可比水平及实际曲线。q、非劣效差距与统计判据尚未设定，须事前明确，不自动补容差、外推节约率或宣称节约75%。

### 开发与确认评价分离

既有750张SoccerNet val已反复用于预算/方法开发，继续明示为开发集及其观察史，不重新命名为盲测。先冻结有限方法与选择规则，主表保留全部正式正负结果；旧S3000诊断单列已放弃，不参与选择。未来确认阶段先审比赛身份/时间相邻性/真实独立来源，再固定未用于训练/选择的比赛评价集，方法和超参冻结后、另获执行许可才一次评价；本次不读取封存test、不自动下载/重划或以新clip伪造新比赛。seed配对反映优化随机性，不能替代新比赛；按比赛为统计单位，独立比赛不足不构造泛化CI。开发集反复选策略后的分数不作为独立泛化证明。

### 未来检查点每1000成功更新（CHECKPOINT-1000-001）

适用所有未来尚未冻结的新正式/开发训练（BBox/Seg/Pose），不热改当前R+S或历史run。完整cp规则为每阶段0、每1000、阶段末，端点与间隔重合只保存一次；阶段间即使global相同也因SGD/EMA不同独立保存。受控用户暂停/取消、可安全保存的失败或时间边界另存完整状态。raw/EMA/SGD/CPU CUDA RNG/采样/loader/计数和逐步全记录不减字段，验证仍阶段0/末、不随检查点频率变化，不自动恢复未知提交。

长度U阶段计划份数=1+floor(U/1000)+int(U%1000!=0)：单阶段6563从15份降至8份（0、1000..6000、6563）；新1500+5063两阶段共10份。按同一cp约114.3MB，单阶段少7份约0.745GiB（不含受控额外份）；原生预测与逐步日志可能仍是主要存储，不能称总空间减半，未来资源Gate须按实测重新算。硬崩溃时最近持久完整状态间隔由500扩大为1000步，最多不足1000步之间的精确恢复能力受限；逐步日志不是可逆参数，未知提交不猜重放，受控暂停仍即时保存。

检查点是成功更新后的只读快照/序列化，不参与loss、optimizer、EMA或采样推进，因此仅改变保存间隔原则上不改变连续训练数值，只改变I/O与可恢复粒度；不是已实际CUDA逐值验证结论。未来执行器须将protocol/真实loop/恢复/计数/资源/journal全部绑定1000，拒绝仍硬编码500的旧入口；当前源码formal/fixed_mix及当前冻结包仍500，不能把方案批准称源码已接入或当前频率已变。首次未来接入时用有意义CPU检查确认不同保存频率下训练参数/loss、RNG/loader游标不变，再按对应任务有界准入，不重复占用当前GPU或无变化测试。


## 2026-11-03直接批准：固定总训练量的SR混合与已有正式R比较

用户已直接要求SR混合正式加入方案并开始执行。本节仅批准一个新的YOLO11s/BBox/SoccerNet/seed17固定预算混合作业，与已完成正式R_seed17比较；不追加R重跑、RR控制、前置S或后置R阶段，不扩比例/seed/规格搜索。Seg/Pose仍只CPU准备，不启动其训练/GPU预检。旧R/RR/SR/短A及其负结果、原始GT、完整日志/预测/检查点/旧封存均保留，不热改旧包；当前量化框架仍有效。以下优先于旧五策略R+S的长预算/整步域交错描述，只适用于本次新独立协议。

- **公平预算与研究问题。** 单阶段6563次成功优化更新，有效batch16，总105008次图像曝光，与现有正式R完全相同。R原每步16R、总105008R；新SR每步8R+8S，终点52504R+52504S。不会为了保持105008R而改为13126步，也不额外训练一个13126步R对照。回答“相同总训练量下，以一半合成曝光替换重复真实曝光是否改善真实验证成绩”；真实池仍相同2100图，R重复曝光减半不能称真实标注节省。
- **独立起点与单一优化过程。** 从已绑定公开yolo11s.pt独立初始化，训练seed17；不接已训R/SR/A权重或任何合成前缀。模型和所有原可训练层/冻结DFL等、原COCO80检测头/person目标规则保持。单阶段final共6563，只有一个SGD与EMA过程，不做阶段reset。输入960、FP32、micro8×2累积为16、workers2。每个micro8固定4R+4S，顺序R,S交错，两micro合计8R8S；不使用先8R后8S的域纯micro，也不把旧整步R/S交错实现称本批内混合。
- **数据与采样。** 仍SoccerNet原train2100/val750、原GT/角色/ignore/有序图池，S原全部16train seed/24149图；不混SportsMOT/Seg来源或重划test。R/S独立原seed17 permutation/cursor与每样本增强随机流，R按原R流、S按原S流取样，每更新各预留8，真实后端step成功才两个cursor一起提交；失败立即保留边界/原始证据，不丢/重放未知更新。每个micro的R/S UID、目标/ignore、增强和来源完整记入逐步日志。S仅使用已经READY/退出验收的无损960 cache（D cache/synthetic_bbox_960_v2，manifest fce4e02611f198ba1e72fe1a5ae47ebf047f83a29b2738ef88a0d1429001b324）；像素/标签/几何与direct不变，不新建缓存或把其当新数据版本。
- **数值严格匹配R。** 原SGD Nesterov、momentum100步从.8线性到.937，LR同100步从0线性到.001，随后按原schedule_at(final_stage=True, updates=6563) cosine到.00001；bias不单设高LR，weight decay仅weight组.0005，bias/norm组0，clip10。box/cls/dfl损失系数7.5/.5/1.5、原微批反传和按有效16归一化保持，不新增人为R/S损失系数；1:1指图像数量，不冒称两域梯度恰各50%。原EMA decay与raw/EMA/SGD/CPU CUDA RNG/loader/所有sampler状态完整保存。HSV .015/.4/.4，translate.05/scale.25/degrees0/shear0/perspective0/fliplr.5/flipud0/mosaic0/mixup0/copy_paste0及原box过滤/role规则保持，不增加域适配或改网络。
- **评价与比较。** 同750张真实SoccerNet val，主结果只末步EMA标准COCO AP50:95/max100及原player/goalkeeper评价范围，辅AP50/AP75/size/recall；先列绝对值，再给混合−正式R百分点差。固定R参考AP.5165115374980012/AP50.941099523964852/AP75.5056873088257927，绑定其已COMPLETE status精确path/SHA和原protocol/public/data/源摘要。新作业仅阶段0公开起点PT与6563末步两次完整750图验证，不复用S前缀PT、无需额外阈值校准，不best/早停/test。旧R8次val原记录保持，新2次验证不减少训练更新；只可用末步作开发比较，不能把验证次数不同造成wall差异称混合训练加速或字节完全同协议独立重复。评价隔离训练RNG/loader generator/模式，成功失败均恢复。
- **证据与停止。** 完整checkpoint在0/500..6500/6563共15份，受控边界额外保存，累计曝光R=S=8×committed_updates，总=16×updates；所有逐步loss/LR/梯度/UID/微批/原生密集预测/JSON与负结果保留，全部大证据直接G。任何混合composition、sampler commit、完整恢复或绑定不符具体FAIL/BLOCKED，不改bool绕Gate；人工取消不自动恢复，未知提交不猜重放。末步是否超过R都完整保留并停止，不以分数触发更多训练。
- **机械准入与资源。** 先完成独立新混合入口与CPU测试/审阅，原40数值模块从已完成A冻结code精确复用，只加本混合模块；根Seg/Pose CPU准备源码不接入。新数值/采样恢复机械GPU检查至多120秒，计入既有422.829秒后累计≤542.829≤600；不重做旧预检或AMP/micro搜索，若真正基础设施一次修订亦在同一累计120秒内。实测每micro4R4S、完整raw/EMA/SGD/RNG/sampler/loader恢复及nextstep loss/梯度exact，cached/direct代表像素/标签/增强一致才PASS。诊断临时模型展示后精确清，小准入/失败保留。
- **有限预算。** 单项最多3小时，原全局2026-11-07T23:55Z不延。完整空间按实测checkpoint×1.1、15份cp、6563步每步2MiB、2×750图(native6650344B+JSON1MiB)、一次750失败partial、瞬态3GiB+2cp重算；另G100GiB和D控制32MiB+100GiB，已有cache和旧失败不删。预算/吞吐/资源须实际Gate并冻结，不用历史值冒充PASS。无GPU/UE并行，真实owned/PID/CT+cmd/锁先核，不重复派发。
- **唯一执行与闭合。** 独立D小控制/G同名canonical RS目录，task detect/strategy R+S但使用明确fixed-budget-mix schema/job_id；不得直接调用旧formal.protocol('R+S')的36750两阶段规则。先查候选，prepare冻结依赖/public/data/cache/正式R参考/本新方案和41执行源码，再隐藏唯一launch。用户人工观察训练，只设有限每60分钟必要生命周期跟进，健康静默；结束核6563/52504R52504S、两val/15cp、小control SEALED逐小SHA、owned退出/GPU锁退休后报告并删除自身，g不动。单seed/已用于开发的单场val不作普适贡献或统计显著性结论。


## 2026-11-03直接批准：合成数据效用量化评估与Seg/Pose CPU准备

用户直接要求将SYNTH-QUANT-001的量化评估写入方案，并优先推进两个足球Seg源的独立性/类别映射/标注质量核验及最小实验CPU适配；Pose继续核验3DSP关节定义和比赛划分。**Seg/Pose本次均不启动训练、GPU预检或模型推理。** 当前BBox A和条件B仍按原独立冻结范围执行；本行政/评估框架更新不热改其code、job_table、source bindings、训练循环、数据或证据。旧3f2998/3814等摘要为合法历史，已运行A不回写；未来未冻结的条件B显式绑定本次新方案和既有获准BBox数值代码，不将新增Seg/Pose准备模块当BBox数值许可。

### 核心评价与绝对结果

合成数据效用以真实数据上的任务表现衡量，没有跨任务统一的“合成质量合格分数”，不以合成训练loss、图片逼真度、FID/KID或通过标注门禁替代迁移成绩。BBox主指标为标准COCO bbox AP50:.05:.95，Seg为标准COCO segm Mask AP50:.05:.95；AP50/AP75和规定面积分层辅助。Pose仅在关节顺序、坐标、标注/可见性、area、OKS sigma及检测范围符合COCO17契约时报告标准OKS AP。共同身体关节上的PDJ/PCK或裁剪射手任务另标名称/归一化/阈值/匹配/缺失规则，不冒称完整COCO17或整场多人AP。原真实GT/有序图池/主后处理与原始预测保持，先报告每臂绝对值，再给配对差。

- Δ_R=100×(AP_SR−AP_R)个百分点：同真实标注池下，相比强直接R是否有实际提升。
- Δ_matched=100×(AP_SR−AP_RR)个百分点：相同额外更新预算下，合成前缀是否优于继续真实训练；更新相同不自动等于wall/能源相同，另报实测成本。
- 当前R51.6511537498%、长RR46.65311479396%、长SR48.5407098794%，分别得到−3.11044387044pp/+1.88759508540pp。原负结果不改；不能只挑较弱RR对照宣称优于真实训练。A/B为预登记短前缀开发，A>R仅触发B派发，不是统计显著性、等效或论文贡献通过线。

### 真实标注效率

标注效率的横轴是唯一真实标注图像数/人物实例数/已标关节数，须同时列比赛/clip来源，不能用重复曝光或优化更新次数代替。未来如获准做效率曲线，应固定真实val/主指标/模型起点与选择规则，事前冻结嵌套真实训练子集和匹配训练预算。在预先选定目标AP q且实际测得两曲线达到时，报告L(q)=1−N_SR(q)/N_R(q)及原始N/成本；目标q、非劣效容差与预算须另外明确，当前不自行设定通过界限，不外推未测区间、不低分后移动q。当前A/R均使用2100真实图，所以尚不能宣称节省真实标注。新增子集/模型/seed作业需另获执行许可，本评估节不自动扩大矩阵。

### 稳定性、泛化与统计单位

未来获准重复时，各训练seed配对报告全部绝对指标、均值/标准差和逐seed差，不能选择有利seed；seed变化仅反映训练随机性，不能替代新增独立比赛。跨比赛泛化须方法冻结后在未见比赛上评价，并保留观察史和全部负结果。匹配区间建议在同一比赛簇上配对重抽并每次重算AP，报告95%区间、簇数/单位与重抽规则；若只一场，只能提供片段条件描述或区间，不能据相邻帧制造比赛级泛化CI。3个同场clip不能冒称3个独立比赛。无充分独立单位时明确缺区间/证据不足，不杜撰CI、显著性或非劣效PASS。最终test只在全方法/参数冻结且另有执行许可后一次评价，不用test改策略。

### 监督/覆盖诊断与总成本

结构和像素对齐、原GT血缘、人物密度/尺度/长宽比/机位/光照/动作覆盖、过滤/缺失/截断率分别报告；未知真实遮挡属性不补造。它们用于定位域差异，不单独证明迁移收益。成本同时列新增及继承的训练/验证GPU wall、数据生成/缓存wall、标注量、存储与失败成本；复用前缀不称成本为零，生成与缓存可另列可复用成本/摊销假设，不将一次缓存加速冒称数值或数据效用提升。源/UID/GT/模型/完整状态/逐步记录/全部评价与失败按本方案保存，主结果末步EMA、不best/test追分。

公开依据：官方COCO API（https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py）；Task2Sim CVPR2022（https://arxiv.org/abs/2112.00054）用真实下游任务检验合成预训练且强调任务依赖；Ruis等2024（https://arxiv.org/abs/2405.19822）讨论强预训练基线及合成域过拟合；Tang/Jia CVPR2023（https://github.com/huitangtang/On_the_Utility_of_Synthetic_Data）研究监督/域适配效用。文献不等于本项目因果证明，不搬用其涨幅作为本项目门限。

### 本次Seg/Pose准备边界

Seg仅两个已讨论足球来源ihelon/football-player-segmentation v1和quantigoai/soccer-dataset v1。核验原标注/目录/来源与比赛线索，保留“实测不重复”和“比赛身份/标注完整性未知”的区别；没有官方split时不随机逐帧拆，不以站点/存储目录不同当独立比赛。CPU适配只准备显式person类别映射、原COCO polygon/RLE/整数实例mask读取、身份绑定、同步几何/独立mask、标准Mask AP输入与有意义的CPU检查；未知类别/来源/监督缺失拒绝，不自动修GT、增加ignore或把孔洞/断开区域桥接抹掉。映射和质量结论按实际核验写唯一日志，不能靠配置flag冒称Trainer或GPU数值已通过。小预算Seg三臂及绝对超参数仍按候选待审/待资源准入，不从CPU准备自动派发。

Pose仅核3DSP官方源码/数据元信息/有限原关节标注与全部可得比赛info：关节名/左右/坐标/可见性、原train/test有无GT、比赛/射门/时间组及可复核划分。人工2D与模型lifting的3D分别标记；17点数量不等于COCO17，缺鼻眼耳等不插值或补伪GT，原生成COCO17不改。原test缺人体GT不强行评价；按比赛而非相邻帧设计候选分组，身份缺失明确BLOCKED。只新增独立CPU模块/检查，不改运行BBox冻结数值模块，绝不启动Seg/Pose训练或GPU预检，不发外部申请/消息。


## 2026-11-03直接批准：短前缀BBox开发分支A与条件真实对照B

用户明确允许将BBOX-NEXT-CANDIDATE-001思路加入方案并开始执行。本节优先于较早“仅三核心/后续只讨论”的执行范围；旧三核心已完整闭合，其code/config/run/检查点/预测/失败/封存摘要全部保持，PAPER不改。本次最多A/B两个开发分支，先A，仅A末步EMA主COCO AP50:95严格高于既有R的0.5165115374980012时才启用B。此比较只是开发派发条件，不是统计显著性/等效界限或论文贡献通过线。A未超过R则停止该候选，不自动追加切点/LR/模型/seed搜索。

- **A：S前缀3000→R6563。** 来源为已完成S_to_R_seed17的first第3000步完整可信checkpoint，继承原独立公开YOLO11s/seed17训练前缀与48000S曝光（全16train seed/24149图池，约1.99名义epoch），不重新执行前3000步、不改旧run为3000步完成。源receipt SHA173a0b01c24489995943c7c3ac127dfd1eff80346fbafdeb22c4257e8d0958e3/114292975B。新做原真实终阶段6563更新/105008R，完整轨迹9563更新、48000S+105008R，实际新增仅6563更新。
- **B：R_extra前缀3000→R6563，条件执行。** 来源为已完成R_to_R_seed17的first第3000步完整checkpoint，48000R前缀曝光，receipt SHA2e0bd6aee38cd14f8fef8aef03dd9cb6010939cb2b60ea0bdb9738c82d0eeed1/114295151B；其余新真实微调与A相同，完整轨迹9563更新/153008R，实际新增6563更新。B只在A完成/owned退出/控制封闭及A>R后准备，不预先派发。只有A同时优于R与B才值得另讨论独立确认，多seed/模型规格尚未启用。
- **固定数值与阶段承接。** YOLO11s/seed17、SoccerNet原train2100/val750、有序UID与GT不变，960FP32 micro8有效16/workers2；SGD Nesterov momentum.937/WD.0005/clip10，真实阶段LRpeak.001/cosine至.00001，原HSV/仿射/翻转/损失/评价和角色/ignore策略不变。先按原source_bindings/protocol恢复完整first backend（原stage_updates30187/phase_update3000），包括raw/EMA/SGD/RNG/三sampler/loader generator；first末EMA评价保护RNG后，承接raw、重新建立终阶段SGD/EMA与local0，R sampler原seed17且已提交0，携原generator进入新阶段并按iteration_already_started=False建立新iterator。既有前缀的学习率轨迹原样继承，不将其称重新压缩的S2 cosine训练。不得将EMA当raw或任意重播种。
- **账与结果定位。** 新branch steps按新增local update1..6563连续记录，另含trajectory_update=3000+local、继承和新增R/S曝光；完整checkpoint遵守原累计曝光=16*committed_updates契约，其committed_updates为trajectory3000+local、phase_updates为local。两个游标及parent receipt写入schema/绑定，不混成3000步重训或丢继承曝光。开发分支不是新的独立训练重复；父前缀成本和来源完整报告，不替代今后另行批准的公开起点独立确认。
- **评价与保存。** 仍有两阶段0/末四个逻辑评价收据。与parent公开起点的图像顺序/模型/GT/后处理/精度绑定完全一致时，PT0仅引用精确既有成功summary及SHA，不复制预测或伪造新推理；first3000的阶段EMA、final0从raw重建EMA、final6563末EMA各做一次完整750真实val，三次新增。任一PT0绑定不齐则不复用，并增加一次完整val及重算资源，不省略。新真实阶段0/每500/末15份完整raw/EMA/SGD/CPU CUDA RNG/sampler/loader checkpoint，所有逐步loss/LR/梯度/UID/曝光、原生密集预测/JSON及负结果保留，受控边界额外保存；主结果仅final末EMA、不best/早停/test。
- **有限时间/资源。** 每分支max3小时，两项GPU串行，原绝对全局截止2026-11-07T23:55Z不延长；完成一项就封小控制。按已测cp和原记录公式，单分支候选新增39.31938921GiB（15cp按114292975B×1.1、6563步2MiB、3×750图native6650344B+JSON1MiB、一次750图partial、3GiB+2cp），另G100GiB/D小控制32MiB+100GiB，旧缓存/失败仍保留。若增加PT0评价或实际实测更大，则按完整新值Gate。预检只验证新恢复/中途切换/新R完整状态下一步一致和worker接线，最多120秒、计入此前实际GPU准备389.469秒，总≤509.469≤600，不重做原330秒或AMP搜索。单次真正基础设施错误最多一次同协议有界修订且总准备预算不重置；人工取消/未知提交/数值失败不自动重训。
- **准备检查与证据边界。** CPU先用已有标注/索引有限核对角色/ignore、尺度/长宽比/增强过滤及遮挡/截断框定义；合成来自可见mask tight box，SoccerNet人工MOT框的可见/amodal定义未独立认证，不能凭来源称其amodal或断言GT错误。不重哈全媒体/重渲染/按val筛seed。若有真实契约冲突则具体BLOCKED。A/B分别独立新D控制/G同名输出，parent/checkpoint/PT0/公开基座/数据/依赖/源码/本新方案全绑定冻结，不热改旧数值模块。新prefix入口和CPU回归通过后root才有限GPU机械预检，PASS和完整资源Gate后启动A。
- **范围闭合。** 新TODO只列A和条件B；A≤R时B记未执行而非负结果，A/B全部结果保留。当前val已用于多轮开发，不称独立盲测。即使A获益，也使用同样2100张真实训练图，不能把减少重复真实曝光称节省真实标注。是否开展seed29配对/低LR/混合/新模型/SegPose训练须后续另定；本批准不扩大到这些任务。用户后续要求的合成贡献量化标准与Seg/Pose细化先调研规划，不自动训练。健康中间态静默，只新增失败/阶段完成/需处理通知，完成当前有限分支及小控制/owned退出后删除自身跟进，不操作每日g。


更新：2026-10-31。文档持续原地维护，不增设协议方案目录或版本化Prompt。real整理与PDF文献库已交付；15份YOLO11公开基座已下载核验通过，状态与来源见weights/yolo11/manifest.json。单一Python3.12与PyTorch/torchvision/Ultralytics环境、CPU准备工具已交付，全部32项CPU检查通过；仍未启动真实网络训练/推理。


## 2026-11-02直接批准：未来新实验仅每阶段0及阶段末验证

用户明确同意：未来新实验只在每个阶段0和阶段末验证，保留逐步日志和每500步完整检查点；当前正在进行的训练不改。本条覆盖较早每1000步val及“仅讨论”的规则，仅对本次批准后尚未冻结的新实验生效；已有训练/缓存冻结包、原计划摘要、原验证记录和失败不回写、不删减。

- 每个阶段保留阶段0与最后一次成功更新后的完整750图val。独立R/S各2次；两阶段S→R/R→R/R+S各4次。第一阶段最终EMA和第二阶段从raw重建EMA后的阶段0属于不同模型状态，即使global update相同也分别命名/评价，不合并。
- 新协议明确evaluation_policy=phase_endpoints_v1、evaluation_every=null及精确evaluation_schedule，实际训练循环和显式恢复都按同一计划执行，不再在500/1000等非终点补val。已闭合的同tag评价遵循原严格收据复用，不挑best；末阶段末步EMA仍是主结果，不读test或以val调参/早停。
- 逐步loss/LR/梯度/UID/曝光/采样及全部负结果继续保留；完整raw/EMA/SGD/RNG/采样/loader状态仍每阶段0、每500成功更新、阶段末和受控失败边界保存。阶段切换另存新阶段0，不覆盖前阶段最终；验证降频不减少任何训练更新、R/S图像曝光或已有证据。
- 每次实际执行的val保留原生密集预测、后处理/标准与辅助指标、stdout/stderr及失败partial。新formal评价外围以finally保护Python/NumPy/Torch CPU/CUDA RNG、独立训练loader generator和原模型训练模式，初始化或评价失败也恢复；不声称CPU夹具已验证实际CUDA吞吐，未来cached GPU准入仍在原最多240秒/累计570秒预算内。
- 原8次R验证与40次长任务验证改为新2/4次，长任务验证/密集预测部分减少90%，不是总训练或总存储减少90%。按原测量和完整记录公式，只将长任务val40改4，候选完整新增上界约110.18GiB（旧303.78GiB），另G100GiB保留及D32MiB小控制+100GiB；正式准入须用实际新checkpoint/吞吐/既有证据重算并冻结，不能靠旧数字或cfg布尔值直接开训。
- 本次降频不扩大当前s/seed17三核心、多seed/其他规格/Pose/Seg范围，不延长原硬截止。新run绑定本次重新封存的计划与实际新源码；当前R→R仍按其旧每1000验证协议运行，CPU缓存与Real退让代码保持原冻结。

## 2026-11-02直接批准：当前单seed探索与后续分支

用户直接要求移除TODO中的扩展训练条目，当前继续YOLO11s BBox单seed探索；不同模型规格方案保留，待s探索完成查看效果及资源后再决定；是否多seed依后续训练进度决定。Pose与Seg仍是项目论文核心组成部分，只因详细方案尚未成熟暂不列入TODO，不能解释为放弃这两个研究任务。此条覆盖下方较早矩阵/自动顺序中的执行优先级，不删除已完成实验或失败证据。

- 当前正式训练范围仅SoccerNet面板、YOLO11s、训练seed17的R、R→R、cached S→R三个核心对照；另2次已完成R50/R100校准为历史开发成本。R已完成、R→R已启动由用户观察，S→R仍按缓存READY/旧组闭合/有界cached预检及完整资源Gate条件推进。S单独、R+S、多seed和其他规格不在当前执行队列。
- n/m/l/x及原R与S→R的比较设计继续保留在本方案，状态为等待s探索完成后查看效果再决定。是否启用、采用哪些规格及有限预算须直接讨论并在启动前固定；未启用项不是负结果，已启用项全部报告，不能试多个规格只保留上涨项或低分重训。当前未验证跨容量效果，不作该贡献主张。
- 多seed不是当前强制完成量。17/29/43保留为早期登记候选，29/43不列TODO、不自动派发；是否追加重复实验依据后续进度、资源和所需稳定性证据再决定。单seed先作为探索，不能把单次结果或相邻帧当重复训练/独立比赛，也不按有利seed筛结果。
- Pose与Seg为论文核心待详细规划任务，现有参数表仅早期候选，尚未构成可执行方案。后续须分别明确真实GT取得/补标、合法整图池与来源内split、逐实例监督有效掩码、loss/评价实现、最小对照、实际预算和资源证据，再讨论具体作业；当前不列41项训练待办、不提前承诺沿用BBox全部矩阵，缺真实GT不以合成内部成绩代替真实迁移。
- 验证降频已由同日后续直接授权，未来未冻结实验按上方最新每阶段0/末规则执行；逐步记录和0/每500/阶段末完整checkpoint不变，活动run仍保持其旧验证协议。

本次仅批准当前范围/后续决策状态的原地更新并重新封存，PAPER继续原SHA。旧已启动run仍保留原EXPERIMENTS摘要/路径/代码，未来新run机械绑定本次新摘要；不能恢复覆盖旧包或据根新SHA误判旧证据被改。

## 2026-11-02直接批准：实验统一命名

用户在直接对话审阅命名设计后要求“命名设计写入到设计文档中去，然后应用”。本次只增加命名与显示规则；模型、数据划分、训练预算、增强、评价、存储和停止Gate均不变，PAPER继续封存。

监控标题、实验选择器、对比图例使用以下统一显示名：

`阶段｜任务｜模型｜数据源｜策略｜训练seed`

| 实验 | 显示名 |
| --- | --- |
| R50预算校准 | 校准R50｜BBox｜YOLO11s｜SoccerNet｜R｜seed17 |
| R100预算校准 | 校准R100｜BBox｜YOLO11s｜SoccerNet｜R｜seed17 |
| 正式真实基线 | 正式｜BBox｜YOLO11s｜SoccerNet｜R｜seed17 |
| 正式额外真实训练 | 正式｜BBox｜YOLO11s｜SoccerNet｜R→R｜seed17 |
| 正式合成预训练与真实微调 | 正式｜BBox｜YOLO11s｜SoccerNet｜S→R｜seed17 |

“数据源”指已冻结的真实训练/评价来源面板；合成版本、完整UID/split和公开权重仍以run.json为准，不能据显示名混合来源。未知字段显示“未登记”，不得根据看似相似的目录名猜数据来源、模型或实验阶段。seed17是训练随机种子，与合成数据seed身份无关。

从本次批准起，尚未创建的新独立实验外层D控制/G输出组统一采用：

`{purpose}_{task}_{model}_{panel}_{strategy}_seed{seed}_{YYYYMMDD}_a{NN}`

例如：`main_bbox_yolo11s_soccernet_S2R_seed17_20261102_a01`。两端group_id必须完全相同，仍在现有runs职责与RUN_LOCATION协议内创建，不增额外模型/策略空目录。字段约定：

- purpose：main为正式实验，cal50/cal100为对应名义真实预算校准。
- task：bbox、pose、seg、number、mot；实现中的detect/segment仅在名称中映射，不改训练任务token。
- model/panel：公开模型原名称及明确来源面板的小写安全标识，例如yolo11s、yolo11s-seg、soccernet、sportsmot；拒绝空白、路径分隔符、..等越界字符。
- strategy：R、S、S2R、R2R、RS，分别对应内部R、S、S->R、R->R、R+S；显示名用箭头/加号，不修改训练采样或策略token。
- seed：训练随机种子；YYYYMMDD取首次开训派发的北京时间日期；a01起为同一实验的执行尝试编号，冲突时核实原组状态，不覆盖、不把低分作为增加尝试的理由。
- 同一run从可信完整checkpoint恢复时保留原ID和日期，在既有恢复证据内登记，不把恢复拆成新的独立实验。预检/缓存修订也不算新的模型训练实验。

已有或已经冻结派发的组、内部job_id/run_id、目录和所有原始证据保持原值，包括当前R/R→R共用的核心调度组；统一显示名按run清单只读生成，稳定选择ID/历史引用不变。未来新单作业组采用上述外层目录名，内部leaf仍可沿用已冻结执行器的R_seed17/S_to_R_seed17等ID，不能仅为外观改动数值执行代码。原始路径和稳定ID在监控详情保留，便于定位。

本次新计划SHA仅供批准后的新实验绑定；旧已启动run继续引用其原冻结计划SHA，这是合法历史绑定，不能覆盖其run.json/code/检查点或伪造为新SHA。新任务准入需绑定本文件本次新SHA；未来机械接线可单独引用新计划摘要，不得以命名更新放宽原数值来源一致性检查。

## 2026-11-01直接讨论批准：检查点降频与G盘留存

用户明确允许“降低保存频率，同时每次实验完可以将当前实验保存的数据移动到G盘，在G盘建立专门保存训练数据的区域”。新独立实验完整科研检查点调整为每阶段0/每500次成功更新/阶段最终；受控停止或失败另存实际边界，不追溯删除旧run已产生的检查点。逐步loss/LR/梯度/UID、阶段0/每1000/最终val及全部原始预测、原指标和负结果继续保留；训练更新、数据、增强、数值、模型选择规则不变。

当前已经运行的bbox_s_seed17_calibration_20261101_a02/R50_seed17保持其原冻结每50规则到闭合，不热改活动代码或run绑定。若原控制器已进入下一run，也保持该run的实际冻结协议，不重复派发。新R100在原控制器退出并确认尚未执行后，以独立公开起点及新500规则冻结，引用原R50有效末步结果选择预算。

用户随后直接要求避免搬运、检查点直接写G。未来独立run的唯一大证据区域为G:/FutsalMOT_Training/runs/<group_id>，检查点、逐步日志、原始预测、评价与stdout/stderr均从开始直接写G，与G:/FutsalMOT_Dataset媒体分开。D仓库保留源码、环境、15份公开基座、唯一实验MD及runs/<group_id>小型控制目录；双端RUN_LOCATION.json精确绑定D控制/G输出，无junction、不覆盖已有组。开始前将不可变代码/依赖/协议小快照留G，结束且worker/writer与GPU锁闭合后封闭control/CONTROL_MANIFEST.json；只同步小型控制证据，不再跨卷搬运完整run。已有R50按原校验迁移流程闭合，原数据/失败status与ARCHIVE_MANIFEST、D RELOCATION保留，不改其历史绑定。G写入失败/路径冲突则保留现场并停止，不能静默改写到D或删除证据。

未来直接G准入按G单run完整新增上界加100GiB余量，以及D小控制快照32MiB上界加100GiB余量分别核算；运行时检查实际写入卷，不再给D重复预留G完整工作集。直接写G省去搬运，不等于G有无限累计容量。仅本次保存与存储条款获准修改，PAPER继续封存，未自动增加实验范围。原R50按50规则133份；未来独立R100按500规则28份，当前这对校准共161份（若两次均按新500则43份），不再套用旧397份。完整s五策略三seed若R50，规定checkpoint约924份，按s实测114192047B/份粗估98.27GiB；原始预测另计，不能只凭checkpoint降频宣称完整15/39块准入。

## 2026-11-01直接讨论批准：正式BBox口径

用户在逐项解释与推荐后明确“按照你的建议，正式开始训练”，授权以下四项对应条款更新；PAPER.md继续原地封存，其余研究范围/完整证据/资源与停止Gate不变。本轮先实际完成YOLO11s有界预检，再逐项准入seed17独立R50/R100预算校准；不是跳过Gate启动15/39次的许可。完整s15/39仍须资源准入；本次后续直接批准的500检查点与G盘留存规则优先。

1. **监督范围与ignore：** 本轮检测目标为球员与门将，保留公开COCO80头/person0。来自源标注、GT flag=1且明确非球员角色的裁判/staff/ball等作为已知背景，不一概ignore；invalid_gt_flag/unknown_role才保留为真实ignore。原GT/索引/源策略不改，训练派生role_policy_v2另记原角色与处理去向。真正ignore仅对框内未匹配anchor中心屏蔽负分类BCE，球员正匹配优先，原box/DFL及正样本归一化保持。没有几何/语义真值不凭空造ignore。首轮一次核train/val实际计数；标准主AP若val有无法可靠计分的真实ignore则明确BLOCKED，不把普通人物伪标iscrowd。
2. **评价：** 主指标用pycocotools2.0.10官方COCO BBox AP50:95，并报AP50/AP75/尺寸分项，默认maxDets=[1,10,100]。GT为本研究球员+门将子集，原图连续xyxy/原面积，不按尺寸/难度删val/test GT；已知非球员误报计FP。模型输出仍按已登记conf0.001、NMS0.7、最多300保留，主COCO API按100截断。现custom_player_ignore_AP@300仅为非官方辅助诊断（使用明确记录的非目标/ignore框、IoA0.5），不参与R50/100预算选择，不与COCO/SoccerNet官方榜绝对分直接比较；SoccerNet完整任务含裁判/球等，当前不是完整官方范围。所有对照同一评价范围/参数。
3. **增强过滤：** 图像/targets/ignore使用同一仿射、裁边、翻转；正目标复用锁定8.3.237默认box_candidates：变换后w>2、h>2、相对缩放参考框面积保留比>0.1、aspect<100。不采用“只要非零就保留一切”；ignore仅去零面积，避免把需忽略区域误转背景。原GT不变，逐图记录被过滤项/原因与矩阵。只对train做一次过滤统计；val/test不作训练式过滤，不按分数调阈值，不重做生产质量/像素验收。
4. **输入：** 原RGB直接等比例LetterBox到960×960，auto=False/scale_fill=False/scaleup=True/center=True，float比例、round缩放尺寸、整数居中pad114、INTER_LINEAR；targets/ignore及推理逆变换共用同一geometry。4K16:9为960×540后上下各210。原生预缩放+LetterBox不必然两次插值，不宣称直接版本天然提精度。训练叠加既定HSV/仿射/左右翻转，val/test仅确定性映射。

本次真实池N_R=2100（SoccerNet train3场/14clip），val=750（1场/5clip）；N_S=24149（原全部16train seed/103机位）。S20 U_S=30187；R50 U_R=6563、R100 U_R=13125，两个独立校准合计19688更新；检查点数以最新存储条款与各run实际冻结频率为准。PT起点、0/每500/最终检查点和每1000/阶段末val、末步EMA选预算等原规则保持。仅真实train/val可用于准备/预算，不读test内容。

## 当前自动推进范围（2026-11-01）

用户最新已授权持续推进：渲染前完成环境与CPU代码准备；全20seed生产根验收/正式发布和所有消费者正常闭合后，自动按DATASET整理synthetic入口，再完成协议/资源/数值Gate并开展原计划实验。本文件较早“准备后停止/不自动训练”指各子任务本身不得越阶段，不能用它重复要求已给出的后续执行许可。当前阶段仍不训练；发布、数据、实现、数值和资源缺一则不能开训。

原生本聊天跟进ID=automation，每20分钟一次，最长7天/504轮，绝对截止北京时间2026-11-08 07:55；健康或无变化安静，阶段完成/新失败/需用户决定时通知。到界不派发新run、按预设完整状态安全收口并删除本跟进，生产a13和每日g保持。主线不可独立解决的用户决策阻塞在完成其他准备后停用，局部缺GT分支登记后继续已就绪主线，不无限原地轮询或扩实验。

环境本次已完成：约16分钟内按固定官方版本安装，CPU导入/算子、pip check与全部32项检查通过，0跳过。原4小时窗口截止11月1日11:49未延长、下载重试0；准备资源峰值约11.33GiB<20GiB。临时.venv/_install_20261101及一次性helper已展示后删除；状态路径不存在是正常收口，后续不得重复安装。CUDA运行/数值/显存和真正模型训练仍待生产释放后的Gate。

自动顺序以2026-11-02最新直接决定为准：已完成数据/环境/代码准备与首SoccerNet YOLO11s独立R50/R100校准后，当前只推进seed17的R、R→R、cached S→R有限探索及必要缓存/准入。s探索完成后，再直接讨论多seed或其他模型规格是否启用；不自动进入原五策略三seed或四尺寸24次块。Pose与Seg是论文核心，另待详细方案与真实GT落实，不在当前TODO或自动训练队列。下游固定检测MOT与已登记评价仍须相应有限准入；基于全部实际结果解释贡献与局限。

## 后续仓库结构设计

本节确定职责与命名，不表示现在创建全部目录。当前已按授权逐项创建实际需要的环境、权重和初步CPU包，不预建未启用的run/结果树；结构以本节职责按需落实。

```text
FutsalMOT_SyntheticDatasetResearch/
  README.md                         # 唯一总入口
  AGENTS.md                         # 简短执行边界
  .gitignore
  docs/
    plan/
      PAPER.md                      # 方向、贡献假设与证据
      DATASET.md                    # 数据入口、结构与整理规则
      EXPERIMENTS.md                 # 本文件：准备、运行参数及实验范围
      TODO.md                       # 唯一执行清单
    paper/
      README.md                     # 一个文献索引：书目信息/来源/版本/本地PDF
      <稳定论文简称>.pdf            # 公开原文；不按篇另建阅读卡或子目录
  pyproject.toml                    # 写代码时才建：单一可安装Python包/命令入口
  requirements.lock                # 建环境时才建：实测可用的精确依赖
  configs/
    experiment.yaml                # 一个现行配置，含数据路径、任务/策略及有限作业表
  src/
    futsalmot/
      __init__.py
      __main__.py                   # inspect/dry-run/train/evaluate统一入口
      data.py                       # G索引、原标签读取、任务适配与R/S采样
      training.py                   # 共享训练调度；不按尺寸复制脚本
      evaluation.py                 # 检测/MOT入口与统一结果口径
      records.py                    # UID/曝光、日志、完整检查点及RNG接口
      adapters.py                   # 只读BBox样本/角色/ignore契约，不冒充loss接入
      layout.py                     # 正式发布绑定与精确迁移dry-run，无apply能力
  tests/                            # 实现任务需要时才建：少量有意义的CPU夹具检查
  .venv/                            # 唯一独立环境，不入Git
  weights/
    yolo11/
      manifest.json                 # 一个公开起点来源/SHA清单
      <官方模型名>.pt
  runs/
    index.csv                       # 一个实际运行索引，含全部失败/中断
    <run_id>/                       # 运行开始时创建，不预建策略/模型目录
      run.json                      # 不可变实际参数、代码/环境/权重/数据指纹
      steps.jsonl                   # 更新、来源UID、曝光、loss/LR/梯度
      events.jsonl                  # 启停、检查点、失败/中断
      stdout.log
      stderr.log
      checkpoints/                  # 0/每500/阶段最终完整状态
      evaluation/                   # 实际评价才创建：指标/必要轨迹/图表
  cache/                            # 确实需要时才建：有界且可重建，不含唯一证据
```

目录设计不授权环境安装、依赖解析、模型下载、建训练骨架或任何GPU运行。这些仍分别由TODO中的准备/实验任务给定输入、产物与资源边界。

### 文件边界

- 人看的研究意图、方法和待定项只改docs/plan四份文件；docs/paper只保存真实文献PDF和一个索引。不要重建literature、protocols、schemas、results、history、版本化Prompt等平行体系。
- configs/experiment.yaml是下一次实际执行的机器配置，不是第二套研究方案。详细实验未确定的字段保留未定并拒绝train；正式运行时把已解析的精确值与输入指纹写入run.json，之后不回写历史run。
- 一个Python包、一个命令入口、一个共享训练循环；任务/尺寸/策略用配置表达。inspect和dry-run不做参数更新、推理或test评价，train/evaluate只有显式命令和对应协议齐备才可运行。
- 环境的可兼容范围由pyproject表达，真正安装的精确版本由requirements.lock锁定；未开始代码任务前两者均不生成。
- 基座权重只存weights，实际训练/评价产物集中runs，不额外复制到results或为论文复制另一套完整权重。检查点和原始日志不入Git；将来论文只引用必要汇总与原run入口。
- 研究仓库不存真实/合成原图、视频、GT、号码裁剪和多套训练RGB；G盘数据用绝对canonical入口引用，任务派生标签/缓存与原GT区分。
- 一项run的不可变参数和原始记录是复现证据，不能因“少文件”删掉或合并丢失失败；只禁止方案副本膨胀。run_id实际启用时统一编码任务、模型、策略、来源面板、训练seed，避免用best/new/final反复覆盖。

结构Gate：以上职责在代码准备任务中落实，首次只创建确实用到的文件；没有消费者的空目录不建。准备结束交付后停止，不自动补齐全树或启动运行。

## 哪些可以先做

在完整数据集和全部实验参数尚未定稿时，可以做独立环境、有限公开权重下载、训练代码框架、CPU夹具检查、数据结构盘点和整理脚本dry-run。已完成且无人写入的真实数据可单独整理，但改变媒体位置前必须有目标结构/迁移清单，不是无方案任意移动。

合成仍在渲染时，不运行训练、GPU推理、CUDA基准、显存探测或大规模并行预处理；也不加载一批大模型占用系统提交内存。环境安装和下载采用轻量有界任务，容量/内存影响生产时停止，不调整UE进程、护栏、页面文件或驱动。

## 硬件利用与提速原则

目标是在相同科学协议下减少端到端时间，充分使用4070Ti SUPER的Tensor Core、CPU数据准备与SSD；判断依据是完整更新/推理吞吐及总wall，不只看GPU利用率。不得靠减少更新、曝光、输入、日志或检查点来声称提速；所有性能选项在正式运行前固定并写入run。

1. 训练优先预检AMP BF16候选，raw参数、EMA及优化器保持FP32；BF16通常不需要GradScaler，保存状态中scaler=null。FP32保留为数值参考，不能因为早先候选写了FP32就让所有正式训练一直不用Tensor Core。若BF16实际不兼容，按一次明确数值修订选择FP16+scaler或FP32，并记录原因；不同策略不能各自挑精度。GPU支持与数值/吞吐验证等渲染释放后进行，不在本轮初始化CUDA。[PyTorch AMP](https://docs.pytorch.org/docs/2.8/amp.html)
2. 每任务/模型尺寸从microbatch 1/2/4/8/16的有限候选逐级探测，保持有效batch16、累积数=16/microbatch。用小型train片段（包含较密集目标）选择端到端最快且有余量的值，不只选占满显存的值；每任务/尺寸最多10分钟。该型号所有R/S组和seed共用同一结果；跨尺寸若microbatch不同，如实记录批内统计与归一化的差异，不称只改变参数量。
3. 数据加载初值workers=2，可在s代表模型上只比较2与4一次，取端到端更快且不超RAM预算者；pin_memory=True、persistent_workers=True、prefetch_factor=2和non_blocking传输作为预检候选。固定CPU线程与prefetch上限，Windows worker使用可pickle的模块级函数与主入口保护。prefetch不能推进“已提交曝光”位置，恢复以消费/提交边界和独立增强随机流为准。[DataLoader](https://docs.pytorch.org/docs/2.8/data.html)
4. 缓存优先解析好的标签、索引和确定性中间结果。4K原图反复解码确为瓶颈时，允许在GPU工作前独立规划有界SSD缓存：缓存原管线中确定性缩放阶段的无损数组、原shape/scale和来源UID，验证与即时计算像素一致后全组复用；不有损重编码、不复制整个4K库、不把缓存当新数据。实际容量先核，训练期缓存另预算，不挪生产SSD池，当前不批量生成图像缓存。
5. 检查点可采用先一致快照到CPU、单writer异步写盘、最多一个待闭合快照的方式减少GPU等待；快照必须对应同一已提交更新的raw/EMA/优化器/RNG/采样位置，队列满则等待，阶段结束全部闭合。不得少于最新获准每500更新检查点或用不断变化的Tensor引用充当快照。初步代码只建立契约，未实现的异步保存不能称已启用。
6. 推理使用model.eval与inference_mode，优先复用已验证AMP模式、固定输入和批量前端；给追踪器仍按原序列/时间顺序逐帧消费结果，不混状态或改变FPS。记录预热、数据读取、网络、后处理、保存的分别与总耗时。torch.compile、TensorRT/ONNX和多run同卡并发不作为当前默认；确有大批推理收益时再做有限对齐验证与独立预算，不额外导出15套文件。

确定性、数值模式、TF32/cuDNN选项、microbatch和workers在正式run前显式记录；启用快速实现后仍保持定义好的损失、采样和评价范围。模型版本/输入未变时复用已通过的性能配置，不逐策略重搜。设备不支持/数值不稳定/额外开销大于收益时如实记录并停止该加速分支。

## 主线证据与非主线临时产物

| 类型 | 保留规则 |
| --- | --- |
| 正式训练、预算校准训练、正式推理/评价，包括失败中断 | 原始loss/LR/梯度、UID/曝光、优化器/RNG、检查点、stdout/stderr、预测轨迹和原指标完整保留，服务论文复核 |
| 正式输入资产 | 公开基座与来源manifest、依赖锁、数据索引/原GT和一次必要迁移收据保留；它们不是测试生成模型 |
| CPU单测、toy诊断、安装验证、非主线性能测试 | 在会话展示摘要/必要错误后，直接删除本任务临时日志、诊断模型和验证报告，不迁到Saved或另建archive留存 |
| 测试源码、初步实现与实际所需配置 | 保留便于复用，测试运行在TemporaryDirectory中，不自动建立持久runs |

运行入口必须区分main/calibration与diagnostic/test。非主线产物只用临时上下文，关闭文件/消费者后按精确清单清理；不要删正在下载的part、第三方共享缓存、生产证据或正式训练失败记录。完整阶段结果在TODO/本文件简短登记即可，不为每次CPU测试另建JSON报告。环境安装的临时包/本任务缓存和安装输出也在展示验证结果后清理，包管理必要元数据与环境本身保留。

## 环境准备任务

本次只读硬件为RTX4070Ti SUPER、约16GiB显存；已安装CPython3.12。优先在项目.venv建立独立Python3.12环境，先核Windows/PyTorch/CUDA wheel兼容性，再锁定torch、torchvision、ultralytics及必要依赖；不要共享UE的Python3.9或旧训练环境，不升级系统驱动/全局包，不安装不必要的全CUDA工具链。

安装规则以[PyTorch官方入口](https://pytorch.org/get-started/locally/)和[Ultralytics安装说明](https://docs.ultralytics.com/quickstart/)为准。只在本任务执行时确定精确版本并记录requirements锁定信息，不照抄网页的最新版或安装期间自动滚动升级。现在可验证CPU导入与依赖一致；CUDA验证等UE释放GPU后。

当前已安装：Python3.12.13、torch2.8.0+cu128、torchvision0.23.0+cu128、ultralytics8.3.237、numpy2.2.6、opencv-python4.12.0.88及本地editable futsalmot-research0.1.0。requirements.lock锁定37项真实第三方包，torch/vision绑定官方URL与完整SHA，无已删除临时file路径；Torch编译CUDA12.8不代表已验证实际GPU运行。

独立CPU复验：三包导入、微小张量、torchvision NMS、完整检查点精度/优化器状态读回、Python/NumPy/Torch CPU RNG续接和全部32项测试通过，0跳过；pip check无依赖冲突。验证进程屏蔽GPU并拦截CUDA初始化，未加载公开/真实模型、未参数更新或推理。临时配置/夹具/玩具权重自动清理；安装包/cache/log/status/helper已展示后精确删除，按最后硬链接口径释放约3.49GB，未动共享缓存和生产。环境约7.41GiB，含15份公开权重准备总量约8.08GiB。

产物仅为实际使用的.venv、一个依赖锁文件和简短环境记录，完成即停止。 空间准入应保护活动SSD池剩余可增长量与最低余量，不能在文件已占用磁盘后又完整重复预留整个池；依可靠物理去重/现有账本核算，不擅改生产护栏。包下载/安装临时文件计入准备任务资源预算；不要为将来可能用的模型预建多套环境、容器或空目录。预估环境/缓存/权重峰值后再执行，超预算报告而不是删历史数据。

## 公开基座模型下载任务

基座仍为YOLO11，不因官方示例更新自动换YOLO26/其他代。按[YOLO11官方模型表](https://docs.ultralytics.com/models/yolo11/)核验来源、任务、文件名、发布版本与许可。

优先取得yolo11n.pt、yolo11s.pt；完整已议范围最多15个文件：n/s/m/l/x各自detect、-seg、-pose。其余架构、分类/OBB、其他YOLO版本不自动加入。只下载原始公开权重，不借predict/train命令自动拉模型并运行，不加载全部模型做GPU检查。

按实际任务创建weights/yolo11，保留一个manifest记录URL、发布标识、文件大小/SHA和预训练来源。单文件不完整允许一次基础设施重试，核验后停止；不得无限换镜像或添加模型。具体下载规模由执行任务给定，15是允许规划上限而不是立即全部下载的指令。

本轮下载交付：Ultralytics assets v8.3.0公开发布，三任务五尺寸15份，共725,808,473B（692.18MiB），全部文件大小/ZIP成员CRC/本地SHA通过；目录仅15个.pt及一个manifest。根会话独立重算15份SHA均与清单一致。发布API digest=null，本地SHA用于固定收到的字节，不能冒称与独立官方digest匹配。x-seg/m-pose的短响应各经一次同源续传后通过，错误与来源信息仅集中记录在必要manifest，不留独立下载日志/诊断报告。没有Torch加载、pickle执行、训练/推理或GPU操作。

## 训练代码准备任务

只创建当前需要的src与一个最小配置入口，不提前创建每模型/策略/run的空文件夹。一个共享入口支持inspect、dry-run以及未来显式train；默认不得启动训练或打开test。各模型只通过配置区分，不复制15份训练脚本。

实现内容：读取G数据索引及train/val引用、标签适配、R/S有序采样、按更新计数、完整记录和检查点恢复边界。R+S不是随意把两域目录合并后称预算相同；每步/累计真实与合成曝光要能核对。YOLO11-pose按整图范式适配，人物标注数量和整图曝光分别记；top-down模型扩展另按ROI定义，不能把两者混为同单位。

不改原GT；新的适配器对小夹具做一次必要检查，复用生成验收。可使用自造小型CPU夹具验证路径解析、mask/COCO点序与采样计数，不打开test图像做方法开发，不做全量标签转换或媒体缓存。

主线训练每步记录loss/LR/梯度、UID/域/重复使用与耗时；保留stdout/stderr、失败与负结果。0/每500/阶段最终保存完整模型、优化器和RNG，预定义最终步，不挑best。借鉴旧日志规范，不复用旧关联器固定1000步/3R1S schema冒充新图像任务预算。CPU夹具最多初检加一次针对修订，交付代码和检查结论后停止。

本轮实际交付：已新增adapters.py，支持真实train/val角色与ignore分离、冻结UID、原生BBox裁边/归一化，以及合成COCO17版本下mask半开BBox/null身份保留。17项CPU测试通过；允许小样核对SoccerNet train一帧15目标+3ignore、SportsMOT train一帧8目标（含合法ID0）、已通过合成train一帧10目标，未读test内容或图像、未改GT。适配接口仍明确native_yolo_training_ready=False；ignore负分类与评价过滤、真实网络循环及高吞吐DataLoader尚缺，不能直接把派生yolo_labels喂原生loss称已处理ignore。单帧核对函数逐次扫描GT/UID，仅供契约验证；正式DataLoader须按冻结clip一次解析建立有界标签/UID索引，不能逐图重扫GT。

本轮实际交付：inspect/dry-run、采样/调度预计算、BBox与Pose/mask契约，完整状态同步保存/可信CPU读回、Python/NumPy/Torch CPU RNG恢复，以及正式合成发布的小型绑定/精确逐文件迁移dry-run。32项CPU测试通过，0跳过；当前生产根与G5尚未发布，真实只读规划返回AWAITING且没有媒体枚举/写入。迁移执行、ignore负分类与评价过滤、网络训练循环、高吞吐DataLoader和完整run恢复集成仍缺；train/evaluate明确BLOCKED，不能靠修改布尔Gate启动。

## 三任务训练流程与具体参数候选

以下是可审阅的首版数值协议，不是已执行结果，也不是最优参数保证。用户本轮要求细化方案，当前不训练。训练数值/criterion适配、合成最终图像池、真实Pose/mask GT及完整存储预算未闭合，不能直接复制表格启动全部作业。

### 先区分epoch、batch和更新

epoch是使用一遍本任务唯一整图训练池；一次optimizer update才是调整一次参数。每次更新固定累计16张整图，这叫有效batch16；显存内一次送入的是microbatch，两者不同。

YOLO11 detect、pose、seg均按整图训练，不是先用另一个检测器裁球员后再训练本轮YOLO-pose。Pose人物帧/关节点数、seg实例数分别登记，但不能用它们除batch当图像epoch。

Ultralytics参考实现存在nbs隐式累积、warmup期间动态accumulate及optimizer=auto自动改选优化器等行为；本方案要求显式优化器、nbs=16、固定累积，以实际提交的optimizer update记预算。权重发布v8.3.0不等于软件已固定为8.3.0，执行时须锁定精确安装版本，并逐项复核源码语义。[参考默认参数](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/cfg/default.yaml)、[参考Trainer](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/engine/trainer.py)。

### 合成整图池的确定规则

首BBox实验N_S使用最终正式发布中全部train seed、实际机位和实际帧号的RGB集合；不按模型成绩筛选、剔除困难视角或改为固定16000，合法空目标图也按完整标注负图保留。实际索引绑定后才填N_S与更新预算，16000仍仅为算例。dev只用于合成诊断，真实val用于已登记预算/参数选择；所有机位、重叠窗口继承原seed split。复用旧捕获时各seed/帧原dataset_version保留，派生样本UID注明来源与派生规则，不把旧v3字节改名成v4。

### 所有任务共用的阶段安排

1. PT：对应任务的公开预训练模型，不更新权重，作为起点参照。
2. 首版候选合成阶段E_S=20个整图等效轮，真实阶段E_R=50轮。旧4000S/1000R更新退出正式默认，只能解释旧短预算设想。
3. 先在s规格为本任务做一次有界预算校准：同公开起点、seed17，分别独立运行R50轮和R100轮两个候选，各只取其末步EMA在val的主要AP。100更高则选100，否则选50；数值完全相同也选50。此为选择预算，不是非劣效或等效检验，不预设差异阈值。
4. 两个校准run都须数值、数据和记录有效；出现NaN/训练失败则校准未完成，不以剩余成功候选冒充两候选比较。两个校准run必须分别从公开起点开始；100轮run第50轮不是独立50轮协议，因为LR时间表不同。校准权重不续作正式结果、不放进三seed主表，两个结果与成本都保留。
5. 每任务校准最多两次，执行前仍需数据/参数/时间/存储准入。无条件的Pose/seg不做校准。选择一次后对该任务所有尺寸/方法/seed共用E_R；不再按n/x表现追加轮数或挑best。20轮S是受控预算候选，不声称已经证明S充分收敛。

设N_S、N_R为该任务实际合法唯一整图数，B=16：

```text
U_S = ceil(20 * N_S / 16)
U_R = ceil(E_R * N_R / 16)      # E_R经一次校准选50或100
S曝光 = 16 * U_S
R曝光 = 16 * U_R
```

共享采样器按固定洗牌循环消费完整池，最后一批用下一轮开头补齐；每阶段最多多15张曝光，记录重复UID，不复制文件充数量。这里20/50是名义轮次，报告实际曝光/N的等效轮数，不能假称每步刚好构成原生dataloader epoch。不同任务N_S不必相同，Pose要按其合法整图池重新计算，缺真实GT时N_R未知。

### 五种训练策略如何对齐

| 组别 | 第一阶段 | 第二阶段 | 总更新/曝光 |
| --- | --- | --- | --- |
| R | 无 | U_R步R | 16U_R张R |
| S | U_S步S | 无 | 16U_S张S |
| S→R | U_S步S | U_R步R | 16U_S张S＋16U_R张R |
| R→R | U_S步R | U_R步R | 16(U_S＋U_R)张R，重复使用同一真实池 |
| R+S | 前U_S步精确混合 | 后U_R步继续精确混合 | 与S→R累计R/S曝光相同 |

混合组不能沿用旧固定4S:1R。令总步U_T=U_S+U_R，零基第i步若floor((i+1)U_R/U_T)>floor(iU_R/U_T)则取R，否则取S；全程恰好U_R个R批次和U_S个S批次。在全局U_S边界，所有长预算组统一重置阶段优化器/LR/EMA，不能只有S→R拥有新的阶段调度。

R主序列、S序列及各自增强随机流按任务/域/训练seed/曝光序号生成；R与S→R末阶段、混合组的R序列一致，各策略S序列一致。R→R首阶段是额外R重复曝光，另记序列，不冒充新增真实标签。匹配更新和图像数不等于匹配全部FLOPs或训练耗时。

### 已知数据的轮数算例

下表只假设N_S=16000、E_R=50；合成最终池未确认，不当成运行事实。

| 项目 | SoccerNet N_R=2100 | SportsMOT N_R=2933 |
| --- | ---: | ---: |
| U_S | 20000 | 20000 |
| U_R | 6563 | 9166 |
| R实际曝光 | 105008 | 146656 |
| R等效epoch | 50.0038 | 50.0020 |
| 长预算总更新 | 26563 | 29166 |
| S→R或R+S累计S/R | 320000S＋105008R | 320000S＋146656R |
| R→R累计R | 425008 | 466656 |

若选100轮，SoccerNet U_R=13125、SportsMOT U_R=18332；不能因为来源不同而直接复用另一个来源的更新数。两个面板仍源内独立，不自动都跑完整矩阵。

### 公共数值参数

这些是首轮明确候选，资源/数值预检通过后再冻结；不按测试结果优化。

| 参数 | 合成/长预算第一阶段 | 真实终阶段或独立R |
| --- | --- | --- |
| 输入 | 960×960 letterbox，保持原宽高比 | 相同 |
| 有效batch | 16张整图/optimizer update | 相同 |
| microbatch/累积 | 在1/2/4/8/16候选中预检，每型号选最快安全值；累积=16/microbatch | 本型号/任务所有策略与seed固定一致 |
| optimizer | 显式SGD，Nesterov=True | 新建同类优化器，不带第一阶段momentum状态 |
| 初始峰值LR | 0.01 | 0.001 |
| 最低LR倍率 | lrf=0.01，按实际update cosine衰减 | 相同，因此终点LR为0.00001 |
| momentum | 0.937 | 相同 |
| weight_decay | 0.0005，仅实施约定的weight组；bias/归一化组为0 | 相同 |
| warmup更新数 | min(500,max(1,floor(0.05U_stage))) | min(100,max(1,floor(0.05U_R))) |
| warmup形状 | LR从0到峰值，momentum从0.8到0.937，不动态改变累积数 | 相同规则 |
| 训练层 | 不主动冻结backbone/neck/head；框架固定投影层按实现保留并记录 | 相同 |
| 数值模式 | 优先AMP BF16候选；FP32作数值参考，GPU预检后冻结 | 本型号/任务各策略一致，不静默切换 |
| 梯度裁剪 | 完整累积后L2 clip=10，记录裁剪前后范数 | 相同 |
| seed | 当前BBox探索仅17；29/43为待进度决定的候选，deterministic=True | 同一已启用范围保持一致 |
| 其他 | nbs=16；workers初值2，有限预检2/4；不默认全图RAM缓存；rect=False；multi_scale=False | 数据加载与数值策略预先固定 |
| 早停 | patience=0并禁用best驱动的终验，按实际update终点停止 | 相同 |

第一阶段LR终点为0.0001。warmup期间所有参数组的LR规则显式实现，不暗用默认高bias warmup LR；weight_decay不再被nbs或自动batch静默改写。原生epoch scheduler不能替代本协议按optimizer-step的调度。

离散调度以每阶段成功更新j=1..U计，阶段0只保存初始状态。K取上表值后再截到U−1（本版U≥2），阶段0的LR=0；j≤K时LR(j)=lr_peak·j/K，j>K时：

```text
LR(j) = lr_peak * [0.01 + 0.99 * (1 + cos(pi*(j-K)/(U-K))) / 2]
momentum(j) = 0.8 + (0.937-0.8)*j/K    # j<=K
momentum(j) = 0.937                    # j>K
```

因此第K步恰为峰值，最后一步恰为峰值的0.01，不因0/1基下标或原生epoch偏移不同而改变。各参数组遵守已声明规则，weight_decay不随warmup另乘系数。


有效batch16仅指每次更新消费16张图，不保证与一次前向16张在BatchNorm统计上完全等价。microbatch按任务/型号预检并对该型号所有策略统一，跨型号差异明确报告；不能为x悄悄降低输入、有效batch或记录量。显存不足先停止资源预检，不将n/s能运行推断为x/seg也能运行。

损失缩放采用一个明确候选：若参考criterion返回L_j已经乘该microbatch的图数，则本次有效batch损失为sum(L_j)/16，每个microbatch反向L_j/16；若实际版本返回均值，则先乘本微批图数得到同一约定。一次zero_grad后累积满16图，再裁剪与optimizer.step，不再额外除累积次数。必须核实实际返回语义，避免双重缩放。

数值对照采用相同microbatch分组的参考求和，而不是强求与一次前向16图逐值一致；除BatchNorm外，criterion内部的批内目标归一化也会产生分组差异。该定义及LR均属于本版候选，不冒称原生默认数值完全不变。绑定实际代码并通过一次手算/参考循环检查前，不签“参数已可运行”。

### 增强与采样

| 增强参数 | 首版三任务共同值 |
| --- | --- |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.4 / 0.4 |
| degrees / translate / scale | 0 / 0.05 / 0.25（缩放约0.75–1.25） |
| shear / perspective | 0 / 0 |
| fliplr / flipud | 0.5 / 0 |
| mosaic / mixup / copy_paste | 全部0 |
| close_mosaic | 0，因为mosaic关闭 |
| 多尺度/TTA | 不启用 |

关闭多图增强是首轮控制选择：否则一个“16张batch”还借入其他原图，R/S曝光必须额外统计。后续研究增强本身需另设问题，不在本轮顺便启用。val/test仅确定性letterbox和归一化，不做训练增强。S/R保持同一增强规则，图像及全部目标几何同步变换。

### BBox检测怎么训

1. 从yolo11n/s/m/l/x.pt各自公开权重开始，主规格s；读取整图及场上球员含门将BBox。
2. 为保留原公开检测头，首版维持COCO80类结构、person行0，训练目标映射0、评价仅本研究person范围；不要同时single_cls=True或让框架自动随机重建单类头。若将来改为单类头，要明示哪些参数重新初始化并另定协议。
3. loss gain候选box=7.5、cls=0.5、dfl=1.5；不加入Pose/mask/号码损失。
4. 先完成源内R预算校准，再冻结R/S→R及策略对照，用同一真实UID与固定阶段规则训练，取末步结果。

源内球员/门将、裁判、staff、ball与未标人群必须遵守DATASET.md和索引。写一个ignore字段不代表YOLO原生loss/evaluator就生效；含ignore的适配按上述用户已批准的role_policy_v2与正目标优先实现；源非目标列表不等同最终训练ignore。没有显式ignore真值时记录来源局限，不凭空造负例/区域或把自定义范围称官方排行榜。

### COCO17 Pose怎么训

Pose是论文核心任务；以下仅保留早期方法/参数候选，真实COCO17 GT与最小实验设计仍待后续详细规划，当前不列TODO、不自动开训，不能将这些候选视为已成熟协议。

1. 各自从yolo11n/s/m/l/x-pose.pt开始，原person类及17点头保留。训练输入仍是完整比赛图，联合预测BBox和关键点，不先裁人物训练本轮YOLO-pose。
2. kpt_shape=[17,3]；点序为nose、双眼、双耳和十二身体关节。v0标准xy=0且无坐标监督；v1/v2均有坐标监督。参考YOLO的第三输出监督点是否标注，不是物理遮挡/可见二分类，不能说它自动学会v1/v2区分。
3. 水平翻转同时采用COCO17 flip_idx=[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]；出画点与坐标/存在掩码同步，不把原投影侧车变成GT。
4. loss gain候选box=7.5、cls=0.5、dfl=1.5、pose=12.0、kobj=1.0；标准COCO17 sigma/有效点语义一次核对。
5. S阶段20轮、R阶段50/100轮选择与公式同上，但N_R/N_S必须是本Pose合法整图池，另报人物与有效点数。图内所有目标人物BBox保留；优先使用全部目标实例均有明确Pose监督有效性的整图。
6. 仅部分人物补了Pose时，不能把未补标人的17点简单置v0交给原生kobj，因为这会产生全零存在目标，而不是完全忽略。必须在训练派生视图明确逐实例pose_supervision_valid掩码，同时屏蔽未补标实例的坐标和kobj损失、仍保留BBox监督；这是适配信息，不改生成契约。若所选实现不支持该掩码，该整图不进入正式Pose视图，资格在冻结N之前确定，不临时删人或把其当背景。确认可标且某点v0与整个人未补标分开登记。

当前真实COCO17 GT未落实，Pose的R校准、R/S→R和真实OKS AP仍BLOCKED。只能先完成适配/参数规划；不能用已下载pose权重或合成内部成绩宣布真实迁移已验证。[COCO17参考配置](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/cfg/datasets/coco-pose.yaml)、[loss语义](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)。

### 人物实例Seg怎么训

Seg是论文核心任务；以下仅保留早期方法/参数候选，真实人物mask GT与最小实验设计仍待后续详细规划，当前不列TODO、不自动开训，不能将这些候选视为已成熟协议。

1. 各自从yolo11n/s/m/l/x-seg.pt开始，整图联合预测BBox与每人物可见mask，原COCO类结构保留，目标person映射0。
2. 原整数instance mask保持只读；同一人遮挡后多个不连通区域仍为一个实例，背景0及ID映射明确，不把号码mask当人体mask。
3. 首版mask_ratio=2、overlap_mask=True；相比参考默认ratio4保留更细训练栅格，但它是明确候选改动，必须先测16GiB资源与小目标mask适配。mask在resize时用正确离散采样/实例排序，不能把整数ID当连续灰度插值。
4. 不能把多连通mask直接转成桥接多边形后称无损。优先让适配器读取原实例mask/RLE并向所选训练实现提供正确目标；若接口无法保留碎片/孔洞，记录转换误差并停止正式Seg准入，不为方便改原GT。
5. 参考loss gain box=7.5、cls=0.5、dfl=1.5，mask项同样复用box gain；不能伪造一个不存在的默认seg=参数。若引入独立mask系数，明确改代码及全部策略相同。
6. 同样S20轮、R50/100轮选择，按有真实人物mask的整图数算N_R；每图全体实例标注，报告实际实例数量，不能按mask数量算epoch。

真实人物mask未取得，Seg正式R训练/校准及真实Mask AP保持BLOCKED；不能用BBox AP替代分割效果。[原标注转换参考](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/data/converter.py)。

### 检查点、验证与最后模型

每阶段0/每500个实际更新/阶段最终保留完整raw model、EMA、optimizer、全部RNG与采样位置；FP32与AMP BF16的scaler明确为null，AMP FP16才保存实际scaler状态，不造不存在的状态。阶段切换另保存重置后的0状态，不覆盖前阶段最终文件。

主评价固定本阶段末步EMA，不从raw/EMA/best中取高者。候选EMA decay=0.9999、tau=2000；每阶段从当前raw初始化并重置EMA更新计数。S→R承接S末步raw，再重建R优化器/EMA，全部长预算组同规则。raw也保存用于复核，但不能事后换主评价模型。

EMA在成功optimizer更新u之后执行，d(u)=0.9999·(1−exp(−u/2000))；浮点参数及浮点buffer按EMA_u=d(u)·EMA_(u−1)+(1−d(u))·raw_u更新，非浮点buffer从raw复制，状态以FP32完整保存。u是本阶段成功更新数，不是microbatch数或epoch；阶段0的EMA直接复制raw。

原生Trainer的last/best保存/optimizer剥离和最终best验证不满足本项目完整状态及末步规则，须保留独立科研检查点并覆盖默认final_eval逻辑，不能把原生last.pt直接称完整科研状态。

未来未冻结实验val仅每个阶段0及阶段最终执行，不再每1000步验证；独立R/S为2次、两阶段为4次，完整保留阶段切换两侧不同EMA状态，不改变data split。imgsz960、conf=0.001、IoU NMS=0.7、max_det=300、无TTA及原类别/ignore/标准与辅助AP口径保持；主BBox COCO API默认maxDets100，辅助AP300单列。全部实际评价的原始输出/失败保留，主结果固定末阶段末步EMA，不读test。已冻结活动run按原验证计划继续，不热改。

### 正式矩阵与资源硬Gate

早期候选矩阵设计保留供后续讨论：s五策略×seed17/29/43＝15次，n/m/l/x各R与S→R×三seed＝24次，每任务理论39次正式；PT不训练、两个校准另计。当前不将39次作为执行目标：只启用BBox s/seed17的R、R→R、cached S→R三核心；多seed待后续进度/资源决定，其他模型规格等待s探索完成查看效果再决定。Pose/Seg尚无成熟方案，不能从该理论矩阵自动复制41项待办。任何后续有限作业必须另行确定启用范围并通过完整Gate，不恢复旧132/177上限。

按当前N_S24149、S20/R50、每500检查点，s五策略三seed约924份完整状态；按已测s单份114192047B粗估98.27GiB。该估算不含原始预测/日志/瞬态与更大模型，也不替代新模型实际测量。旧每1000步验证口径下s15候选约480次750图评价、原始预测保守上界约2229.75GiB；新每阶段0/末规则下该未启用候选理论48次评价、按每图上界6650344B为约222.97GiB原生预测。924份完整checkpoint及逐步日志等开销不随之减少，也不追溯重算旧run；这仍不是整个矩阵准入许可。原始输出无损压缩实际大小与理论上界分开记录，不能据分数/压缩率乐观承诺全部任务。

当前R50原规则闭合后按原流程校验归集至G；新独立R100按500规则工作集保守约89.17GiB直接写G，G需完整上界加100GiB，D仅小控制上界加100GiB，实际按冻结公式重算。每个作业闭合且小控制证据封闭、无遗留写者后，才准入下一有限作业。先只执行已启用的R50/R100校准；正式15/39仍需完整工作集、累计G留存与时间Gate，未获得无限下载/重跑/扩展授权。不借生产预留或删除旧失败证据通过资源门禁。

仍待冻结的项目：已装精确软件的criterion与调度适配、最终S及各任务R整图池、预算校准结论、统一microbatch/数值验证、完整checkpoint路径/时间/容量，以及有限正式作业表。以上未闭合，状态为“环境与CPU工具通过；网络/评价实现及数据/资源Gate未闭合，未开训”。

## 追踪、多任务与扩展

第一步不同检测器接相同ByteTrack主参照/OC-SORT辅助配置，报告HOTA及DetA/AssA/IDF1/IDSW。它检验检测收益传入MOT，不声称训练了关联器。各序列从空状态开始。

第二步才在固定检测输出下比较BBox运动基础、+预测mask、+预测Pose、+预测号码、+组合；接口/代价/缺失回退/共享编码器/阈值必须先确定，不能临时调到上涨。真值不能充当推理线索。

分割/Pose/号码须先另定真实GT取得或人工补标的资源方案。GT缺失的分支BLOCKED，不能只用合成内部成绩宣称真实迁移。五尺寸范围保留为候选，但不会自动开启无监督的39次任务块。

跨架构、连续ID模型、近邻合成对照、低真实量曲线是有条件扩展。原核心132/总177是候选矩阵上限而非执行目标；不为凑满次数或寻找正结果无限加实验。每个扩展先写问题/对照/预算，在本文件原地审阅后才启用。

## 终止Gate

准备：环境/权重/CPU代码任务各自完成既定产物即停止，不顺便训练。GPU预检仅在UE退出/任务确认释放后执行，使用train小样、限定诊断更新，预检权重丢弃；初检加一次针对修订，仍失败则停止。

正式训练：只执行冻结作业；明确基础设施错误最多同协议重试一次，人工取消或低分不是重跑理由。完成全部规定更新、记录/检查点及预算核对即PASS，没提升也可PASS。缺GT/资源/参数则具体BLOCKED。

正式评价：全部方法与分析范围冻结后统一执行已登记test，保留全体比赛/seed原值和负项；不能根据结果改split、筛best、扩矩阵。结束即时更新实际runs记录与本TODO状态，不生成一套新方案归档。


</details>


## SHORT-PAIR-FOLLOWUP-001：30分钟有限接续与实际准入工具准备

同一bbox-sr-mix已实际更新ACTIVE/每30分钟，原UNTIL20261107T235500Z和总上限不变；有限链mixed完整收口→新短SR→新短RR，RR无分数条件。不是已经启动后两组，当前mixed继续原冻结15cp。tools/short_pair_preflight.py SHA b5fc34fdcfb4aad9b4d488d4062255dff9f9463a2fac6f2af0f306a0e5253e36已CPU语法与独立只读审阅，GPU未执行；每策略最多60秒，新两项合计120并沿旧475.376累计不重置。root须以完整helper进程寿命保守计入清理时间，为后一项冻结实际剩余额度，不能只用cleanup前admission elapsed或延长原准备界限。

helper实做first1磁盘cp/可信读回、first2、fresh恢复重放first2及loss/状态/实际梯度exact，再以实际当前raw重建final5063/新SGD EMA并更新首R步；明确仅两步first机械探针，不假称实际1500全阶段。前项小control逐SHA、可信owned全部退出和GPU释放由root先做，helper仅补两小收据引用核验；失败时锁可能先退休，必须等其worker/controller/wrapper完全退出后再派发。新SR/RR分别对应真实strategy admission，不能复制改字当实测。

规范命名候选只读检查：同日S2R a01已存在且为弃用S3000诊断，不能覆盖；新S2R a02和新R2R a01的D/G路径均不存在。首次新设计占用a02是明确已知命名冲突消解，不是旧诊断恢复或基础设施重试；实际派发时仍用canonical_group_name按真实带时区时间生成并复核冲突，不猜增编号。新组未创建/未GPU/未正式更新。所有接口/预算/源集合和闭合步骤已补入同一跟进，无需因既有执行授权再次询问。


## FIXED-MIX：R_plus_S_fixed_budget_seed17 COMPLETE

UTC 2026-11-03T08:37:12.741552+00:00。单阶段独立公开YOLO11s/seed17，6563/6563更新；实际R/S各52504，总曝光105008/105008。每有效16含8R8S、两个micro8各交错4R4S；SoccerNet train2100/val750/S24149，960FP32/workers2，单SGD/EMA，原100步warmup与final .001 cosine至.00001，原损失累积、不另加域权。15计划完整cp、2次实际真实验证、逐步与原生预测及失败全部保留，不best/test。固定总曝光比较R，真实曝光减半；单seed有限开发证据。结果：{"state": "COMPLETE", "pid": 28200, "create_time": 1793690435.9826293, "update": 6563, "updates": 6563, "phase": "final", "phase_update": 6563, "exposures": {"R": 52504, "S": 52504}, "schema_version": "bbox_fixed_budget_mix_v1", "job_id": "R_plus_S_fixed_budget_seed17", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_RS_seed17_20261103_a01\\R_plus_S_fixed_budget_seed17", "terminal": {"tag": "step006563_final_final", "update": 6563, "main_metrics": {"AP50_95": 0.5393183988067314, "AP50": 0.9570249787145262, "AP75": 0.5460631695185958, "AP_small": 0.3819022472769935, "AP_medium": 0.5286679643890724, "AP_large": 0.6039763825979725, "AR_max1": 0.054625905068382954, "AR_max10": 0.45919147224456963, "AR_max100": 0.6293443282381335, "AR_small": 0.42519685039370075, "AR_medium": 0.6313390663390664, "AR_large": 0.6529677419354838}, "raw_prediction_bytes": 4239591398, "images": 750, "wall_seconds": 348.5620000000017, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step006563_final_final", "update": 6563, "main_metrics": {"AP50_95": 0.5393183988067314, "AP50": 0.9570249787145262, "AP75": 0.5460631695185958, "AP_small": 0.3819022472769935, "AP_medium": 0.5286679643890724, "AP_large": 0.6039763825979725, "AR_max1": 0.054625905068382954, "AR_max10": 0.45919147224456963, "AR_max100": 0.6293443282381335, "AR_small": 0.42519685039370075, "AR_medium": 0.6313390663390664, "AR_large": 0.6529677419354838}, "raw_prediction_bytes": 4239591398, "images": 750, "wall_seconds": 348.5620000000017, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed17_20261103_a01/R_plus_S_fixed_budget_seed17。


## MIX-FIXED-RESULT-001：固定总量R+S完整终验与强R配对

实际UTC2026-11-03T08:37:12Z末步完成，北京时间16:37；root于2026-11-03T08:47:40.979269+00:00核验。公开YOLO11s独立起点/seed17、SoccerNet同2100R池/750真实val、S24149，单阶段6563成功更新、R52504/S52504/总105008，与纯R同6563/105008总曝光。末步EMA主COCO AP50:95=53.9318398807%、AP50=95.7024978715%、AP75=54.6063169519%；强R对应51.6511537498%/94.1099523965%/50.5687308826%，差分别+2.2806861309pp/+1.5925454750pp/+4.0375860693pp。small/medium/large绝对AP=0.3819022472769935/0.5286679643890724/0.6039763825979725。这是同seed/同已开发真实比赛下固定总预算的正开发信号，不是显著性、独立比赛泛化或标注节省证明；仍用了全2100真实标注池。旧S3000不入主结果或用于选策略。

15完整cp小收据/bytes、两val精确tag/PT与末summary、末262144B有界steps尾部6563/R52504S52504均核对；50文件790025B小control逐SHA通过、manifestSHAe202af3b87371aa04bd1aee716dadb4d5f96d9e33525a169c08deb7228f3d669，准确controller28200/CT1793690435.9826293与wrapper31292/CT1793690435.9425623真实退出，未见遗留训练writer/GPU锁；WDDM仍有桌面C+G程序，未误称整GPU进程列表为空，也未关闭用户程序。终验小收据G本组completion_verification.json；不全量读取或hash模型/NPZ/媒体。driver已有COMPLETE不重复，本条为终验与比较。

前两次root只读终验扫描分别误匹配自身-c文本与把WDDM桌面C+G列表当必须空，均在任何写入/建组/GPU前被拒；现按精确script argv/parent_pid和真实训练进程核验修正，未改变训练或科学Gate，也没有消耗GPU机械attempt。随后按最新授权准备新短SR独立a02（同日a01是弃用旧诊断），再无分数条件新短RR；不因当前混合上涨跳过既定配对或扩矩阵。


## SHORT-PAIR：short2_S_to_R_seed17 STARTED

UTC 2026-11-03T08:52:41.365328+00:00。独立公开YOLO11s/seed17，S->R1500+5063/总6563；实际已提交0/6563，R/S曝光0/0。960FP32/micro8有效16/workers2，两阶段各warm100/LR.001→.00001，原SGD/EMA/损失/增强；raw承接、新SGD EMA。0/每1000/每阶段末共10计划完整cp，4端点真实750验证及逐步/原生预测/全部失败保留，不best/test，不因SR低分取消RR。结果：{"state": "public_initialized_zero_updates"}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_S2R_seed17_20261103_a02/short2_S_to_R_seed17。


## SHORT-PAIR-SR-ADMIT-001：新短SR实际准入并唯一派发

当前混合已完整闭合后，root显式运行独立tools/short_pair_preflight.py原bytes，首SR实际PASS59.734秒/4诊断更新；2临时pt各114289775B/合228579550B已展示可信收据并精确清理，代表R/S cached/direct字段逐值一致。first1完整磁盘状态→first2→fresh公开模型恢复重放first2，raw/EMA/SGD/全部CPU CUDA RNG/sampler/loader/loss全部指标/实际梯度逐值exact。另用真实两步first后的raw重建final5063，raw保存、新SGD空、新EMA0和首R更新正确；不是已实跑1500步的阶段转换声明，生产完整边界有CPU guard与实际loop。

首4步耗时[3.077999999997701, 0.4850000000005821, 2.672000000002299, 1.2959999999984575]，稳态单步仅短窗.485s非保证。GPU peak_reserved9026142208B/peak_allocated8465099264B/free6682574848B；完整checkpoint114289775B，以更保守114293679B×1.1计10cp，6563步各2MiB、4×750(native6650344+JSON1MiB)、一次750partial和3GiB+2cp，完整新增44.09030254558GiB，G另100GiB/D小控制32MiB另100GiB，实际free通过。时间保守9472.266s<=10800；冻结max3h，实际runtime截止2026-11-03T11:52:37.757823+00:00（北京时间19:52:37），不重置/不延全局。

D runs/main_bbox_yolo11s_soccernet_S2R_seed17_20261103_a02，G:/FutsalMOT_Training/runs/同名，job short2_S_to_R_seed17，2026-11-03T08:52:28.6765556+00:00隐藏派发，wrapper16652/CT1793695948.6930575、controller5676/CT1793695948.7314644，完整D本组launch命令/当前GPU锁已核。launchSHAb82eab4b4362308c26c820cfd952c885f8001979196e74fc5ffb74787761db4b；job_tableSHA320f2bb407e765d47645afda1b6daf043029d422a3b93243346913215e13f56c；code_bindingsSHA22293ac1dd027218a35e83da97fe46fd98fe5828c8e4bb444ee6ba6a0536c5ac；admissionSHAead0a0bed92605b575837215ba7a2a0f5b8ad30c8f4cc4c1d0388502183badcd；storage_revisionSHAe8b460f5b82f9b0b19fe80dd3f55b54dce48e43c52a42d774efe57e7138b36e7。独立公开起点，不复用弃用A权重；first1500S+final5063R，总6563/24000S+81008R，10计划完整cp/4端点真实750val，首阶段0正在验证，未冒称已经提交S更新。41执行源码精确旧A40+short_pair8b692...，无Seg/Pose扩入，原formal/当前已完成mixed冻结包不改。process_lifetime_isolation_verified=false。

### PREP-TIMING-ATTENTION-001：RR机械预算待决，不影响独立SR

root没有在helper启动时持续持有进程退出计时句柄，首次观测其退出时进程已消失；无法追溯精确ExitTime，lifetime_receipt.json只能保留从派发到确认退出的102.568758965秒保守上界。这不是59.734秒实际机械elapsed被改写，也不能把等待观察的上界冒称真实GPU耗时；为遵守全进程保守计账，原新两项120秒剩17.431241035秒、原累计600秒剩22.055241035秒，无法可靠安排完整RR预检。不能只用stdout时间或cleanup前elapsed假装已精确测到退出、重置预算或偷偷省略RR独立恢复检查。

现已向用户异步请求：是否给RR另批最多60秒机械预检、训练更新/数值和全局截止不变。等待明确答复；等待不是批准。当前SR已获准且准入完整，继续其独立训练；RR仍保留为无分数条件的科学对照，但未获足够机械预算前不派发。收到许可需用独立前台有界observer从启动保留Windows进程句柄记录确切ExitTime，先闭合SR再实际RR准入，不能重复SR或将其admission改字当RR。当前30分钟跟进保留，其他独立工作未完成，不提前删除/停止SR。


## RR-PREFLIGHT-EXTRA-001：RR追加最多60秒机械预检获准

用户直接回复“允许追加最多60秒预检”。该许可只用于当前有限链后续独立RR的一次机械准入，训练1500+5063、数值、数据、检查点/评价与单项3h/全局Nov7截止不变，不重做SR预检。为覆盖既有保守退出计账，短配对准备总上限120→180秒、原全过程累计600→660秒；已用SR仍按102.568758965秒上界保留，不擦除不确定性或把等待上界当真实GPU耗时。RR实际全进程启动至退出最多60秒，helper协作窗口最多50秒预留导入/清理，必须精确记录ExitTime；到界安全停并保留失败，不强杀、不无限重试。

EXPERIMENTS仅在10.1资源段原地替换对应预算句，未追加对话段落，SHA 8f23e187304d067a36cbe288f28438a431bb2de0ed393ecdb2d03b75c121284d；PAPER保持原SHA。旧现运行SR方案68e6.../源/配置/准入/G helper原bytes不改，新RR显式绑定新方案。根tools/short_pair_preflight.py仅增加独立生命周期预算门禁及完整deadline/STOP检查，数值主体/short_pair41源码不改，helper新SHA 0860175a3dbf508e391e264da8c3a6e4d47b559cdfaae6fbaeebdf7f8e602a5f；GPU未运行。新门禁只允许RR_additional_60_v1，精确660/180/60且总账475.376+pair衔接，禁止把追加许可挪给SR或额外作业。


## SHORT-PAIR：short2_S_to_R_seed17 COMPLETE

UTC 2026-11-03T10:15:28.522257+00:00。独立公开YOLO11s/seed17，S->R1500+5063/总6563；实际已提交6563/6563，R/S曝光81008/24000。960FP32/micro8有效16/workers2，两阶段各warm100/LR.001→.00001，原SGD/EMA/损失/增强；raw承接、新SGD EMA。0/每1000/每阶段末共10计划完整cp，4端点真实750验证及逐步/原生预测/全部失败保留，不best/test，不因SR低分取消RR。结果：{"state": "COMPLETE", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_S2R_seed17_20261103_a02\\short2_S_to_R_seed17", "update": 6563, "exposures": {"R": 81008, "S": 24000}, "terminal": {"tag": "step006563_final_final", "update": 5063, "main_metrics": {"AP50_95": 0.5193718544942069, "AP50": 0.9491886402473647, "AP75": 0.5093377663553025, "AP_small": 0.35500903361665503, "AP_medium": 0.5105453019247118, "AP_large": 0.58542286953441, "AR_max1": 0.05335880933226066, "AR_max10": 0.4517095736122284, "AR_max100": 0.6148129525341914, "AR_small": 0.4031496062992126, "AR_medium": 0.6167076167076166, "AR_large": 0.6399999999999999}, "raw_prediction_bytes": 4248034110, "images": 750, "wall_seconds": 348.95300000000134, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "phase": "final"}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_S2R_seed17_20261103_a02/short2_S_to_R_seed17。


## SHORT-PAIR-SR-RESULT-001：新短SR完整验收与固定总量比较

实际UTC2026-11-03T10:15:28Z（北京时间18:15:28）COMPLETE，root于2026-11-03T10:19:11.218362+00:00完成验收。独立公开YOLO11s/seed17，1500S+5063R/总6563成功更新，实际24000S+81008R=105008，与已有单阶段R/混合总曝光相同，真实唯一标注池仍2100。960FP32/micro8有效16/workers2；两阶段各100warmup/.001→.00001，raw承接/新SGD EMA。末步EMA标准COCO AP50:95=51.9371854494%、AP50=94.9188640247%、AP75=50.9337766355%。与纯R51.6511537498%/94.1099523965%/50.5687308826%相比，差+0.2860316996pp/+0.8089116283pp/+0.3650457530pp；混合对应53.9318398807%/95.7024978715%/54.6063169519%，短SR主AP较混合-1.9946544313pp。small/medium/large绝对AP=0.35500903361665503/0.5105453019247118/0.58542286953441。

首阶段0公开起点AP38.5987112477%，first1500末EMA38.7754322615%，final0 raw重建EMA38.7537814425%，final5063末51.9371854494%。阶段间EMA与raw不同，微小差不误报状态损坏。短低LR合成阶段未出现此前长预算同样的明显真实域下降，仅是本开发配置的现象；阶段长度与LR同时改变，不能把原因单独定为过拟合或LR。相对强R仅+0.2860pp，尚缺匹配短RR/多seed/独立比赛证据，不称显著或合成专属因果。减少R重复曝光不等于减少标注，两组仍全2100真实图池，旧弃用S3000不入主表/选择。

实际10完整cp位置/阶段计数及小收据bytes、4tag/750图summarySHA、末262144B有界step尾部6563/R81008S24000已核；50文件792256B G control SEALED逐小SHA，manifestSHA4169ccbc8624f2850a0bd893b216b84e48dd469fb122aded290e42fbf5b49540；原controller5676/CT1793695948.7314644与wrapper16652/CT1793695948.6930575确认退出，无原worker/未知训练写者/GPU锁；桌面C+G正常保留。不全哈PT/NPZ/媒体，不搬整个run。G本组completion_verification.json保存终验，driver已有COMPLETE不重复。

按既定无分数条件配对，root已准备唯一新R2R canonical组main_bbox_yolo11s_soccernet_R2R_seed17_20261103_a01；RR有界独立机械预检由精确ExitTime观察器执行，仍待实际PASS与完整资源Gate，不能用SR准入改字替代。


## SHORT-PAIR：short2_R_to_R_seed17 STARTED

UTC 2026-11-03T10:22:23.362220+00:00。独立公开YOLO11s/seed17，R->R1500+5063/总6563；实际已提交0/6563，R/S曝光0/0。960FP32/micro8有效16/workers2，两阶段各warm100/LR.001→.00001，原SGD/EMA/损失/增强；raw承接、新SGD EMA。0/每1000/每阶段末共10计划完整cp，4端点真实750验证及逐步/原生预测/全部失败保留，不best/test，不因SR低分取消RR。结果：{"state": "public_initialized_zero_updates"}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R2R_seed17_20261103_a01/short2_R_to_R_seed17。


## SHORT-PAIR-RR-ADMIT-001：RR独立机械准入/精确退出计时通过并唯一派发

在SR完整终验/GPU释放后，root显式观察器执行独立RR真实准入：helper PASS43.484s，4诊断更新，包含first1完整磁盘保存/可信读回→first2→fresh公开模型恢复重放first2，raw/EMA/SGD/全部RNG/sampler/loader/loss全部指标及实际梯度exact；代表R/S cached/direct一致。两步first后用真实raw重建final5063空SGD/EMA0、首R步通过，明确不是已实际训练1500完整阶段的声明。两个114289775B临时pt合228579550B已展示并精确清，stderr0，全部owned退出/锁退休。

精确Popen句柄GetProcessTimes完整退出耗时47.058288574s<=60，observer VERIFIED_EXIT/returncode0/无owned残留；wrapper29576/CT1793701170.5108376，实际身份和所有子进程见G prep/runtime_identity/lifetime_receipt。追加准备总量保守149.627047539<=180，全过程保守625.003047539<=660，未重新计满或追加预检，不再开任何GPU机械toy。

实测4步[1.9060000000026776, 0.5, 1.514999999999418, 1.4219999999986612]，短稳态.500s不是全程保证；cp114289775B，按较保守既有完整cp最大bytes×1.1，10cp/6563步2MiB/4×750 native+JSON/一次750partial/3GiB+2cp完整新增44.09030564874GiB，G另100GiB、D32MiB另100GiB实际free通过。GPU peak_reserved9055502336B/peak_allocated8478208000B/free6653214720B；时间保守9666.812s<=10800，单项固定3h，deadline 2026-11-03T13:22:19.640839+00:00（北京时间21:22:19），不延全局Nov7。

D runs/main_bbox_yolo11s_soccernet_R2R_seed17_20261103_a01、G:/FutsalMOT_Training/runs/同名，job short2_R_to_R_seed17，2026-11-03T10:22:10.0290233+00:00（北京时间18:22:10）唯一隐藏派发，wrapper16376/CT1793701330.0496707、controller28228/CT1793701330.0891285，完整D launch命令与GPU锁一致。launchSHAb82eab4b4362308c26c820cfd952c885f8001979196e74fc5ffb74787761db4b；job_tableSHA31d5c573dceba4c5a1c98166c67e3e55301b5411aeb3363e15acabfd5e1e2c9d；code_bindingsSHA22293ac1dd027218a35e83da97fe46fd98fe5828c8e4bb444ee6ba6a0536c5ac；admissionSHA5823d1792af0add9e2eda3cc5fdf84930c685eab0f07788102cdcf9dc980eed8；storage_revisionSHA75493eeed46a3d10f9ef686200dd6a9e9daddd3c46ba43a9bfdb6909af5f47df。独立公开YOLO11s/seed17，1500R_extra+5063R=6563/105008R/0S，阶段各warm100/.001→.00001、raw承接新SGD EMA、960FP32/micro8有效16workers2，10cp/4真实750val/逐步与原生预测负结果全部保留。启动核验RUNNING first0是PT0评价，不冒称已提交R更新。此为有限链最后一个训练，不追加其他作业；Codex进程隔离未验证。


## SHORT-PAIR：short2_R_to_R_seed17 COMPLETE

UTC 2026-11-03T11:42:18.555593+00:00。独立公开YOLO11s/seed17，R->R1500+5063/总6563；实际已提交6563/6563，R/S曝光105008/0。960FP32/micro8有效16/workers2，两阶段各warm100/LR.001→.00001，原SGD/EMA/损失/增强；raw承接、新SGD EMA。0/每1000/每阶段末共10计划完整cp，4端点真实750验证及逐步/原生预测/全部失败保留，不best/test，不因SR低分取消RR。结果：{"state": "COMPLETE", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_R2R_seed17_20261103_a01\\short2_R_to_R_seed17", "update": 6563, "exposures": {"R": 105008, "S": 0}, "terminal": {"tag": "step006563_final_final", "update": 5063, "main_metrics": {"AP50_95": 0.5239314248147453, "AP50": 0.9431883152301549, "AP75": 0.5225501088128192, "AP_small": 0.35653676166936504, "AP_medium": 0.5145290429139183, "AP_large": 0.5847682452626696, "AR_max1": 0.0537510056315366, "AR_max10": 0.45401246983105387, "AR_max100": 0.6077534191472245, "AR_small": 0.39448818897637794, "AR_medium": 0.6085380835380835, "AR_large": 0.6395483870967741}, "raw_prediction_bytes": 4243185930, "images": 750, "wall_seconds": 342.79699999999866, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "phase": "final"}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R2R_seed17_20261103_a01/short2_R_to_R_seed17。


## SHORT-PAIR-RR-RESULT-001 / BBOX-FINITE-CLOSE-001：短RR完整验收与有限链最终比较

实际UTC2026-11-03T11:42:18Z（北京时间19:42:18）RR末步COMPLETE，root于2026-11-03T11:43:46.933400+00:00终验。独立公开YOLO11s/seed17，1500R_extra+5063R，共6563成功更新/105008R/0S，原两阶段各100warmup/.001→.00001/新SGD EMA/raw承接，960FP32/micro8有效16workers2；没有续SR、取best或追加更新。driver已有COMPLETE不重复，本条为根终验与全链配对。

| 方法 | 成功更新 | 实际R/S曝光 | 末步EMA AP50:95 | AP50 | AP75 |
|---|---:|---|---:|---:|---:|
| 单阶段纯R | 6563 | 105008R/0S | 51.6511537498% | 94.1099523965% | 50.5687308826% |
| 固定总量R+S混合 | 6563 | 52504R/52504S | 53.9318398807% | 95.7024978715% | 54.6063169519% |
| 新短S→R | 6563 | 81008R/24000S | 51.9371854494% | 94.9188640247% | 50.9337766355% |
| 新短R→R | 6563 | 105008R/0S | 52.3931424815% | 94.3188315230% | 52.2550108813% |

同seed17/同750真实开发图/同总105008图像曝光，混合−R=+2.2806861309pp；短SR−R=+0.2860316996pp；短RR−R=+0.7419887317pp；短SR−短RR=-0.4559570321pp；混合−短RR=+1.5386973992pp。四臂绝对全部COCO分项、status路径/SHA和比较公式收据在G本组finite_chain_comparison.json；不存在的CI不补造。短RR small/medium/large AP=0.35653676166936504/0.5145290429139183/0.5847682452626696。

解释：匹配两阶段结构与预算后，短SR低于短RR约0.4560pp，故短SR较单阶段R的小幅正差不能单独归为合成来源收益；阶段拆分/优化路径也能解释部分差异。混合在固定总计算下仍为本轮末步最高，比强R+2.2807pp、比短RR+1.5387pp，是有限正开发信号。所有组仍从全2100真实图池采样，只减少重复曝光，不证明真实标注节省。仅单seed/一场已反复使用开发val，无统计显著性、跨比赛或普适结论；不自动扩seed/比例/低标注/SegPose/test。旧S3000诊断弃用且不入此主表/选择，长预算负结果仍按历史保留。

RR阶段端点：first1500末EMA AP55.6383040938%，final0由raw重建EMA AP55.6308457110%，最终AP52.3931424815%。后续R训练/重启调度未保持该阶段高点，符合优化路径或重复训练不利的诊断，但未分离因果；first终点是同run阶段观察，不是独立R1500复现实验，严格保留预登记末步主结果，不事后用55.6383%替换最终52.3931%或据此自动追训。

10完整cp位置/phase计数/既有收据bytes及4端点750图summary SHA通过；末262144B有界逐步尾部global6563/R105008S0一致，terminal.summary update5063为局部终阶段计数，不是漏1500更新。G小control50文件792356B为SEALED、scope control_snapshot_only，逐小SHA核对，manifestSHA26971d5fc8884b1c063bf0a467c8ed0bd9094351568d6ccf151c60ac84e5484c。准确controller28228/CT1793701330.0891285、wrapper16376/CT1793701330.0496707与原worker身份退出，无未知训练writer，GPU锁退休；WDDM桌面程序保留。root不全哈PT/NPZ/媒体、不搬整组、不改原模型/预测/GT/已SEALED控制。G本组completion_verification.json保留终验。

缓存与混合、新短SR、新短RR及所有小控制/owned均闭合，本有限训练链停止。删除bbox-sr-mix定时跟进，保留每日g、全部缓存/模型/失败/日志和原始预测；不自动开后续作业。EXPERIMENTS SHA8f23e187304d067a36cbe288f28438a431bb2de0ed393ecdb2d03b75c121284d/PAPER d48cb4160a513773cde0e910fe1c45c7eb5a63bf058871eafd38e9ad27b71c17保持，方案正文不写入此结果或对话。


## BBOX-BUDGET1500-PLAN-001

UTC 2026-11-03T12:03:01.395290+00:00。用户直接批准补证据建议入方案并启动前两项：独立R1500与混合1500，不是新seed/续prefix。旧方案8f23e187304d067a36cbe288f28438a431bb2de0ed393ecdb2d03b75c121284d→9d8d1afbaffdf591804725e87c33eca45b155239b7c3c98a7aade7c26f50a8cf，PAPER不变；按主题更新1.1/4.5/4.6/5/10。各max90min，新机械各全进程60秒/两项120秒，旧实际625.003047539保留，累计745.003047539。后续29/43/独立比赛/低标注只设计；旧四臂/GT/源码/失败原bytes保持。当前正式0更新、CPU准备中，不读test。


## BUDGET1500：budget1500_R_seed17 STARTED

UTC 2026-11-03T12:09:45.148901+00:00。独立公开YOLO11s/seed17，策略R，0/1500成功更新；实际曝光{'R': 0, 'S': 0}。SoccerNet train2100/val750/S24149，960FP32/micro8×2/workers2，单SGD/EMA，100步warmup .001完整cosine至.00001；0/1000/1500三份完整cp、0/末两次真实750验证，全部逐步/原生预测/失败保留，不best/test。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed17_20261103_a01/budget1500_R_seed17。


## BUDGET1500-R-ADMIT-001

UTC 2026-11-03T12:12:28.329452+00:00。R1500机械实际PASS33.547秒/3诊断更新，完整odd1磁盘恢复后第2步raw/EMA/SGD/全部RNG/sampler/loader/loss/实际梯度逐值exact；代表R/S cached/direct全字段一致。2诊断pt228580062B先展示再TemporaryDirectory精确清理。observer原生句柄计helper完整36.98206090927124秒<=60、returncode0、admissionPASS；原ATTENTION只因退出瞬间Windows conhost35696/CT1793707572.5461483仍在，根后续已核含其全部7owned准确身份退出/锁退，原receipt不覆盖，conhost精确退出时间未知明示，不将迟到观察界当GPU耗时。补核root_closure_verification保持原receiptSHA与事实，不重做GPU。新旧准备累计661.9851084482713秒，混合余一次60秒，全新块120/总745.003047539不重置。

当前独立短预算R1500已于2026-11-03北京时间20:09:30隐藏唯一派发，D/G main_bbox_yolo11s_soccernet_R_seed17_20261103_a01；controller14292/CT1793707770.7727044、wrapper11104/CT1793707770.7541747，命令本组launch.py。初始RUNNING final0是PT0真实验证，非已提交训练步。root13CPU PASS，实际GPU恢复PASS33.547秒/完整helper36.982秒；原observer仅conhost退出瞬间ATTENTION，现全7owned退出/锁退有独立补核原receipt保持。完整新增22.648GiB+G100GiB/D32MiB+100GiB和3723秒<=5400资源通过，冻结41源。R完整终验后唯一R+S1500独立准入接续，不因分数取消。30分钟有限跟进bbox-r已创建，Nov7原截止，正常静默；两项闭合删除，g保留。方案SHA9d8d1afbaffdf591804725e87c33eca45b155239b7c3c98a7aade7c26f50a8cf/PAPER原SHA保持。

资源storage_revision详列3cp/1500步2MiB/2val+1partial/瞬态；实测cp114290031B、保守max114293679B乘1.1，新增22.648031550459564GiB；实测t=[1.906,.531,1.563]秒，稳定第二步.531按2倍及1200val/600cold清理/30cp/300margin=3723.000000008秒<=5400。launchSHA77227b895d0c0273fcb9fa7b9d53f34ae0952f4b7415b5f692ca4b4a9e2a5436，job_table85c55bc087f4fb5493128433a2f67509b2e64a614e9854d54b1603eb30164aee，source41 code_bindingsc5d2ad1b919334bcb8789ac981e7627ebaafef705c7d27538e4fdaf68a039d65，admissionf0b219e4e2c531ce76e34f41cdc66ba2e0dbe080c3d5e1433f2e34e858a4a107，storagee975b535e59a552e23b4f2463faa2f2c0386bd5abdc318cb35b041b17c135735。初始GPU锁/组RUNNING已实际核，首步待PT0完成，不把0当成功训练。process_lifetime_isolation_verified=false，不承诺Codex重启隔离。


## SEG-READINESS-005：标注血缘、来源分组和原生loss CPU核验

UTC 2026-11-03T12:37:00.669070+00:00。用户仅准训练期间继续Seg CPU准备/核验，未启Seg训练或GPU预检；活动BBox的冻结41源/配置/环境/任务及其接续保持。EXPERIMENTS仍9d8d1afbaffdf591804725e87c33eca45b155239b7c3c98a7aade7c26f50a8cf/PAPER仍d48cb4160a513773cde0e910fe1c45c7eb5a63bf058871eafd38e9ad27b71c17，未改方案。

**原GT与处理链**：四份原COCO SHA与RLE候选完整JSON/ledger元数据血缘核对正确，8精确重复(7ihelon+1Quantigo)原除id字段确实完全相等，保留最小id、全部aliases；11有界原polygon/RLE mask逐像素相同。原GT/候选原文件未改。strict会隔离ihelon479/512仅剩33，不能自动用其充完整面板；RLE候选保留1359图/14003人物，只规范已有COCO光栅，不人工修自交、不buffer0/凸包/桥接、不补GT。QuantigoB3两人原area/raster差>10%，ann4482为303.5/613；差按raster分母50.49%，按原area分母101.98%，分母明示。ihelon面积全同raster，bbox浮点与raster整数边界差不是自动错误证据。主AP辅助size分组沿原area的现行候选保持，不静默重算；未来评价策略须冻结。

root已实际看两局部及扩大上下文：ihelon0.jpg ann3虽有proper-crossing，其光栅在该例覆盖弯腰人物；QuantigoB3 frame268/ann4482仅腿部极窄mask，扩大原图中同一近场可见人体躯干/头部未覆盖、邻近mask对应他人，标注完整性ATTENTION，不能只RLE转换就质量PASS。COCO0..424文件名/imageID差与425帧MP4支持0基268对应，但没有原Supervisely frame/object引用；不据序号猜合并。公开包12文件仅每Batch COCO/视频/key_id_map/meta；objects/figures分别64/679、52/1660、150/7128，figure数等ann数但ID无直接对应关系，缺objectKey不自动证明整个COCO无效，疑难人物实例/碎片仍需上下文或原标注链裁定。该原GT不改，不自动补轮廓/删一人留背景；如最终排除，另冻结整图共同排除或合法ignore。

**CPU可追溯读取器**：新增segmentation_candidate_reader.py SHA f2df0b7509981f602fff04d4b84c358e494fa7dc2f9904d487156adc058a4ece，CPU_review显式绑定bundle/候选manifest/原GT/derived/ledger五级SHA并拒绝越根/reparse/篡改，逐实例分离source_annotation与derived_annotation、完整别名与原ID/原sourceSHA。固定unverified组标签/independenceFalse/trainingFalse，不作训练Dataset。root独立8CPU PASS/0skip0.093秒；实际ihelon209.jpg图120读回14人物/7对别名。原segmentation_data与views SHA未变。

**来源独立性**：只读元数据及3个Quantigo首帧/既有ihelon图。root已实际看Batch2 JUV1-1RM35:02、Batch3 JUV1-3RM72:15及同水印，保守同场候选，不把3视频当3场；Batch1未知，未证前可三段归同一unresolved场组，禁止视频级随机跨train/val。与2017欧冠决赛相符仅视觉推断，UEFA官方报告 https://www.uefa.com/newsfiles/UCL/2017/2019641_FR.pdf 只能旁证，不能证明上游文件URL。ihelon0.jpg场边NYRedBulls/MinnesotaUnited标识与上述赛事不同线索，但只取得16/512，余496图未核。旧16对847的0精确RGB重合/minpHash14不证明完整独立。原sourceURL空/None、Q2023日期为导出而非比赛时刻。现还不能声称跨源独立确认；平台许可声明与上游来源链分列，未联系作者、未下载新媒体。Q声明844/实际847已登记不改配额。

**原生loss契约**：新增test_segmentation_native_cpu.py SHA aaca9fcc85e2d53dd845a3ac2d8c430f56e43692c6f113a7be94600c4dd4fcdb，随机n-seg96CPU1线程，仅forward/loss/backward、optimizer.step=0/真实样本=0/公开权重=0/CUDA=0，root独立5PASS/0skip0.488秒，结束RNG/线程/模式恢复。正/空micro1/重叠/孔洞/断开测试可入criterion，4项loss box/seg/cls/dfl；seg gain沿box，native返回乘micro图数，有效16仍待正式累积适配。实际96→proto24，候选mask96→48后criterion再nearest到24，mask_ratio2不等于半分辨率loss。3x3夹具在ratio2留1像素而proto4消失，须原生尺度审计。root对四doc各8个最小raster实例共32做960→480→240身份仿射有界探针，9个已被现有训练过滤，余未发现额外proto空mask；这不是全库/随机增强发生率，也不代表公开s960已GPU核验，不据夹具静默删真实GT。

**最小科学训练尚缺**：①补齐496媒体并核帧/GT对应与比赛级来源组，冻结train/dev UID和保守独立性范围；②确定可见场上参与者(球员/裁判)/未标观众监督语义，裁定4482等疑难完整性、统一GT候选与原area/bbox评价声明，不按模型成绩挑清洗；③完成原生Seg有效16累积/完整rawEMA优化器全部RNGsamplerloader快照与disk下步恢复；④实现真实预测mask裁剪/逆letterbox至原图RLE、评分/去重/标准MaskAP与全部预测记录；⑤冻结最小成对训练的绝对更新/R/S曝光/调度/终点与有限资源，BBox释放后且用户明确允许Seg GPU时才能真实960显存/吞吐/checkpoint/输出容量准入。现R500/RR500+500/SR500+500只是旧候选、非同总预算，不自动套用或启动；当前机械CPU通过不等于可解释的Seg迁移实验准入。

正式小收据(原GT/模型不在此重复hash)：

- G:\FutsalMOT_Dataset\real\football_seg_public\derived\GT_READINESS_AUDIT_20261103.json · SHA256 16cbd68f5b95351c18655d52a8556602fe40b48ae8925a96fa3cad1be7dd509a

- G:\FutsalMOT_Dataset\real\football_seg_public\derived\CANDIDATE_READER_CPU_RECEIPT_20261103.json · SHA256 6a9200be302561bb9aed65b59873e391c4abc22762b3d4132e19e29a2760d2d1

- G:\FutsalMOT_Dataset\real\football_seg_public\SOURCE_INDEPENDENCE_RECHECK.json · SHA256 0250419c6d34ae1bec8dc2417f91a57f2251c04d9a70bfac3bbf71576edd17e6

- G:\FutsalMOT_Dataset\real\football_seg_public\CPU_NATIVE_LOSS_CONTRACT.json · SHA256 7cefd79fea068f46245bba9f29732df2785b7ef60532ae5c5d3f0571a32d9a6d

- G:\FutsalMOT_Dataset\real\football_seg_public\CPU_NATIVE_RESOLUTION_PROBE.json · SHA256 0d37e167b86c3e4321c9fb14db84e3bfa7ff4fa14ff9e9c59dc172b857f32d0f

- G:\FutsalMOT_Dataset\real\football_seg_public\derived\GT_VISUAL_REVIEW_20261103.json · SHA256 4b4c296d25a9415c914da77c9e37d52297513d5985416d50efc9b1757f7ac1f9


## BUDGET1500：budget1500_R_seed17 COMPLETE

UTC 2026-11-03T12:37:18.652991+00:00。独立公开YOLO11s/seed17，策略R，1500/1500成功更新；实际曝光{'R': 24000, 'S': 0}。SoccerNet train2100/val750/S24149，960FP32/micro8×2/workers2，单SGD/EMA，100步warmup .001完整cosine至.00001；0/1000/1500三份完整cp、0/末两次真实750验证，全部逐步/原生预测/失败保留，不best/test。结果：{"state": "COMPLETE", "pid": 14292, "create_time": 1793707770.7727044, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 24000, "S": 0}, "schema_version": "bbox_budget1500_v1", "job_id": "budget1500_R_seed17", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_R_seed17_20261103_a01\\budget1500_R_seed17", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5516342616501645, "AP50": 0.958764562216891, "AP75": 0.5687237972749589, "AP_small": 0.3508693325303684, "AP_medium": 0.5414928260561843, "AP_large": 0.6149733688430563, "AR_max1": 0.05544046661303299, "AR_max10": 0.46280168946098155, "AR_max100": 0.6451327433628318, "AR_small": 0.42047244094488195, "AR_medium": 0.6485135135135136, "AR_large": 0.6645161290322581}, "raw_prediction_bytes": 4273717704, "images": 750, "wall_seconds": 378.71800000000076, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5516342616501645, "AP50": 0.958764562216891, "AP75": 0.5687237972749589, "AP_small": 0.3508693325303684, "AP_medium": 0.5414928260561843, "AP_large": 0.6149733688430563, "AR_max1": 0.05544046661303299, "AR_max10": 0.46280168946098155, "AR_max100": 0.6451327433628318, "AR_small": 0.42047244094488195, "AR_medium": 0.6485135135135136, "AR_large": 0.6645161290322581}, "raw_prediction_bytes": 4273717704, "images": 750, "wall_seconds": 378.71800000000076, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed17_20261103_a01/budget1500_R_seed17。

三张非主线临时视觉审阅图（来源首帧联系图、两GT局部叠加、4482扩大上下文）均由root实际查看后按精确路径清理；小收据保留其SHA/源图和视频帧/crop坐标及观察，原RGB/视频/GT一律未删除。


## BUDGET1500-R-RESULT-001

UTC 2026-11-03T12:41:29.494374+00:00。独立R1500于2026-11-03T12:37:18Z（北京20:37:18）COMPLETE1500/24000R/0S，末步EMA AP50:95=55.1634261650%、AP50=95.8764562217%、AP75=56.8723797275%。相对既有R6563末51.6511537498%，短预算高3.5122724152pp；相对旧mixed6563末53.9318398807%，短R高1.2315862843pp。说明较强的短程R对照可改变此前混合收益的解释，不能继续声称旧mixed6563优于充分开发R；新mixed1500仍需完成同预算比较，不能依据R结果取消或增第三个预算。单seed/已开发750val，仅开发证据，非显著/泛化或标注节约。原RR首1500端点只是同run观察，未拿来替代此独立起点，采样流不同不要求位元重现。

3计划完整cp0/1000/1500小收据/bytes、2精确tag/750summary、末262144B有界tail1500/R24000S0、50文件小control SEALED逐SHA、controller14292/CT1793707770.7727044与wrapper11104/CT1793707770.7541747退出/无worker或未知writer/GPU锁退均核。G completion_verification.json SHAffb8325d5b0a90a5f1f554f9732ff0402e396a9096451b5c610495fc683129c2；statusSHA6c3e2649e8265f3964fcc2ed78cdeeb40ce2a62de9236b51709998ef75cb337f，controlSHA d29b825bdbd0b8036cffe585143de0ed2e1518bb134f47aacfa592dfc454c5d2。模型/NPZ/媒体未全哈或搬迁，所有负结果保持。

按既定顺序已创建同日RSa02新1500协议候选，旧RSa01完整6563结果保留；a02为明确命名冲突消解，不是失败重试。自动跟进在root转换期间暂PAUSED防重复，R+S实际准入/冻结/启动后恢复30分钟。Seg只CPU核验，未授权训练/GPU预检。


## BUDGET1500：budget1500_R_plus_S_seed17 STARTED

UTC 2026-11-03T12:42:53.072366+00:00。独立公开YOLO11s/seed17，策略R+S，0/1500成功更新；实际曝光{'R': 0, 'S': 0}。SoccerNet train2100/val750/S24149，960FP32/micro8×2/workers2，单SGD/EMA，100步warmup .001完整cosine至.00001；0/1000/1500三份完整cp、0/末两次真实750验证，全部逐步/原生预测/失败保留，不best/test。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed17_20261103_a02/budget1500_R_plus_S_seed17。


## BUDGET1500-RS-ADMIT-001

UTC 2026-11-03T12:44:16.315474+00:00。用户请求已完成S→R后继续，root核旧SR/RR均闭合、当前有限队列实际刚完成独立R1500，按已批准顺序只接续R+S1500，不重跑旧两阶段。

当前独立R1500已完整验收，末AP55.1634%、1500/24000R、3cp/2val/50小control SEALED及owned/GPU退出。下一唯一R+S1500已于2026-11-03北京时间20:42:39隐藏派发：D/G main_bbox_yolo11s_soccernet_RS_seed17_20261103_a02，job budget1500_R_plus_S_seed17，wrapper35844/CT1793709759.624544，controller34348/CT1793709759.643014，完整D本组launch.py；以dispatch/真实CT+cmd/GPU锁为准。初始final0为PT0真实验证，非已有成功更新。实际独立机械PASS36.609秒/全helper40.335591316秒，完整资源22.6480GiB+G100GiB/D32MiB+100GiB/3771秒<=5400通过，冻结41源。每步8R8S/1500步总12000R12000S、warm100完整.001cosine、3cp/2val，末EMA不best/test。30分钟bbox-r已恢复ACTIVE，本混合完整闭合后删除，不追加其他训练；Seg仅CPU准备，GT/独立性/Trainer尚未准入，不训练。EXPERIMENTS SHA9d8d1afbaffdf591804725e87c33eca45b155239b7c3c98a7aade7c26f50a8cf/PAPER原SHA保持。

混合GPU机械实际PASS36.609秒/3诊断更新，odd1完整disk→2→fresh恢复replay2，全部raw/EMA/SGD/CPU CUDA RNG/sampler/loader状态及loss/实际梯度逐值exact，代表R/S cached/direct一致。临时2pt228580062B已展示精确清。原生句柄计helper40.335591316223145秒<=60/returncode0/admissionPASS；原ATTENTION只conhost退出瞬间短暂残留，root_closure_verification已核全部owned退出/锁退，原receipt保留且host精确退出时刻未知，未重做GPU。新块两项完整helper77.31765222549438秒<=120；历史625.003047539+新=702.3206997644944<=745.003047539，预算闭合，不再GPUtoy或预检。

实测cp114290031B，保守max114293807*1.1；3cp/1500步2MiB/2*750val+一次partial(native6650344B+JSON1MiB/图)/3GiB+2cp，完整新增22.64803220704198GiB另G100GiB/D32MiB+100GiB。稳态诊断.547s，以1500*.547*2+1200val+600cold/cleanup+30cp+300margin=3771<=5400，runtime绝对截止2026-11-03T14:12:49.373152+00:00，不重置，不延Nov7。

launchSHA77227b895d0c0273fcb9fa7b9d53f34ae0952f4b7415b5f692ca4b4a9e2a5436；job_tablea7c61afe078d881d2bf919e7b7339bb797b3dbb6ebb43bb8102ecf0507e5fe3c；code_bindingsc5d2ad1b919334bcb8789ac981e7627ebaafef705c7d27538e4fdaf68a039d65；admissionf023e2589bc05824c1699a428a9275a808976b0450140cface7174042a7a0cae；storagee1be7d58f162d53dc068ce830ee3552dfd16126c6674b632830e4cb305f7b3b1。actualcontroller/CT+cmd/锁及小状态已核RUNNING，stderr0；初始update0是阶段0，不能冒称训练成功步。process_lifetime_isolation_verified=false，不承诺Codex重启完全隔离。全部旧模型/预测/GT/失败及原RSa01保持。


## BUDGET1500：budget1500_R_plus_S_seed17 COMPLETE

UTC 2026-11-03T13:12:31.106403+00:00。独立公开YOLO11s/seed17，策略R+S，1500/1500成功更新；实际曝光{'R': 12000, 'S': 12000}。SoccerNet train2100/val750/S24149，960FP32/micro8×2/workers2，单SGD/EMA，100步warmup .001完整cosine至.00001；0/1000/1500三份完整cp、0/末两次真实750验证，全部逐步/原生预测/失败保留，不best/test。结果：{"state": "COMPLETE", "pid": 34348, "create_time": 1793709759.643014, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 12000, "S": 12000}, "schema_version": "bbox_budget1500_v1", "job_id": "budget1500_R_plus_S_seed17", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_RS_seed17_20261103_a02\\budget1500_R_plus_S_seed17", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5404465730435029, "AP50": 0.9585818177620226, "AP75": 0.5413643962243975, "AP_small": 0.3597382159195037, "AP_medium": 0.5302998201044834, "AP_large": 0.6096003676731887, "AR_max1": 0.054635961383749, "AR_max10": 0.4571500402252614, "AR_max100": 0.635528962188254, "AR_small": 0.415748031496063, "AR_medium": 0.6381326781326782, "AR_large": 0.6585806451612902}, "raw_prediction_bytes": 4262352847, "images": 750, "wall_seconds": 359.01600000000326, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5404465730435029, "AP50": 0.9585818177620226, "AP75": 0.5413643962243975, "AP_small": 0.3597382159195037, "AP_medium": 0.5302998201044834, "AP_large": 0.6096003676731887, "AR_max1": 0.054635961383749, "AR_max10": 0.4571500402252614, "AR_max100": 0.635528962188254, "AR_small": 0.415748031496063, "AR_medium": 0.6381326781326782, "AR_large": 0.6585806451612902}, "raw_prediction_bytes": 4262352847, "images": 750, "wall_seconds": 359.01600000000326, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed17_20261103_a02/budget1500_R_plus_S_seed17。


## BUDGET1500-RS-RESULT-001 / BBOX-TWO-BUDGET-CLOSE-001

UTC 2026-11-03T13:17:25.595117+00:00。mixed1500于2026-11-03T13:12:31Z（北京21:12:31）COMPLETE1500/R12000S12000，末步EMA AP50:95=54.0446573044%、AP50=95.8581817762%、AP75=54.1364396224%；同R1500的55.1634261650%/95.8764562217%/56.8723797275%差−1.1187688607/−0.0182744455/−2.7359401051pp。小/中/大绝对AP mixed35.9738/53.0300/60.9600%，R35.0869/54.1493/61.4973%，差+.8869/−1.1193/−.5373pp；小目标只是此seed分层开发信号，不从单正分层定普适贡献。

| 方法 | 成功更新 | R/S曝光 | 末AP50:95 | 末AP50 | 末AP75 |
|---|---:|---|---:|---:|---:|
| 独立R1500 |1500|24000/0|55.1634%|95.8765%|56.8724%|
| mixed1500 |1500|12000/12000|54.0447%|95.8582%|54.1364%|
| 原R6563 |6563|105008/0|51.6512%|94.1100%|50.5687%|
| 原mixed6563 |6563|52504/52504|53.9318%|95.7025%|54.6063%|

同1500混合−R −1.1188pp，同6563 +2.2807pp；在预先有限两候选预算内两方法的开发末步最优均为1500，选后仍是R高1.1188pp，不从额外checkpoint/弃用S3000挑best。纯R从1500到6563下降3.5123pp，mixed仅下降.1128pp，可描述本配置下两预算敏感性不同；与合成正则化/缓和重复训练不利相符只是机制假设，不是过拟合或正则化因果证明。新增结果削弱此前“混合提升充分开发R最高精度”的论点；目前支持的是收益取决于预算/训练配方，而非稳定普适的最高AP提升。AP50几乎相同但AP75明显较低，差主要出现在严格定位指标；不能因AP50接近就事后判等效。

两臂都允许全2100真实唯一图池，R曝光减半不是节约标注；没有预定非劣效界限，不称同等性能/通过。单seed、一场已反复用750dev，无独立比赛确认/统计显著或普适结论；不据负结果再加LR/比例/seed搜索。原短SR/RR和长预算负结果全部原样保留，当前有限链不追加训练。

验收：3完整cp0/1000/1500小收据及bytes、2精确tag/750summary/末EMA、有界末step1500/R12000S12000、G control50文件794039B SEALED逐小SHA、controller34348/CT1793709759.643014和wrapper35844/CT1793709759.624544真实退出、无已知worker/未知writer/GPU锁均核。statusSHA02a91225d7f4830c1aa4515dc6d7b7af6ecd41e11b3ec974b8bf244b8673d701，controlSHA5a552a21262131842e37582ba082196a608b20b12bff1f7494768748cf7c7cec；G completion_verification.json与budget_comparison.json小收据保存绝对值/来源SHA和差，不全哈模型/NPZ/媒体，不搬整run。driver已有COMPLETE不重复，有限bbox-r原生删除、每日g不动。观察台“未登记”来自tools/training_monitor.py尚未把bbox_budget1500_v1列为显示分类，实际run/protocol/41源/参数/source绑定全部已冻结登记，非训练缺记录。


## LOWLABEL25-PLAN-001

UTC 2026-11-03T13:34:36.455929+00:00。用户明确要求观察台标训练预算，并认可继续BBox标注节约实验训练。约1.12pp单seed差不定义为稳定负作用；AP为AP50:95非单IoU.95；无预定等效/非劣效门限，不因55%或1pp大小自动判等效。新增仅N25=525的R与1:1混合各1500步，公开独立seed17，完整参数/流程按主题更新第1/5/10节，不往方案堆结果对话。方案原9d8d1afbaffdf591804725e87c33eca45b155239b7c3c98a7aade7c26f50a8cf→c3450093007a43c16ae747ebc8a294ba7e1ffda202cd71d8329a7a57cfafc09f，PAPER原SHA保持，已完成run旧绑定合法不回写。子集配置原train2100中525/完整105个5检测帧块，3match/14clip全覆盖，match块配额30/37/38；相同525用于两臂，50%嵌套仅设计。配置SHA bdac16a1dc34b8ebdd9cc8f9a5c6cc98b1844bacbec4af90b25ffaebe7b95bf0；有序selected525 SHA52967e8fda4585b3d20d7f520f5a76cfd67c00c72c09d8ec3bb92e2b6cebb46b。选择后审计8064球员门将框/1363ignore非目标，不依据标签挑图。未选1575不训练/伪标签；完整GT来源binding保留。新块机械每臂全helper60秒/coop50、两臂新增120，旧实际702.3206997644944保持，新累计上限822.3206997644944，旧剩余额度不转作GPUtoy。各正式max5400秒/原Nov7总界限不延。当前仅CPU实现准备，未GPU/正式低标注更新。


## SEG-FULL512-BACKEND-001 / MONITOR-BUDGET-001

UTC 2026-11-03T13:44:31.312131+00:00。观察台已实际重载v7，旧服务身份精确核验后仅退出只读服务，训练不受影响；HTTP API返回预算显示，正式注册两短配对、固定混合、1500与N25新schema，旧未登记属于显示分类缺项。选择菜单/比较菜单/图例/详情均显示总步、阶段或继承/新增、R/S曝光、真实唯一图数；既有run清单不改。新服务PID29860/CT1793712975.763151，127.0.0.1:8765，截止2026-11-04T00:23:06.528282Z，未延长原服务。monitor dd03c95ded99120274c1865f79a7760d05a51b2acfc260a6c7f6881c118b0e21；6预算CPU与5命名CPU均PASS/0skip，HTTP实际11菜单均有预算且无未登记BBox；尚未声称浏览器视觉验收。

Seg ihelon v1已补齐512/512图，327587594B；新496共317189922B，其中A01新增161后失败原证据保留，独立有界A02仅续缺335，最终13:26:31Z完成；三次HTTP200短读均按原URL/严格完整字节重取，不放宽门限，原16/161图及原GT/sourcecard不变。CPU指纹复用下载必要解码，Q847仅一次低分辨率取帧。512x847=433664对：lowres RGB exact0、pHash<=8候选0、最小Hamming12；不等于已证明比赛独立。发现ihelon4.jpg(image321)/517.jpg(image451)完整decodedRGB相同、各16人物GT版本不同，配对MaskIoU均值.910170；512文件仅511唯一RGB，必须同组同split，原图/两GT均留存，不自动删图/挑版本。

1359精确UID候选仅分保守ihelon512与Quantigo847两组，Q三个Batch同组，全部activated_split=null/independence_verified=false/training_admitted=false；44幅ihelon和9幅Q场景接触表已实际查看，服饰/广告/转播视觉支持跨来源不同场景，官方日期/源视频URL尚缺，不能把平台名或三Batch当独立比赛。844声明与847实帧差可能为0基last-index相加，属于推断，不删首帧凑数。Q4482薄腿遗漏、Q5895两像素碎片、ihelon5904三像素碎片/960消失和4261后排遮挡边界均在疑难ledger追溯；自交不直接等于坏GT，不作凸包或全删479图。原COCO栅格RLE是表示转换而非人工修复，原GT/area/bbox全保留。新5份小收据SHA实际核验，路径见下。

Seg CPU后端segmentation_backend.py a3e8ba5cc4ce426e9f7a368031a477d2029c76cfaa4fefbea35a09c0d50c8fe9根独立11项PASS/0skip12.145秒：随机n-seg96夹具，micro1x16原生四损失归一化、SGD/EMA、R与S各自完整磁盘恢复的下一步loss/梯度/raw/EMA/SGD/RNG/采样/loader generator逐值exact，失败恢复保护调用者状态。两临时cp各35137681B展示后精确清；仅CPU夹具optstep，不冒称公开s/960真实训练。公开seg基座factory/manifest校验已接，尚未执行public-s前向。single_domain_per_update_v1明确禁止混域单更新；旧通用检查点的16图提交边界不能直接容纳8R8S，未热改全局记录器放宽。

距离真实最小Seg训练仍缺：比赛原始来源链与正式保守split准入、疑难标注最终处置/派生版本冻结、生产mask逆几何与完整原生预测/MaskAP接线、公开s/960实际数值恢复及GPU资源/吞吐准入、完整500步生命周期/有界执行入口。当前全部Seg不训练、不GPU预检；既有BBox数值冻结集合不加入Seg模块。

- G:/FutsalMOT_Dataset/real/football_seg_public/ihelon_v1/DOWNLOAD_COMPLETION_20261103_A02.json SHA 38d46d186ec008a6b25974589fb92f891137993661551191069fb2df3c215525
- G:/FutsalMOT_Dataset/real/football_seg_public/ihelon_v1/CPU_FINGERPRINTS_20261103_A02.json SHA 8d21a00cb8f278712ba3cb61f65ee58393bc3709307bad9db6555fcbd39211fa
- G:/FutsalMOT_Dataset/real/football_seg_public/derived/CROSS_SOURCE_FULL512_AUDIT_20261103.json SHA de5c159909fb5c091c5afbe0368a72e40146a97faa6c23fca4a923540e4f4681
- G:/FutsalMOT_Dataset/real/football_seg_public/derived/GROUP_SPLIT_CANDIDATE_20261103.json SHA ebee7c78508ff726e2f06cb891022323b91859e00119cbc61ba7c991a3473ea0
- G:/FutsalMOT_Dataset/real/football_seg_public/derived/DERIVED_QUALITY_REVIEW_LEDGER_20261103.json SHA a7d22e18a9c81c63f15a54c91a188de44173ed3a862ba07902e02e22f963bcd9


## LOW-LABEL25：lowlabel25_R_seed17 STARTED

UTC 2026-11-03T13:47:55.759774+00:00。公开YOLO11s独立起点seed17，唯一真实525/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset25_seed17_v1.json", "sha256": "bdac16a1dc34b8ebdd9cc8f9a5c6cc98b1844bacbec4af90b25ffaebe7b95bf0"}。策略R，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1575不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed17_20261103_a02/lowlabel25_R_seed17。


## LOWLABEL25-R-ADMIT-001

UTC 2026-11-03T13:51:56.759812+00:00。N25纯R于2026-11-03T13:47:42.638153Z（北京21:47:42）隐藏唯一派发，D/G main_bbox_yolo11s_soccernet_R_seed17_20261103_a02，job lowlabel25_R_seed17。a01为已完成全2100 R1500，本a02是N25新协议命名冲突消解，不是失败重训。wrapper9108/CT1793713662.639969、controller34940/CT1793713662.657884，完整D本组launch.py；真实CT/cmd/GPU锁与组RUNNING已核。当前leaf0是PT0真实验证，不冒称实际新增更新。runtime从13:47:52.221524Z起5400秒，固定截止15:17:52.221524Z/北京23:17:52，不因恢复重置；不承诺Codex重启隔离。

root实际预检PASS48.937秒/3诊断更新，完整disk odd1->2->fresh恢复2，全状态/loss/实际梯度逐值exact，实际选中catalog525/未选1575进入0，代表R/S cached/direct样本全部字段exact。两临时pt共228562142B已显示且精确清；cp114281071B，第二步.64秒为短窗。观察器持原生句柄GetProcessTimes全helper52.39515829086304秒，VERIFIED_EXIT/returncode0/所有owned退出/锁退休；conhost另等.047秒并明确无exact宿主ExitTime，不混为helper耗时。历史准备702.3206997644944保持，新累计754.7158580553574/822.3206997644944，新块52.39515829086304/120，混合另最多60全helper。

准备首次在任何正式launch之前因同一缓存路径正/反斜杠字符串拒绝；根按系统调试定位5字段中只有input_cache路径表示不同。原准入与42训练源码全不改，新增独立CPU tools/low_label_prepare_compat.py和G预检/prepare_compat.py（SHA459db0105a9e0d93ef1c43af1ebe38c5213c24c61370486caf8f765e43dae055）仅把原prepare的两处cache比较交原formal.cache_specification规范，实际路径固定v2、manifest精确fce4e026...，改SHA或路径仍拒绝；4CPU路径/错误SHA/错误路径/AST逆变换检查通过。不放宽数值或其他来源，不重做GPU、不篡改admissionPASS，也不把额外准备代码说成没执行。原prepare首次失败现场未伪造run失败，原storage bytes保留。后续混合request事先规范cache字段，避免重犯。

完整新增24295280650B=22.62674332596GiB：3cp按保守114293807B*1.1、1500步2MiB、2*750原native6650344+JSON1MiB、一次750partial、瞬态3GiB+2cp，G另100GiB/D小控制32MiB另100GiB。actualfree G762892058624/D121582989312B，保守完整time4050<=5400，已冻结storage_revision；不删预测/失败或减少更新绕Gate。

原42训练源为已冻R1500的41+low_label_bbox2923b3fa...，subset manifest bdac16a1.../有序525 SHA52967e8f...，原GT只读/未选1575禁用，val750不变；公开独立YOLO11s、seed17、960FP32/micro8x2/workers2、warm100/.001完整cosine到.00001、SGD/原loss增强、1500/24000R、3完整cp/2端点val/全部记录与失败保留，末EMA不best/test。root13CPU PASS/0skip5.613秒，source未变不重复测。

launchSHA77900eb807a67d60976e7e553a640976f389e97c56244223f50e4a6f94e473ba；job_table d459ddb658da22627d16979e0c9cdae7b01307b9456fbbf45bcaf5416dfb7580；codebindings551971ea68ae429864c2975ef1c3bea29353c5dc181f731034b5aacc220840e6；admission453b68beec6dae1eab878d139218c51edc9d657bcb4c49f3d8913f67b8ab467f；storage d76096ba46f38b9281616d16736ff17e1cb2781f0e0b253d2a0372ea0cfd700c。G startup_verification.json保留准确身份和实际HTTP新菜单“正式｜BBox｜YOLO11s｜SoccerNet｜R｜seed17｜总1,500步 · R24000/S0 · 真实图525/2100”。原生bbox-n25已ACTIVE每30分钟静默接续：纯R完整cp/val/末曝光/SEALED逐小SHA/owned全退/锁退后，实际独立准入唯一同525 mixed1500，不以分数为条件；全部闭合后比较N25两组与全2100 R1500性能gap并删除自身，g保留。无预定非劣效界限不自设标注节约PASS，Seg仍禁止训练/GPU。


## LOW-LABEL25：lowlabel25_R_seed17 COMPLETE

UTC 2026-11-03T14:13:38.257706+00:00。公开YOLO11s独立起点seed17，唯一真实525/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset25_seed17_v1.json", "sha256": "bdac16a1dc34b8ebdd9cc8f9a5c6cc98b1844bacbec4af90b25ffaebe7b95bf0"}。策略R，1500/1500成功更新，实际曝光{'R': 24000, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1575不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 34940, "create_time": 1793713662.657884, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 24000, "S": 0}, "schema_version": "bbox_low_label25_v1", "job_id": "lowlabel25_R_seed17", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_R_seed17_20261103_a02\\lowlabel25_R_seed17", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5178672831102502, "AP50": 0.948368333113909, "AP75": 0.49859485323435504, "AP_small": 0.2975364297228139, "AP_medium": 0.5096618824894804, "AP_large": 0.5806042485410985, "AR_max1": 0.05370072405470636, "AR_max10": 0.44722445695897034, "AR_max100": 0.6098250201126307, "AR_small": 0.35551181102362206, "AR_medium": 0.6137714987714988, "AR_large": 0.6320645161290323}, "raw_prediction_bytes": 4275771978, "images": 750, "wall_seconds": 369.9539999999979, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5178672831102502, "AP50": 0.948368333113909, "AP75": 0.49859485323435504, "AP_small": 0.2975364297228139, "AP_medium": 0.5096618824894804, "AP_large": 0.5806042485410985, "AR_max1": 0.05370072405470636, "AR_max10": 0.44722445695897034, "AR_max100": 0.6098250201126307, "AR_small": 0.35551181102362206, "AR_medium": 0.6137714987714988, "AR_large": 0.6320645161290323}, "raw_prediction_bytes": 4275771978, "images": 750, "wall_seconds": 369.9539999999979, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed17_20261103_a02/lowlabel25_R_seed17。


## LOW-LABEL25：lowlabel25_R_plus_S_seed17 STARTED

UTC 2026-11-03T14:18:34.837551+00:00。公开YOLO11s独立起点seed17，唯一真实525/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset25_seed17_v1.json", "sha256": "bdac16a1dc34b8ebdd9cc8f9a5c6cc98b1844bacbec4af90b25ffaebe7b95bf0"}。策略R+S，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1575不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed17_20261103_a03/lowlabel25_R_plus_S_seed17。


## LOWLABEL25-R-RESULT-001 / LOWLABEL25-RS-ADMIT-001

UTC 2026-11-03T14:21:25.263326+00:00。用户明确“纯R已完成，继续R+S混合”；root验收N25纯R于14:13:38Z（北京22:13:38）COMPLETE1500/R24000S0，末EMA AP50:95=51.7867283110%、AP50=94.8368333114%、AP75=49.8594853234%。同1500全2100 R为55.1634261650%，少3.3766978540pp；这是525唯一图下性能gap，不事后设等效/标注节约PASS。3完整cp0/1000/1500收据与bytes、2精确tag750真实val、bounded末step1500/R_exposures24000/S_exposures0、G control51文件817423B SEALED逐小SHA、准确controller34940和wrapper9108创建身份退出、无已知worker/未知writer/GPU锁退休均实际通过；未重哈pt/NPZ/媒体。纯RstatusSHAe92110314634b1562ee9cc4f9034e212038424ecedc6d1582e3eb45285fd4c55，controlSHAaa7a4d6bc59d05745db3481e2a39f1e01653966109b8311d9f12f1bd461ad5bc，G completion_verification.json。driver已有COMPLETE不重复造结果。

混合N25独立机械PASS37.484秒/3诊断更新，完整8R8S odd-step disk->next->fresh恢复next全状态/loss/实际梯度逐值exact；代表R/S cached/direct全部字段exact，实际catalog525/未选真实进入0。两临时pt共228562142B已展示精确清，cp114281071B，第二步.594秒仅短窗。持句柄GetProcessTimes全helper41.172343015670776秒，VERIFIED_EXIT/return0/owned全退/锁退休；新配对52.39515829086304+41.172343015670776=93.56750130653382<=120，旧702.3206997644944保持，累计795.8882010710282<=822.3206997644944，不借余量再预检。request事前用原cache_specification规范缓存路径，原prepare直接成功，原42训练源码和准入原bytes不变，无需重复R的准备端兼容。

新组D/G main_bbox_yolo11s_soccernet_RS_seed17_20261103_a03，job lowlabel25_R_plus_S_seed17，于2026-11-03T14:18:18.606492Z（北京22:18:18）隐藏唯一启动。a01为全2100混合6563、a02为全2100混合1500，本a03为新N25协议命名冲突消解，不是失败重试。wrapper35388/CT1793715498.608974，controller29744/CT1793715498.6285424，完整D本组launch.py；真实CT/cmd/GPU锁/组RUNNING与观察台预算已核。初始0为PT0真实750验证，不冒称新增提交。runtime起14:18:30.806432Z/max5400，固定截止15:48:30.806432Z（北京23:48:30），不重置、不延Nov7。

同一525/2100冻结子集bdac16a1.../有序525 SHA52967e8f...，未选1575不训练/伪标签，val750不变。公开独立YOLO11s seed17/960FP32/micro8x2有效16/workers2，单阶段1500，每micro交错4R4S，累计12000R+12000S。warm100 LR0到.001/momentum.8到.937后完整cosine到.00001；SGD Nesterov/WD.0005/clip10/原loss增强/角色ignore不改。3完整cp0/1000/1500、2端点真实val、全部逐步与原生预测/JSON/失败保留，末EMA不best/test，source42仍已冻41+low_label_bbox2923...，无Seg/Pose数值接入。

混合完整新增24295280650B=22.62674332596GiB另G100GiB，D控制32MiB另100GiB，按3cp实测保守max114293807*1.1、1500步2MiB、2x750(native6650344+JSON1MiB)、一次750partial、瞬态3GiB+2cp，actualfree G753797943296/D121579864064B，保守完整time3912<=5400，已冻结storage_revision。launchSHA77900eb807a67d60976e7e553a640976f389e97c56244223f50e4a6f94e473ba，job_table eae52def0474490477e562e829771bebc31205ad90ea64bbfc3a33453fb1c7e9，code_bindings551971ea68ae429864c2975ef1c3bea29353c5dc181f731034b5aacc220840e6，admission c55c07d83f9f917500272295cd8676d6b638d5a9122f4c43140e5e29bea88355，storage49ad00122e74eae2d646bce18e64f76ea60d46635f683ad0f40fe4c580d93d78。G startup_verification小收据记录准确身份/菜单/限时，不承诺Codex重启隔离。

原生bbox-n25同一任务已更新为当前唯一混合终验，每30min/原Nov7截止，健康静默；无后续自动训练。完整cp/val/末曝光/SEALED逐小SHA/owned全退/GPU锁退休后，列525两组绝对主AP及差、相对全2100 R1500性能gap，保留全部负结果，无预定非劣效界限不自设PASS，单seed已开发单场val仅有限证据；完成即删除自身，g不动。Seg仍只CPU、不训练不GPU。


## LOW-LABEL25：lowlabel25_R_plus_S_seed17 COMPLETE

UTC 2026-11-03T14:49:08.252854+00:00。公开YOLO11s独立起点seed17，唯一真实525/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset25_seed17_v1.json", "sha256": "bdac16a1dc34b8ebdd9cc8f9a5c6cc98b1844bacbec4af90b25ffaebe7b95bf0"}。策略R+S，1500/1500成功更新，实际曝光{'R': 12000, 'S': 12000}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1575不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 29744, "create_time": 1793715498.6285424, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 12000, "S": 12000}, "schema_version": "bbox_low_label25_v1", "job_id": "lowlabel25_R_plus_S_seed17", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_RS_seed17_20261103_a03\\lowlabel25_R_plus_S_seed17", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5241190226766204, "AP50": 0.9501861137220792, "AP75": 0.5044386421679721, "AP_small": 0.33073175167020624, "AP_medium": 0.512115297420896, "AP_large": 0.5977780741084034, "AR_max1": 0.053167739340305706, "AR_max10": 0.4485317779565567, "AR_max100": 0.6228278358809332, "AR_small": 0.37047244094488185, "AR_medium": 0.6271007371007371, "AR_large": 0.6425161290322581}, "raw_prediction_bytes": 4267161942, "images": 750, "wall_seconds": 362.28199999999924, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5241190226766204, "AP50": 0.9501861137220792, "AP75": 0.5044386421679721, "AP_small": 0.33073175167020624, "AP_medium": 0.512115297420896, "AP_large": 0.5977780741084034, "AR_max1": 0.053167739340305706, "AR_max10": 0.4485317779565567, "AR_max100": 0.6228278358809332, "AR_small": 0.37047244094488185, "AR_medium": 0.6271007371007371, "AR_large": 0.6425161290322581}, "raw_prediction_bytes": 4267161942, "images": 750, "wall_seconds": 362.28199999999924, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed17_20261103_a03/lowlabel25_R_plus_S_seed17。


## LOWLABEL25-RS-RESULT-001 / BBOX-N25-CLOSE-001

UTC 2026-11-03T14:54:03.264134+00:00。混合N25于2026-11-03T14:49:08Z（北京22:49:08）COMPLETE1500/R12000S12000。root实际核3完整cp0/1000/1500小收据/bytes、2精确tag/750真实val summary和末EMA、bounded末step1500/R12000S12000、G control51文件816420B SEALED逐小SHA、controller29744/CT1793715498.6285424和wrapper35388/CT1793715498.608974真实退出、无已知worker/未知writer/GPU锁退休；statusSHA2fa39a58351da16d9be7556c71d7dd7f24e0419c1ab5bf01a32ec7981e44084e，controlSHA790b874f8fb54877c340863598cff964f72460bfe585009f783ec169412c4df8，G completion_verification.json。未重哈PT/NPZ/媒体或搬整组，driver已有COMPLETE不重复。

四组同公开YOLO11s独立seed17、960FP32/micro8x2有效16、workers2、相同1500步/100warm .001完整cosine、3cp/2真实750val/末步EMA，不best/test；真实图数量与累计曝光分列：

| 真实标注池 | 方法 | 更新数 | R/S曝光 | AP50:95 | AP50 | AP75 |
|---|---|---:|---|---:|---:|---:|
| 2100/100% | R |1500|24000/0|55.1634%|95.8765%|56.8724%|
| 2100/100% | R+S |1500|12000/12000|54.0447%|95.8582%|54.1364%|
| 525/25% | R |1500|24000/0|51.7867%|94.8368%|49.8595%|
| 相同525/25% | R+S |1500|12000/12000|52.4119%|95.0186%|50.4439%|

25%标注混合−纯R主AP+0.6251739566pp/AP50+.1817780608pp/AP75+.5843788934pp；100%混合−R主AP−1.1187688607pp。低标注正信号与全量负差同时保留，不能只选有利组；说明本配方观察收益随真实标注资源而变，不据单seed认定稳定副作用或普适贡献。25%混合相对全量R仍少2.7515238974pp，相对全量混合少1.6327550367pp；纯R由全量到25%少3.3766978540pp，混合补回.6251739566/3.3766978540=18.5144%性能缺口。这18.5%不是节省18.5%标注，也不能凭AP比例称等效。混合两数量档差较小，可描述较弱标注数量敏感性的开发现象，但全量混合起点本来低1.1188pp，不能因此直接证明更优标签效率曲线。

25%小/中/大目标AP：R29.7536/50.9662/58.0604%，mixed33.0732/51.2115/59.7778%，差+3.3195/+.2453/+1.7174pp；small为事后探索性分层，未给CI，不当多重比较后的独立通过线。真实图池525/同一冻结UID，未选1575未参与训练/伪标签，确实测试了更少真实标注条件。两方法同总更新和总24000图曝光，但mixed真实曝光只有一半；结果支持整体混合配方的实际效用信号，不能拆成合成内容/减少重复真实训练/训练噪声各自的因果贡献。

当前可得：在本固定seed/开发比赛/1500步配方内，加入合成数据在25%真实标注条件下带来小幅增益并部分补偿减标注损失，可称潜在标注效率价值。当前不可得：25%真实标注等效替代100%、稳定节省75%标注/人工工时、统计显著/独立比赛泛化或普适提升。没有预先非劣效/等效界限；本轮是既有完整GT上的低标注模拟，真实训练只准525，未测人工小时与合成制作成本。AP高或差距小都不能自行定义等效。不能改拿旧较弱R6563的51.65%当全量目标来宣布节省75%；本比较必须沿同1500充分开发R55.16%。

后续建议仅讨论：冻结当前配方后做配对重复seed及未反复开发的独立比赛确认；补50%档以形成25/50/100真实标注效率曲线。若要主张“维持性能节省多少标注”，在确认实验前事先定义有应用依据的允许AP损失/目标精度，报告不确定性和达到同目标所需N_R；不根据本结果倒定门限。若要分离减少R重复和S内容，可另议相同R曝光的纯R750控制，但这属额外机制问题，不自动扩执行。原全部正/负结果保留。

G label_budget_comparison.json保存四组绝对值/精确status引用SHA和所有差值/缺口计算，不从delta反推。有限N25两项及小控制/owned均闭合，bbox-n25原生已删除，每日g不动，之后无自动BBox训练/新seed/50%/SegPose。Seg保持仅CPU准备，不训练不GPU。EXPERIMENTS/PAPER原SHA保持。


## LABEL-CURVE-EXPAND-001

UTC 2026-11-03T23:57:31.662330+00:00。用户直接批准seed17的50%/10% R与混合4项，并在异步范围澄清中明确选择seed29/43复现全部100/50/25/10，共16追加训练。按主题更新方案第1/4.6/5/10节完整流程/绝对预算/顺序/独立确认；原c3450093007a43c16ae747ebc8a294ba7e1ffda202cd71d8329a7a57cfafc09f→5dcb2c911a5e1a985e2f64c4e3157b728c537c2f2f42989dac8fec7f6cbfb691，PAPER原SHA保持，旧冻包合法不回写。固定1050/210/2100 manifest及精确UID嵌套，原525文件bdac...不改；N50复用原nested50有序池192889...，N10每clip在原25中SHA选3整块，无val/test选样。新20训练各1500步、3cp/2val/完整记录，公开独立起点；current4后再16，非分数条件。训练seed29/43不改变数据选择seed17。机械历史795.8882010710282保持，新增前4cap240/后16cap960，总上限1995.8882010710282，不用未花额度开toy；每正式max90min/原Nov7截止。

独立比赛小metadata已核train4/7/11、dev6与test8/9无match交；旧116/187恰属train7/11不可当独立比赛。候选game8/9各750检测图，访问史保留，game9历史observed，game8有元数据/下载与GT结构检查，不称整体首次盲测。冻结24末EMA×两比赛=48评价/36000图，不重训确认比赛；未执行评价或读testGT内容/图像。全部原生预测上界258.1264GiB还需partial/瞬态/100GiB，不能把未来全部项当已整体资源PASS；逐项/阶段按实际剩余空间重算，不删旧证据。当前仅新代码与CPU准备，尚无本4项正式更新。


## LABEL-CURVE：label50_R_seed17 STARTED

UTC 2026-11-04T00:04:57.310139+00:00。公开YOLO11s独立起点seed17，唯一真实1050/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset50_seed17_v1.json", "sha256": "fa1b6729f7eb3438157c7abfb91c561062026f11aea40cd016f0da6f807f799b"}。策略R，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1050不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed17_20261104_a02/label50_R_seed17。


## LABEL-CURVE-N50-R-ADMIT-001

UTC 2026-11-04T00:08:42.535734+00:00。seed17新增四项的root独立18CPU PASS/0skip10.040秒，观察台预算/命名新曲线7PASS/0skip.001秒；stage2含失败历史门禁根19PASS/0skip1.114秒。初次用tests包名调用因本仓库tests非包而未发现有效用例，改真实discover入口后通过，非训练或数值失败。新label_curve_bbox.py SHA c6eaf4117441747aee277bcc4ffcbdd79c7b99b4d8341aecc2eb4a25f3c37e83，source42=原全量R1500冻41+此新模块；原step/backend/schedule/snapshot/restore code对象保留，loop仅完整源验证后过滤入口和原seed17硬码初始化行两处机械变化，AST逆等原loop。正式seed_all/samplers/loadergen分别实际configseed/configseed/seed*10000+17，seed17保持170017；CPU实际29/43 odd8R8S完整disk下一loss/梯度/参数/EMA/SGD/RNG exact，四份实际manifest池2100/1050/525/210及val750已核。旧N25/原41和GT不改。

首次N50_R预检G bbox_labelcurve_prepare_N50_R_20261104_a01，D R_seed17_20261104_a01尚无job_table/零正式更新；3诊断更新完成后在恢复重放step2返回处触发协作50秒保护，admission FAIL_STOP/TimeoutError elapsed52.266秒，原生全helper56.02125358581543秒/return1/ATTENTION，owned全部退出/GPU锁退休。两次step2 loss相同不能冒称已完成全状态exact，因为比较尚未执行；无强杀/放宽门限/原admission改PASS，全部原失败与日志保留。TemporaryDirectory正常收口无pt残留，但该失败没有最终checkpoint收据输出，不补造已删除临时诊断SHA。超时只定位到协作窗口，未测分阶段冷启动开销，不声称已证明根因是GPU慢。

按既定仅一次同协议有界基础设施retry，新独立G bbox_labelcurve_prepare_N50_R_20261104_a02/D R_seed17_20261104_a02保留原helper24dc443a48d835149d6db20266256de046386446c7219ae6db725858ccb60b1f和observerf1999a70d23b4c19c72dcbd49f77038e53a3c08779255744c386f324e2844b3f，数值/来源/时间门限均未改；原56.02125358581543完整入账。retry实际GPU机械PASS30.938秒/3诊断更新，raw/EMA/SGD/全部RNG/sampler/gen/iterator下一loss/实际梯度全exact，代表R/S cached/direct exact，实际catalog1050/未选1050进入0，cp114284143B，两个临时pt228568286B已展示精确清。原生GetProcessTimes全helper34.29388785362244秒/VERIFIED_EXIT/return0/owned退出/锁退；本块失败+PASS累计90.31514143943787/240，历史累计886.2033425104661/1035.8882010710282，不转未来预算、不重置。后16前置闭合receipt必须将此次失败单列failed_preparation_attempts，measured为4PASS加此FAIL耗时，stage2实际同协议/原requestSHA/42sourcebindings逐项核，不能把失败历史丢掉。

50%纯R正式于2026-11-04T00:04:44.863284Z（北京08:04:44）隐藏唯一启动：D/G main_bbox_yolo11s_soccernet_R_seed17_20261104_a02，job label50_R_seed17；wrapper22264/CT1793750684.8646336，controller23652/CT1793750684.882066，完整D本组launch.py，实际CT/cmd/GPU锁/小状态与观察台1050/2100预算均核。runtime起00:04:53.832400Z/max5400，固定deadline01:34:53.832400Z（北京09:34:53），不重置。初始0为PT0真实750验证，不冒称已有提交；不承诺Codex重启隔离。公开独立YOLO11s/seed17，1050原train UID与原预定50%有序池严格一致，960FP32/micro8x2有效16/workers2，单阶段1500/24000R0S，warm100/.001完整cosine到.00001，SGD/原损失增强/roleignore不改，0/1000/1500三完整cp、0/末两真实val与所有逐步/原生预测/失败保留，不best/test。

实际完整新增24295280650B=22.62674332596GiB另G100GiB，D32MiB另100GiB；cp按保守114293807*1.1、1500步2MiB、2x750(native6650344+JSON1MiB)、一次750partial、瞬态3GiB+2cp。actualfree G744721907712/D121696694272B，第二步不足.5秒按.5保守，完整time3630<=5400，已冻结；未声称其后19训练/48评价整体均PASS。launchSHA0f4a6365f732535c140772581b2ab3113ebe7551445c267f7ef2d0c27a04d451；job_table a389065811f477bd3d96c0832d0d81419332eec20a17e44e1ddcb237f14a01d4；codebindings051941b4d99a9b4679378c4ce5f08332bf4aca7b43184d66ef92ada6f3c20c88；admission68c84b5d5fe5a887a30694949a475b7b896b08c6584b5c29d290250dfb425176；storage223c2ab5ac1cba05082541fb93d34d09cf33f5c3f11aced978a44cc64ebc2690。

观察台v8实际localhost8765健康，PID5780/CT1793750441.2193077，72h有限服务截止2026-11-07T00:00:41.387415Z（北京11月7日08:00:41），早于研究全局Nov7T23:55Z。旧v7登记进程已退出，新用户20训练/48确认范围续开只读服务；根启动检查初次错请求/health导致验证helper报错，实际服务已正常启动，改/api/health后核实而未重复启动，不把此当训练故障。新schema预算/真实图数/训练seed显示已注册。原G全部训练证据不搬、不全哈媒体；Seg仍只CPU，不训练/GPU。


## LABEL-CURVE-SUBSET-AUDIT-001

选择已冻结后的原train GT metadata计数：{"50": {"images": 1050, "source_player_goalkeeper_annotations": 15757, "source_ignore_or_nontarget_records": 2628}, "10": {"images": 210, "source_player_goalkeeper_annotations": 3207, "source_ignore_or_nontarget_records": 565}}。仅事后量化监督规模，未解码图片/读取testGT或据此改选择；原subset/GT SHA保持。


## LABEL-CURVE-FOLLOWUP-001

UTC 2026-11-04T00:14:31.240578+00:00。原生同聊天heartbeat已创建ID bbox/ACTIVE/每30分钟/原2026-11-07T23:55Z截止，覆盖当前首50R与其余3项→seed29/43四档各8共16→固定24末EMA两比赛共48确认。正常无变化静默，不例行训练AP/steps巡查；逐项实际完工/SEALED小SHA/owned退出/锁退后继续。新stage2前置失败历史门禁d71a.../4d454...根19PASS/0skip1.114秒；实际首次FAIL与retryPASS的19项数值/source/plan/subset字段逐项相同，未改原FAIL。独立评价候选后端仅CPU准备仍在推进，尚未读取真实testGT内容/媒体、未模型加载或GPU评价，外层48资源/锁/可信EMA/限时/小control收口必须真实落实后才派，不能以本队列登记当Gate已PASS。当前首训练继续原42冻结，后加confirmation/Seg模块不接入。


## LABEL-CURVE：label50_R_seed17 COMPLETE

UTC 2026-11-04T00:31:14.026933+00:00。公开YOLO11s独立起点seed17，唯一真实1050/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset50_seed17_v1.json", "sha256": "fa1b6729f7eb3438157c7abfb91c561062026f11aea40cd016f0da6f807f799b"}。策略R，1500/1500成功更新，实际曝光{'R': 24000, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1050不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 23652, "create_time": 1793750684.882066, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 24000, "S": 0}, "schema_version": "bbox_label_curve_v1", "job_id": "label50_R_seed17", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_R_seed17_20261104_a02\\label50_R_seed17", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5391124870461028, "AP50": 0.9529373905354191, "AP75": 0.543726776977106, "AP_small": 0.3691995026487901, "AP_medium": 0.5282757909233087, "AP_large": 0.6030736712308223, "AR_max1": 0.05482703137570394, "AR_max10": 0.4582059533386967, "AR_max100": 0.633296460176991, "AR_small": 0.43543307086614175, "AR_medium": 0.6357739557739558, "AR_large": 0.6535483870967742}, "raw_prediction_bytes": 4267730843, "images": 750, "wall_seconds": 351.797, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5391124870461028, "AP50": 0.9529373905354191, "AP75": 0.543726776977106, "AP_small": 0.3691995026487901, "AP_medium": 0.5282757909233087, "AP_large": 0.6030736712308223, "AR_max1": 0.05482703137570394, "AR_max10": 0.4582059533386967, "AR_max100": 0.633296460176991, "AR_small": 0.43543307086614175, "AR_medium": 0.6357739557739558, "AR_large": 0.6535483870967742}, "raw_prediction_bytes": 4267730843, "images": 750, "wall_seconds": 351.797, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed17_20261104_a02/label50_R_seed17。


## LABEL-CURVE：label50_R_plus_S_seed17 STARTED

UTC 2026-11-04T00:37:38.349164+00:00。公开YOLO11s独立起点seed17，唯一真实1050/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset50_seed17_v1.json", "sha256": "fa1b6729f7eb3438157c7abfb91c561062026f11aea40cd016f0da6f807f799b"}。策略R+S，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1050不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed17_20261104_a01/label50_R_plus_S_seed17。


## LABEL-CURVE-N50-R-CLOSE-RS-START-001

UTC 2026-11-04T00:38:25.963186+00:00。当前50%纯R已于北京08:31:14 COMPLETE1500/R24000S0并终验，末AP50:95 53.9112%；3cp/2val/51小control818699B SEALED/owned退出/锁退。50%同1050混合于北京08:37:26唯一启动D/G main_bbox_yolo11s_soccernet_RS_seed17_20261104_a01，controller14132/CT1793752646.1709087、wrapper26616/CT1793752646.1537173；固定runtime={"started_at_utc": "2026-11-04T00:37:35.052315+00:00", "deadline_unix": 1793758055.0523157, "max_run_seconds": 5400}，0为PT0验证。独立GPU机械PASS33.610s，全进程37.0342175961s；本四项含旧FAIL累计127.3493590355/240。继续10R→10混合→seed29/43共16→两场48确认，全部原Gate/Nov7截止，Seg不训练/GPU。方案5dcb2c911a5e1a985e2f64c4e3157b728c537c2f2f42989dac8fec7f6cbfb691原bytes保持。

root实际终验纯R：checkpoint0/1000/1500收据路径/bytes、两750summary精确tag、末EMA、2MiB有界尾部update1500/R_exposures24000/S_exposures0、51份小control逐SHA及准确owned真实退出/GPU锁退休，无未知writer。controlSHA863af7fdf999d663814f45c413c94395b1eae48f260ef982b70fca21643ef905，statusSHA3729cff198d8177c516c52764bcac62619a99c5280e1f3ef212c1b324d78db69，G completion_verification.json；未重哈模型/NPZ/媒体，不重复driver COMPLETE。AP50:95=0.5391124870461028，AP50=0.9529373905354191，AP75=0.543726776977106；相对全量R1500低1.2521774604pp、相对25%R1500高2.1245203936pp，仅seed17开发集曲线现象，不先宣称标注等效。

混合独立预检actual catalog1050/unselected0，代表R/S cached/direct、奇数8R8S完整disk恢复下一loss/实际梯度/raw/EMA/SGD/RNG/sampler/loader全exact；3诊断更新/2临时cp228568286B已展示精确清，helper33.610秒，全原生GetProcessTimes37.03421759605408/VERIFIED_EXIT/return0/owned全退/锁退。本4项保留首FAIL56.02125358581543与首PASS34.29388785362244，总127.34935903549194；历史累计923.2375601065202。后10R/10RS按真实剩余额度各须仍有60，不能预先当均已准入或加预算。

正式同1050独立公开YOLO11s seed17/960FP32 micro8x2 workers2，1500步/R12000S12000，warm100/.001完整cosine至.00001/原SGD与loss增强；3cp0/1000/1500、两真实750val、所有逐步/原生预测/失败保留，末EMA不best/test。42源原bytes冻结。实际完整空间24295280650B另G100GiB/D32MiB+100GiB，freeG735634747392/D121695154176，time3630<=5400。dispatch {"started_at_utc": "2026-11-04T00:37:26.151839+00:00", "launch_sha256": "0f4a6365f732535c140772581b2ab3113ebe7551445c267f7ef2d0c27a04d451", "job_table_sha256": "ef45f4d34103b936ff21a540427e54b0710fef34b450b07817b384085b1c89b2", "code_bindings_sha256": "051941b4d99a9b4679378c4ce5f08332bf4aca7b43184d66ef92ada6f3c20c88", "admission_sha256": "188c39b30f5c0e1005528d5c6a095bbc46cee41d2fbe4e96135db9b515bff875", "storage_revision_sha256": "147d9ffc8eec45f501dff627345287c9882babdce1cfd3b46f9d1d8dbef3f1ab"}。startup status真实RUNNING/0；初始0是PT0验证，不能冒称已成功训练；进程重启隔离未验证。

## LABEL-CURVE：label50_R_plus_S_seed17 COMPLETE

UTC 2026-11-04T01:03:25.772545+00:00。公开YOLO11s独立起点seed17，唯一真实1050/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset50_seed17_v1.json", "sha256": "fa1b6729f7eb3438157c7abfb91c561062026f11aea40cd016f0da6f807f799b"}。策略R+S，1500/1500成功更新，实际曝光{'R': 12000, 'S': 12000}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1050不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 14132, "create_time": 1793752646.1709087, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 12000, "S": 12000}, "schema_version": "bbox_label_curve_v1", "job_id": "label50_R_plus_S_seed17", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_RS_seed17_20261104_a01\\label50_R_plus_S_seed17", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5420737254675314, "AP50": 0.9554526598559951, "AP75": 0.5436078567109385, "AP_small": 0.34660672170712403, "AP_medium": 0.5307576076260638, "AP_large": 0.6149965132572532, "AR_max1": 0.05481697506033789, "AR_max10": 0.4609513274336283, "AR_max100": 0.6394006436041834, "AR_small": 0.402755905511811, "AR_medium": 0.6418673218673219, "AR_large": 0.6660645161290322}, "raw_prediction_bytes": 4265999281, "images": 750, "wall_seconds": 345.7349999999997, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5420737254675314, "AP50": 0.9554526598559951, "AP75": 0.5436078567109385, "AP_small": 0.34660672170712403, "AP_medium": 0.5307576076260638, "AP_large": 0.6149965132572532, "AR_max1": 0.05481697506033789, "AR_max10": 0.4609513274336283, "AR_max100": 0.6394006436041834, "AR_small": 0.402755905511811, "AR_medium": 0.6418673218673219, "AR_large": 0.6660645161290322}, "raw_prediction_bytes": 4265999281, "images": 750, "wall_seconds": 345.7349999999997, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed17_20261104_a01/label50_R_plus_S_seed17。


## LABEL-CURVE：label10_R_seed17 STARTED

UTC 2026-11-04T01:22:29.230101+00:00。公开YOLO11s独立起点seed17，唯一真实210/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset10_seed17_v1.json", "sha256": "fccc2c998d3b6a6dd4316db0f2a66d97aa5f02571208df360b58911d611245b1"}。策略R，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1890不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed17_20261104_a03/label10_R_seed17。


## LABEL-CURVE-N50-PAIR-N10-R-START-001

UTC 2026-11-04T01:23:30.042170+00:00。50%标注配对已完整验收：R53.9112%、R+S54.2074%，混合主AP+0.2961pp，仍低于全量R0.9561pp；同1050池/1500步。10%纯R(210图)于北京09:22:17唯一启动D/G main_bbox_yolo11s_soccernet_R_seed17_20261104_a03，controller11576/CT1793755337.0921447、wrapper26308/CT1793755337.0750775；runtime {"started_at_utc": "2026-11-04T01:22:25.805835+00:00", "deadline_unix": 1793760745.8058355, "max_run_seconds": 5400}，初始0是PT0。N10独立机械PASS46.953s/全50.4237632751s，含旧FAIL本4累计177.7731223106/240；余62.2268776894s，仅足下一项60s许可，不加预算。随后10混合→seed29/43各8→24模型两场48确认，原Gate/Nov7截止，Seg/Pose不训练/GPU，方案5dcb原bytes保持。

N50混合于01:03:25Z/北京09:03完成，root核3cp0/1000/1500收据/bytes、2精确tag真实750summary、末EMA、2MiB有界末step1500/R12000S12000、51小control818032B逐SHA、准确14132/26616和knownworker退出/GPU锁退休无未知writer。controlSHA7baa61feedb433fe83c7ebe004e7ae0262f096d3c6072afd02daa8544acdc921/statusSHAdf9a0530775c4f4d0409a9a57451480678742a2b93ce3150ccbeb424214bb31e，G completion_verification.json；driver已有COMPLETE不重复，未全哈模型/NPZ/媒体或搬整组。N50混合AP50:95 .5420737254675314/AP50 .9554526598559951/AP75 .5436078567109385；同N50混合-R pp={"AP50_95": 0.29612384214285914, "AP50": 0.2515269320575997, "AP75": -0.01189202661675548}。相对全量R主AP gap=-0.9560536182633084pp。G label_budget_comparison.json保存已完成6组绝对值/精确引用；N50/25有小幅正开发信号，全量负差同时保留，不将50%混合略高全量混合当等效/普适节约标注，不自设非劣效线。

N10纯R独立准入：真实训练catalog210、未选1890进入0，原完整source/val/GT不改；R/S cached/direct exact，3诊断更新disk完整恢复下一loss/实际梯度/raw/EMA/SGD/RNG/sampler/gen/iterator全exact。2临时pt228558430B已显示收据精确清，cp114279215B。helper46.953秒；持句柄GetProcessTimes全helper50.423763275146484/VERIFIED_EXIT/return0/precise/owned全退/锁退。历史累计973.6613233816666，四项累计177.77312231063843/240（保留首FAIL），后10混合须按剩余62.22687768936157核完整60许可。

10R公开YOLO11s独立seed17/210原train图/1500更新R24000S0，960FP32/micro8x2/workers2/warm100 .001完整cosine至.00001/原SGD loss增强roleignore，3cp0/1000/1500和2真实750val，逐步/原生预测/失败完整，末EMA不best/test；42数值原bytes冻结。a03为a01预检FAIL与a02已闭合50R之后的新N10命名冲突消解，不是失败retry。实际完整空间24295280650B另G100GiB/D32MiB+100GiB/freeG726559879168/D121693679616，time3630<=5400。dispatch={"started_at_utc": "2026-11-04T01:22:17.073851+00:00", "launch_sha256": "0f4a6365f732535c140772581b2ab3113ebe7551445c267f7ef2d0c27a04d451", "job_table_sha256": "da15dc8cb74ac4b99279fae963d56b109f32821c64b5af012bb20a8e54e283dc", "code_bindings_sha256": "051941b4d99a9b4679378c4ce5f08332bf4aca7b43184d66ef92ada6f3c20c88", "admission_sha256": "71621a1c28008f169fb0884a81719a22a7f7f46594d78951d9a1bed138a1b712", "storage_revision_sha256": "9ab4a5ce7f49b3cea66273ce7246baaef4a9510631f5867d5c40a9f069ff3a59"}。startup真实RUNNING/0，0为PT0验证非训练提交；进程隔离未验证。

独立确认外围CPU准备继续委派既有代理bbox_independent_audit，仅虚构CPU/不得真实testGT媒体权重/GPU/派发，不改已核候选或原42；仍需root审阅/真实24闭合registry/资源与唯一生命周期Gate，不能当48已准入。


## MONITOR-FILTER-001

UTC 2026-11-04T01:35:21.224050+00:00。用户直接要求训练期间优化观察台记录选择，新增用途/任务/模型/数据源/策略/seed/真实标注量/总更新预算/运行状态九字段组合筛选、关键词AND搜索、联动选项数量/匹配数/零结果提示、一键清空、同浏览器本地记忆。查看/对比共用过滤；已手选但筛选外的记录保留并标注，不自动换图；自动跟随明确不受筛选约束。所有菜单继续显示预算与状态，原始记录不改。仅改HTML与已有静态JS，服务按请求读原静态路径，未改Python服务/未重启PID5780/未延原服务截止。

客户端21项PASS/0skip，含8项新增组合/未知预算/零结果/选择保留/不变性和原13曲线回归。真实IAB浏览器验证50%+1500匹配2/16、50%+R+S匹配1/16、零匹配保留选中、刷新后筛选恢复、清空回16/16；读到同组主/对比重复时按systematic-debugging实际复现，原因是菜单排除了主记录但未同步清理compareId，补回互斥检查后浏览器核compare选择空/selectedIndex0且只剩主曲线。截图位于允许visualizations工作区training-monitor-filters.jpg。

静态资源SHA={"tools/training_monitor.html": "f6b817c1061ded762179bc5885712d626c6be5e8a5cdcc84e7fea84656cb6eb9", "tools/training_monitor_charts.js": "20c01a7cf5facebf7389e58cb4de7bcfcb567e6b10ea14b43958128270a3bc6b", "tests/test_training_monitor_filters.cjs": "1d8963bdddf6eddab9823c45d13ee8acc1dc7c1a2b5afc0c8d1e8db37b2edec9"}。训练唯一controller11576/CT1793755337.0921447精确launch仍活，42codebinding/label_curve模块和EXPERIMENTS/PAPER SHA保持；未读PT/NPZ/媒体/真实testGT，未调用GPU或修改/停止训练。Seg/Pose仍不训练。


## LABEL-CURVE：label10_R_seed17 COMPLETE

UTC 2026-11-04T01:47:27.710720+00:00。公开YOLO11s独立起点seed17，唯一真实210/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset10_seed17_v1.json", "sha256": "fccc2c998d3b6a6dd4316db0f2a66d97aa5f02571208df360b58911d611245b1"}。策略R，1500/1500成功更新，实际曝光{'R': 24000, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1890不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 11576, "create_time": 1793755337.0921447, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 24000, "S": 0}, "schema_version": "bbox_label_curve_v1", "job_id": "label10_R_seed17", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_R_seed17_20261104_a03\\label10_R_seed17", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.48613034511755193, "AP50": 0.919438674920068, "AP75": 0.4472762478267003, "AP_small": 0.2863592159193913, "AP_medium": 0.471657786006334, "AP_large": 0.5659777795349183, "AR_max1": 0.05095534995977474, "AR_max10": 0.4350462590506837, "AR_max100": 0.5741351568785197, "AR_small": 0.3291338582677165, "AR_medium": 0.5714987714987714, "AR_large": 0.6290322580645161}, "raw_prediction_bytes": 4270167321, "images": 750, "wall_seconds": 346.21799999999985, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.48613034511755193, "AP50": 0.919438674920068, "AP75": 0.4472762478267003, "AP_small": 0.2863592159193913, "AP_medium": 0.471657786006334, "AP_large": 0.5659777795349183, "AR_max1": 0.05095534995977474, "AR_max10": 0.4350462590506837, "AR_max100": 0.5741351568785197, "AR_small": 0.3291338582677165, "AR_medium": 0.5714987714987714, "AR_large": 0.6290322580645161}, "raw_prediction_bytes": 4270167321, "images": 750, "wall_seconds": 346.21799999999985, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed17_20261104_a03/label10_R_seed17。


## LABEL-CURVE：label10_R_plus_S_seed17 STARTED

UTC 2026-11-04T01:51:16.601843+00:00。公开YOLO11s独立起点seed17，唯一真实210/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset10_seed17_v1.json", "sha256": "fccc2c998d3b6a6dd4316db0f2a66d97aa5f02571208df360b58911d611245b1"}。策略R+S，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1890不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed17_20261104_a02/label10_R_plus_S_seed17。


## LABEL-CURVE-N10-R-CLOSE-RS-START-001

UTC 2026-11-04T01:52:16.807644+00:00。10%纯R已于北京09:47:27 COMPLETE1500/R24000S0，root终验3cp/2val/51小control818847B SEALED/owned退出/锁退，末AP50:95 48.6130%。同210真实图的10%混合于北京09:51:04唯一启动D/G main_bbox_yolo11s_soccernet_RS_seed17_20261104_a02，controller7196/CT1793757064.340304、wrapper13608/CT1793757064.3224485；runtime {"started_at_utc": "2026-11-04T01:51:13.171241+00:00", "deadline_unix": 1793762473.1712415, "max_run_seconds": 5400}，初始0为PT0非训练。独立机械PASS33.063s/全36.4252002239s，四项含旧FAIL总214.1983225346/240，原额度未增加。待本混合全收口后才写四项完工+失败历史receipt并开始seed29/43各8共16，之后24末EMA两比赛48确认；原Nov7截止/Gate，Seg/Pose不训练/GPU。方案5dcb/PAPERd48cb原bytes保持。

用户要求下一实验并提及历史S→R；root核当前队列实际已完成10%R，按已批准四项顺序继续唯一10%混合，未重跑历史S→R。N10_R actual末EMA AP50:95 .48613034511755193/AP50 .919438674920068/AP75 .4472762478267003。root核3完整cp0/1000/1500收据/bytes、两精确tag/750summary、2MiB有界末step1500/R24000S0、51小control818847B逐SHA、准确11576/26308与knownworkers退出、GPU锁退休无未知writer；controlSHA99072f731bdd317b9625777f2dd1c22b27ed7ce8208b4e8d1d56d3f9d7cb3116/statusSHA852d881e25a8b9ba18d93c5ac480e7a6b4a98755cc4dd62e411dfea333cbdea2，G completion_verification.json。driver已有COMPLETE不重复，未重哈模型/NPZ/媒体或搬整run。

N10混合独立机械PASS33.063秒/3诊断更新，catalog210/未选1890进入0，R/S cached/direct exact、完整disk下一loss/实际梯度/raw/EMA/SGD/所有RNG/sampler/loader exact；cp114279215B，两临时pt228558430B已展示收据精确清。GetProcessTimes全helper36.42520022392273秒/VERIFIED_EXIT/return0/precise/owned退出/锁退，lifeSHA4fac1ee71fdae63b67a3408b526b39922b54dfc59fa8cbff7f87872d014019f9；四项预检含首FAIL累计214.19832253456116<=240，历史累计1010.0865236055894<=1035.8882010710282，未擦除FAIL/重置预算或转额度开toy。

正式同210冻结UID/公开独立YOLO11s seed17/960FP32 micro8x2有效16 workers2；每micro交错4R4S，1500步R12000S12000，warm100 .001/momentum.8到.937完整cosine至.00001/原SGD Nesterov WD.0005 clip10与loss增强roleignore，原42数值冻结；3cp0/1000/1500、两真实750val、逐步与全部native预测/失败保留，末EMA不best/test。RS a02为已闭合N50 a01后的新N10协议命名冲突消解，非失败retry。实际完整空间24295280650B另G100GiB/D32MiB+100GiB，freeG717475303424/D121586368512，time3630<=5400。dispatch={"started_at_utc": "2026-11-04T01:51:04.320688+00:00", "launch_sha256": "0f4a6365f732535c140772581b2ab3113ebe7551445c267f7ef2d0c27a04d451", "job_table_sha256": "6f103d1ffa1f259a68f86578fd4590c37c600fa2490c6cbbbe7139800c7b15f7", "code_bindings_sha256": "051941b4d99a9b4679378c4ce5f08332bf4aca7b43184d66ef92ada6f3c20c88", "admission_sha256": "8576661505350c0cc09357e75da49fadb19676f6a0723149cc340dfcb9141b71", "storage_revision_sha256": "88e4dc9ebfa9b2eb9aad68c1996eb0e2c7a78445a0364fe1be69badd8ffcd9d0"}。startup实际RUNNING/0，0是PT0真实验证非提交训练，隔离未验证。

独立确认外围CPU代理交付候选bbox_confirmation_lifecycle.py f051e878281c8aade7de1bbc11e900dcc3681176e708f0febbd9806fc17c11af、observer8ce2991eeac2496dd095802dd33d8ec9b4cb1bb54f761edac606c4d52695497e、tests8853964d806fe834e694f4c079d3ccad6d9e8b8c54199bb19d2d417fb6f32a9a，代理23CPU PASS5.730s，root尚需独立审阅验证。原candidate785ae4/训练42不改；外围iterator唯一机械AST接点最外_shutdown、43源、900s/STOP/owned/SEALED，不构成真实48准入，真实24 registry/ignore与空间/GPU仍待Gate。

## LABEL-CURVE：label10_R_plus_S_seed17 COMPLETE

UTC 2026-11-04T02:16:48.566680+00:00。公开YOLO11s独立起点seed17，唯一真实210/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset10_seed17_v1.json", "sha256": "fccc2c998d3b6a6dd4316db0f2a66d97aa5f02571208df360b58911d611245b1"}。策略R+S，1500/1500成功更新，实际曝光{'R': 12000, 'S': 12000}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1890不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 7196, "create_time": 1793757064.340304, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 12000, "S": 12000}, "schema_version": "bbox_label_curve_v1", "job_id": "label10_R_plus_S_seed17", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_RS_seed17_20261104_a02\\label10_R_plus_S_seed17", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5015849435184775, "AP50": 0.9406538760724287, "AP75": 0.4637188646302103, "AP_small": 0.3498675157436837, "AP_medium": 0.49221871127032185, "AP_large": 0.5696252498946437, "AR_max1": 0.051609010458567986, "AR_max10": 0.4391995172968624, "AR_max100": 0.5939159292035399, "AR_small": 0.3901574803149606, "AR_medium": 0.5948894348894349, "AR_large": 0.6229677419354839}, "raw_prediction_bytes": 4259885250, "images": 750, "wall_seconds": 334.7029999999995, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5015849435184775, "AP50": 0.9406538760724287, "AP75": 0.4637188646302103, "AP_small": 0.3498675157436837, "AP_medium": 0.49221871127032185, "AP_large": 0.5696252498946437, "AR_max1": 0.051609010458567986, "AR_max10": 0.4391995172968624, "AR_max100": 0.5939159292035399, "AR_small": 0.3901574803149606, "AR_medium": 0.5948894348894349, "AR_large": 0.6229677419354839}, "raw_prediction_bytes": 4259885250, "images": 750, "wall_seconds": 334.7029999999995, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed17_20261104_a02/label10_R_plus_S_seed17。


## BBOX-BATCH-AUTH-001

UTC 2026-11-04T02:25:30.615164+00:00。最新用户直接授权批量执行seed29/43全部16组，取消逐项重复CPU测试/恢复探针/GPU机械预检；复用已验证同配置的真实证据，使用独立复用准入schema明示fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新测量PASS。保留原42训练数值/完整记录/公开独立起点、单GPU互斥、存储/截止和前项完整收据闭合；任一失败/人工STOP即停队列，无自动retry/resume。EXPERIMENTS新SHA32c9415b2d81de7fce6a9221af72495822ffe3b70e537004b7bb64af02b065a2，PAPER d48cb原值，旧5dcb run合法历史原bytes不回写。N10混合已于北京10:16:48 COMPLETE1500/R12000S12000并root终验，主AP50.1585%/+1.54546pp；seed17四新增项全部闭合，curve4_completion SHA024a4606b7a24aa1bae1fb3fa955fbbaeb34fbb55f444d530e975dabedb3a064/实际准备214.19832253456116含旧FAIL。批量工具当前正在实现，尚未seed29派发，不运行旧逐项预检入口。

原方案5dcb2c911a5e1a985e2f64c4e3157b728c537c2f2f42989dac8fec7f6cbfb691→32c9415b2d81de7fce6a9221af72495822ffe3b70e537004b7bb64af02b065a2，按用户新直接请求仅调整批量衔接/同配置证据复用与后16重复预检条款；方案仍只写研究流程，实际授权/变更与结果记此唯一日志。后16原960秒新增探针不执行、不转作其它搜索；历史1010.0865236055894秒不重置。新批量工具源码独立冻结，不加入训练42，不复制admission改seed伪装GPU实测；GPU/真实测试/其它seed比例扩展未新增。

N10混合root核3cp0/1000/1500小收据bytes、2精确tag真实750summary/末EMA、2MiB有界末step1500/R12000S12000、51小control818142B逐SHA、准确7196/13608及worker退出/GPU锁退休无未知writer。controlSHAcc88a2190517616fa81e3d2f757e96e7686343d727ffc6277917d3f611ce9f5d/statusSHAdc398a0a0e3e1537c6f7e35397856ba1307179de4be8477926c9e7498dbc463e，G completion_verification.json。AP50:95 .5015849435184775/AP50 .9406538760724287/AP75 .4637188646302103；同210纯R .48613034511755193/.919438674920068/.4472762478267003，对应差+1.5454598401/+2.1215201152/+1.6442616804pp。是单seed开发信号，未预设等效界限，不宣称节省90%标注或普适收益。


## LABEL-CURVE-BATCH FAIL_STOP

UTC 2026-11-04T02:32:24.993336+00:00。已完整闭合0/16，具体ValueError: 准入收据不一致。保留G:\FutsalMOT_Training\runs\bbox_label_curve_batch_20261104_a01\result.json，无自动重试/恢复/强杀或跳过。


## LABEL-CURVE：label100_R_seed29 STARTED

UTC 2026-11-04T02:36:27.385224+00:00。公开YOLO11s独立起点seed29，唯一真实2100/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset100_seed17_v1.json", "sha256": "81e363fe89f3b1cda00f967487ad033ee84e00d68654d96ea305ca0680a05f92"}。策略R，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选0不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed29_20261104_a02/label100_R_seed29。


## BBOX-BATCH-START-001 / BBOX-BATCH-JSON-001

UTC 2026-11-04T02:38:08.170712+00:00。固定16组批量a02已于北京10:35:57唯一启动：G bbox_label_curve_batch_20261104_a02，CPU supervisor12512/CT1793759757.6260707、wrapper15864/CT1793759757.6090045。首项seed29/N100 R实际RUNNING，D/G main_bbox_yolo11s_soccernet_R_seed29_20261104_a02，controller27912/CT1793759774.6515486；runtime {"started_at_utc": "2026-11-04T02:36:23.680750+00:00", "deadline_unix": 1793765183.6807508, "max_run_seconds": 5400}，0为PT0验证。队列自验收衔接seed29再43各四档R/混合，无需聊天逐项启动；复用已验证配置，新诊断0/新CPU和GPU预检false，原42/1500/90min/3cp2val/完整记录不变。轻量互斥/空间/截止/完工检查保留；任一失败/STOP停队列，不自动retry/resume。首a01为JSON元组/列表准备差异零派发FAIL，原字节保留，a02只做一次明确机械修订。方案SHA32c9415b2d81de7fce6a9221af72495822ffe3b70e537004b7bb64af02b065a2，PAPER原d48cb；旧5dcb冻结合法，Seg/Pose不训练/GPU。

新工具原67ff025...只做4个极小调度mock0.021s、无torch/训练CPU夹具/GPU预检；root审阅其固定队列/复用准入/42数值不变/子进程与小收口路径。首次真实batch a01在派发前原prepare_group actual==admission抛ValueError准入收据不一致，0正式更新/无job_table或launch或dispatch/G训练组，旧supervisor27004与wrapper17812/host全部真实退出/两锁退。只读重建元数据定位59处source identity在内存为tuple、JSON为list，canonical表示一致，非数值失败。失败G原result/queue/helper与D空准备组admission/storage原bytes保持，preparation_failure_verification SHA0d446d43981ca1a25244904ccba3bb0fc480a67440e9b787f790a4344c9ddafd。

一次明确同协议机械修正：make_reuse_admission写盘后读回唯一admission，再传原prepare，保留严格实际bytes比较；新known_preparation_failures只允许精确绑定且原D只有admission/storage、无正式产物/G输出的已核零派发失败作命名消解。工具新SHA3affe1982935dd7f3bbe940001bfbeababcbd3b401fb56fb24480aa7b82640ca，旧G67ff不热改，不用现行4个旧mock成绩冒称修正版已测；新实际首任务准备/冻结/启动已通过。新G batch a02 queueSHA6813d2f8745c0d169776535eb872edd2d717e12b11e8b6dffc3b41a62cd67c7d、launchSHA068a75d5a757977d8b178fc66c967d7cf0a8db30b1593e1625678f0ec36bb58c、prepare五处准入接点逆AST精确，原训练42/source前端c6eaf不改。R seed29 a02是原a01准备失败后首次正式模型，不恢复/覆旧组。

本批已批准训练seed29再43，各100/50/25/10标注量R后混合共16，公开独立YOLO11s/960FP32/micro8x2/workers2/1500步，总R24000或R12000S12000；标注子集选择仍seed17固定2100/1050/525/210，原GT/UID不改。新独立复用准入schema/scope与fresh=false/diag0/measurement0明确绑定旧四项真实GPU/source证据，不冒称seed29实际新预检。原driver写每项STARTED/COMPLETE，batch在唯一日志补完整验收，不重复driver条目。每项完整新增24295280650B，批量按remaining*该上界+G100GiB和remaining*D32MiB+D100GiB、remaining*(5400+600)核原Nov7截止；每个模型runtime自原代码固定5400秒。全队列不自动开独立比赛评价，后48仍需实际资源/24模型/GT等Gate。

实际首job dispatch={"schema": "bbox_batch_child_dispatch_v1", "batch_queue_reference": {"path": "G:\\FutsalMOT_Training\\runs\\bbox_label_curve_batch_20261104_a02\\queue.json", "sha256": "6813d2f8745c0d169776535eb872edd2d717e12b11e8b6dffc3b41a62cd67c7d"}, "index": 0, "started_at_utc": "2026-11-04T02:36:14.635306+00:00", "wrapper_pid": 19676, "wrapper_create_time": 1793759774.634079, "command": ["D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\.venv\\Scripts\\python.exe", "-B", "-X", "utf8", "-u", "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\runs\\main_bbox_yolo11s_soccernet_R_seed29_20261104_a02\\launch.py"], "launch_sha256": "0f4a6365f732535c140772581b2ab3113ebe7551445c267f7ef2d0c27a04d451", "job_table_sha256": "9ee07c033c164271077fc144431a29388c6490667bd44c60fc6626d4f83e412c", "code_bindings_sha256": "051941b4d99a9b4679378c4ce5f08332bf4aca7b43184d66ef92ada6f3c20c88", "admission_sha256": "3a5520f7b768539356be7c553384bd3aed407b22e5d72f02e37afb537e85accd", "storage_revision_sha256": "4ef9b5fba0ff5012c6c2529d56771be0d2b4a5e072aa98dde3cf89ccce9aae4e", "process_lifetime_isolation_verified": false, "processes": [{"pid": 19676, "create_time": 1793759774.6340785, "command": ["D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\.venv\\Scripts\\python.exe", "-B", "-X", "utf8", "-u", "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\runs\\main_bbox_yolo11s_soccernet_R_seed29_20261104_a02\\launch.py"]}, {"pid": 28128, "create_time": 1793759774.637575, "command": ["\\??\\C:\\WINDOWS\\system32\\conhost.exe", "0x4"]}, {"pid": 27912, "create_time": 1793759774.6515486, "command": ["C:\\Users\\20113\\AppData\\Roaming\\uv\\python\\cpython-3.12-windows-x86_64-none\\python.exe", "-B", "-X", "utf8", "-u", "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\runs\\main_bbox_yolo11s_soccernet_R_seed29_20261104_a02\\launch.py"]}]}；runtime={"started_at_utc": "2026-11-04T02:36:23.680750+00:00", "deadline_unix": 1793765183.6807508, "max_run_seconds": 5400}；startup RUNNING/0，0是PT0真实验证，非提交训练。G batch与首组startup_verification记录准确身份/锁/数值未变/两stderr0；不承诺Codex重启进程隔离。G batch/STOP.json会仅对精确自有当前child写D STOP更新边界收口，不强杀；失败/未知提交不新attempt或自动续。


## SEG-READINESS-006 / SEG-PREDICTION-CPU-001

UTC 2026-11-04T02:58:20.296458+00:00。用户授权在BBox批量训练期间继续Seg来源/疑难GT和最小实验准备，明确不启动Seg训练。只新增独立CPU候选模块，未加入BBox冻结42，未改活动batch/queue/run、旧Seg backend、原GT/候选/分组与两方案。无模型加载、无优化器更新、无CUDA/Seg GPU预检、无媒体下载/全库重栅格/全媒体hash。

来源只读复核：ihelon SOURCE_CARD的ref=ihelon/football-player-segmentation、平台CC0 v1；Quantigo ref=quantigoai/soccer-dataset、平台CC BY-SA4.0 v1。平台声明不替代上游视频授权/身份；ihelon图像URL为空/date_captured=0，Q info.url=None，仍缺可绑定的原视频URL、比赛日期/身份。代理只读打开两Kaggle原页，本次无可读正文，未获得新增原始比赛来源证明，不据队服或平台名猜定比赛。1359 UID仍两保守组/全部split=null，Q三批同组。复用433664跨源配对的既有指纹结果，仅说明未检出exact/<=8 pHash候选，不能证明比赛独立；两来源候选不能兼作已证独立训练/开发/确认三组。类别仍ihelon person、Q Player+Referee→person0，其他类别显式非人物；整场人物覆盖并未由类别名称自动得到认证。

新增segmentation_readiness.audit_candidate_constraints从已绑定GROUP_SPLIT_CANDIDATE及原始四COCO/派生document/manifest/ledger读小JSON，无RGB解码：实际核1359精确图像身份、2组、8条除ID外完全一致的重复、1764条polygon-part自交finding涉及526个源图、4疑难实例与1同RGB异GT对。526为四COCO合计受影响源图，非“全部错误/应删除”数量。返回BLOCKED_NOT_ADMITTED，训练/激活split始终false。首调用严格比较完整entry因bundle额外counts而拒绝；只改为按名称唯一匹配并严格逐一核name/source_binding/document/ledger/manifest五项引用，保留SHA校验，原JSON未改。

处理队列：精确重复仅在派生候选内保留最小ID与全部aliases，不按IoU合并；原多边形→原COCO像素RLE仅表示转换，不称人工修复。Q4482(腿状mask/覆盖不完整待审)、Q5895(原2像素/ratio2消失)、ihelon5904(原3像素/960消失)、4261(后排遮挡边界)仍PENDING_REVIEW_WHOLE_IMAGE_OR_LEGAL_IGNORE。同RGB ihelon4.jpg(image321)/517.jpg(image451)各16人物GT不同，配对MaskIoU均值.9101703578、无一对像素完全相同，保持must-link同组/同split且PENDING_REVIEW_WHOLE_DUPLICATE_PAIR；不挑版本、不合并两套GT。共6张语义疑难/异GT图在候选队列待审，未实际删除/隔离任何原图。人工裁定需绑定源SHA、annotation与上下文；若排除须整图，或另实现合法ignore，不能删除一人而当背景。自交不能buffer/凸包/桥接或整批删除。

新增segmentation_prediction：只接已detach的单图CPU FP32原生116通道预测与32通道proto，固定person0/score.001/IoU.7/maxdet300/NMS torchvision.ops.nms，官方COCO segm maxDets1/10/100。绑定Ultralytics8.3.237的ops/nms/segment predictor源码与torchvision0.23.0+cu128/boxes.py；clone保护原生张量。原生FP32预测/proto/NMS输入框/原图框写单图native.npz，JSON保留原图RLE/score/UID/来源/几何/策略，SHA+bytes小收据，不覆盖失败/已有目录。MaskAP桥仍用原图GT、原area/bbox，不拿训练过滤mask作GT。

初版8项CPU夹具在31x61及61x31拒绝，明确复现64输入的实际top/left15与推断16的1像素差，未掩盖失败。候选改为recorded_letterbox_logits_v1：proto logits双线性至输入→按实际left/top/resized_shape去pad→双线性到原图→裁框/阈值0；框按实际resized_shape分轴逆变换。不冒称与原生retina推断padding逐值等价，该候选策略尚未正式实验冻结。独立代码审阅发现NMS导入分支与Nx4K mask内存问题，已显式先import torchvision固定分支并记录版本，逐实例chunk1逆映射/编码、原图最多3840*2160、单图native与RLE JSON分别32MiB拒超界；这些是候选上界，不是已测公开s/960或全库资源PASS。

新必要CPU检查：readiness 6项PASS/0skip0.007秒，prediction 10项PASS/0skip2.853秒。只测新接口，零模型/optimizer/GPU，不重复BBox预检或旧Seg backend训练/恢复夹具。覆盖同RGB拆组/非精确合并/篡改疑难SHA/已激活split拒绝，原生identity几何parity、奇数长宽独立坐标、NMS/系数、孔洞/断开、空检测/空mask、原GT MaskAP与native/RLE/SHA往返、逐实例和大原图拒绝。临时预测证据仅测试TemporaryDirectory，测试结束清理，未新增报告树。

还缺的实际条件：①原比赛/视频来源链、正式比赛分组及精确train/dev UID（独立确认另需独立组，不随机逐帧拆）；②疑难GT/同RGB异GT与小目标监督覆盖裁定、所有方法共用的最终派生视图；③新逆几何/NMS/COCO与辅助BBox评价的正式协议及真实公开s/960全路径接线；④完整500阶段的持久提交/停止/恢复、原生输出/partial/worker退出/小control生命周期；⑤合成1600帧实际固定清单与完整输出/内存/时间/单GPU资源；未来实际公开s/960恢复与GPU运行仍需单独准入，当前用户禁止不执行。已有CPU训练恢复backend a3e8...保持原bytes，本轮未重测、不借旧11PASS冒充新生产执行器。记录当前可完成部分和未满足条件，不伪造最小训练已可启动。

代码SHA：{"src/futsalmot/segmentation_prediction.py": "2e86b059765545593b9fe78b167be0d36900123f81aeab8defe3af958efa6567", "src/futsalmot/segmentation_readiness.py": "3ad1d7cc9f94360d4a1cbf1ac7915345744d4afb5716872fee8baf411bc058ee", "tests/test_segmentation_prediction.py": "cd96affcdda5c323a2b7b5660815cbf21ad924068526895c67a997a6dc4b0d82", "tests/test_segmentation_readiness.py": "c7dcffe683aa909e14fb83230823cb4188c99e5840e56f0474f5f75ad9e95a6c"}

证据：G:/FutsalMOT_Dataset/real/football_seg_public/derived/GROUP_SPLIT_CANDIDATE_20261103.json SHA ebee7c78508ff726e2f06cb891022323b91859e00119cbc61ba7c991a3473ea0；CROSS_SOURCE_FULL512_AUDIT_20261103.json SHA de5c159909fb5c091c5afbe0368a72e40146a97faa6c23fca4a923540e4f4681；DERIVED_QUALITY_REVIEW_LEDGER_20261103.json SHA a7d22e18a9c81c63f15a54c91a188de44173ed3a862ba07902e02e22f963bcd9；四COCO/派生RLE小摘要链实际核，未写这些文件。EXPERIMENTS32c941.../PAPERd48cb...前后保持。BBox批量交接由原CPU supervisor继续负责，root本轮未单独prepare/派发任何训练。


## LABEL-CURVE：label100_R_seed29 COMPLETE

UTC 2026-11-04T03:00:56.025855+00:00。公开YOLO11s独立起点seed29，唯一真实2100/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset100_seed17_v1.json", "sha256": "81e363fe89f3b1cda00f967487ad033ee84e00d68654d96ea305ca0680a05f92"}。策略R，1500/1500成功更新，实际曝光{'R': 24000, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选0不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 27912, "create_time": 1793759774.6515486, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 24000, "S": 0}, "schema_version": "bbox_label_curve_v1", "job_id": "label100_R_seed29", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_R_seed29_20261104_a02\\label100_R_seed29", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.548592284796249, "AP50": 0.9567477711954151, "AP75": 0.5582864111392855, "AP_small": 0.34887948495115484, "AP_medium": 0.5380217304573179, "AP_large": 0.6118205394961417, "AR_max1": 0.054816975060337905, "AR_max10": 0.46374698310539014, "AR_max100": 0.643543845534996, "AR_small": 0.4110236220472441, "AR_medium": 0.6475798525798526, "AR_large": 0.6609677419354839}, "raw_prediction_bytes": 4270687714, "images": 750, "wall_seconds": 329.1409999999996, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.548592284796249, "AP50": 0.9567477711954151, "AP75": 0.5582864111392855, "AP_small": 0.34887948495115484, "AP_medium": 0.5380217304573179, "AP_large": 0.6118205394961417, "AR_max1": 0.054816975060337905, "AR_max10": 0.46374698310539014, "AR_max100": 0.643543845534996, "AR_small": 0.4110236220472441, "AR_medium": 0.6475798525798526, "AR_large": 0.6609677419354839}, "raw_prediction_bytes": 4270687714, "images": 750, "wall_seconds": 329.1409999999996, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed29_20261104_a02/label100_R_seed29。


## LABEL-CURVE-BATCH 验收：label100_R_seed29

UTC 2026-11-04T03:00:57.434559+00:00。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R=2100/曝光{'R': 24000, 'S': 0}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={"AP50_95": 0.548592284796249, "AP50": 0.9567477711954151, "AP75": 0.5582864111392855, "AP_small": 0.34887948495115484, "AP_medium": 0.5380217304573179, "AP_large": 0.6118205394961417, "AR_max1": 0.054816975060337905, "AR_max10": 0.46374698310539014, "AR_max100": 0.643543845534996, "AR_small": 0.4110236220472441, "AR_medium": 0.6475798525798526, "AR_large": 0.6609677419354839}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据G:\FutsalMOT_Training\runs\main_bbox_yolo11s_soccernet_R_seed29_20261104_a02\completion_verification.json；无best/test/新策略。


## LABEL-CURVE：label100_R_plus_S_seed29 STARTED

UTC 2026-11-04T03:01:24.191637+00:00。公开YOLO11s独立起点seed29，唯一真实2100/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset100_seed17_v1.json", "sha256": "81e363fe89f3b1cda00f967487ad033ee84e00d68654d96ea305ca0680a05f92"}。策略R+S，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选0不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed29_20261104_a01/label100_R_plus_S_seed29。


## LABEL-CURVE：label100_R_plus_S_seed29 COMPLETE

UTC 2026-11-04T03:27:00.093558+00:00。公开YOLO11s独立起点seed29，唯一真实2100/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset100_seed17_v1.json", "sha256": "81e363fe89f3b1cda00f967487ad033ee84e00d68654d96ea305ca0680a05f92"}。策略R+S，1500/1500成功更新，实际曝光{'R': 12000, 'S': 12000}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选0不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 27760, "create_time": 1793761271.4465024, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 12000, "S": 12000}, "schema_version": "bbox_label_curve_v1", "job_id": "label100_R_plus_S_seed29", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_RS_seed29_20261104_a01\\label100_R_plus_S_seed29", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5421217148867459, "AP50": 0.9602809768966277, "AP75": 0.5443835824117951, "AP_small": 0.3631857897172691, "AP_medium": 0.531526646584413, "AP_large": 0.6070751366436327, "AR_max1": 0.0544549477071601, "AR_max10": 0.4578037007240547, "AR_max100": 0.6416130329847144, "AR_small": 0.4224409448818897, "AR_medium": 0.6457985257985259, "AR_large": 0.6559354838709678}, "raw_prediction_bytes": 4266343570, "images": 750, "wall_seconds": 330.3279999999995, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5421217148867459, "AP50": 0.9602809768966277, "AP75": 0.5443835824117951, "AP_small": 0.3631857897172691, "AP_medium": 0.531526646584413, "AP_large": 0.6070751366436327, "AR_max1": 0.0544549477071601, "AR_max10": 0.4578037007240547, "AR_max100": 0.6416130329847144, "AR_small": 0.4224409448818897, "AR_medium": 0.6457985257985259, "AR_large": 0.6559354838709678}, "raw_prediction_bytes": 4266343570, "images": 750, "wall_seconds": 330.3279999999995, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed29_20261104_a01/label100_R_plus_S_seed29。


## LABEL-CURVE-BATCH 验收：label100_R_plus_S_seed29

UTC 2026-11-04T03:27:02.065541+00:00。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R=2100/曝光{'R': 12000, 'S': 12000}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={"AP50_95": 0.5421217148867459, "AP50": 0.9602809768966277, "AP75": 0.5443835824117951, "AP_small": 0.3631857897172691, "AP_medium": 0.531526646584413, "AP_large": 0.6070751366436327, "AR_max1": 0.0544549477071601, "AR_max10": 0.4578037007240547, "AR_max100": 0.6416130329847144, "AR_small": 0.4224409448818897, "AR_medium": 0.6457985257985259, "AR_large": 0.6559354838709678}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据G:\FutsalMOT_Training\runs\main_bbox_yolo11s_soccernet_RS_seed29_20261104_a01\completion_verification.json；无best/test/新策略。


## LABEL-CURVE：label50_R_seed29 STARTED

UTC 2026-11-04T03:27:50.532936+00:00。公开YOLO11s独立起点seed29，唯一真实1050/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset50_seed17_v1.json", "sha256": "fa1b6729f7eb3438157c7abfb91c561062026f11aea40cd016f0da6f807f799b"}。策略R，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1050不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed29_20261104_a03/label50_R_seed29。


## SEG-REAL-DATASET-SEARCH-007

UTC 2026-11-04T03:37:38.312588+00:00。用户要求暂离训练方案，确认正式真实Seg数据或寻找更适合来源。实际完成原两候选状态复核与主来源搜索，未修改EXPERIMENTS/PAPER/DATASET、原GT/分组/代码/活动BBox batch，不开Seg训练/GPU。结论仍无已准入的正式真实Seg库；ihelon512/511唯一RGB与Quantigo847/三video保守一组保持候选，不因数据已下载就称正式。

新优先候选为Acme AI的sadhliroomyprime/football-semantic-segmentation v1：发布API明确100帧、Real Madrid-Manchester United2017欧洲超级杯、原YouTube https://youtu.be/I8RoMceZ7W8、平台CC BY4.0。API原description称按实例标注，导出名称semantic/fused不能单凭名称排除。实际HTTP206只取原COCO首2MiB+末64KiB（全文件153300437B未下载/未全SHA），解析9个完整annotation；首图3个Team A、3个Team B、1个Goalkeeper B各有独立ID/iscrowd0，部分实例2/5连通分片，11类表在尾段直接读出。说明至少抽样确有独立人物实例，不等于整100图全instance质量通过。

仅取并实际查看原图Frame1(14).jpg，1920x1080/677388B/SHA55470303abc3b846e07f2b45dbd68c2898dfacaa7762ec4f927515d985b8b15a；可见RMA0-0MU02:16/门前近景/历史视频水印，与发布说明相容但不以单帧认证整库或确切日期。7个人物mask做一次CPU原COCO光栅，source area分别87773/109237/165465/119477/128396/89788/30230，raster分别86715/108121/164730/119091/126824/88765/29146；各raster框宽高较原框少1px，原值不改。各polygon约2072—3195顶点，现32MiBJSON/1024顶点审计上界不兼容；需独立有界流式/按图派生与原PNG像素约定核验，不能热改旧reader/删点/放宽门限。本源只有单场100图，更适合作来源较清楚的额外比赛评价候选，不直接替代主训练库。尚缺全量覆盖/语义与遮挡核验、跨库查重、真实许可/上游链和正式用途划分。

排查的其他线索：①MV-Soccer CVPRW2024确有人物mask实验，但作者仓库当前dce9e2441ebf31188e1a761ea1e01bc6eb516eb6只核到demo/模型短链，无可下载原GT/原图/比赛split，2311808是三库bbox数不能写mask数；只列作者索取候选，未发送任何消息。②Varun Sharman2025实例分割论文虽称1000+图，第1.5节明确内部未发布/未共享，不列可用公开库。③SoccerSum官方人物为检测类别，分割是场地框区域，不能充人物实例GT。④lifatsastain150帧模型卡明确bbox经SAM生成mask，不充独立人工GT。⑤Voxel51镜像沿Kaggle512样本源，不算新独立比赛。⑥新Kaggle abdurrahimmsai/football-segmentation的README为2999图/640拉伸、data.yaml四类只有0/1/2/3并链接Roboflow q1npj v1，API许可Unknown与yaml CC BY4.0声明不一致，原比赛/标注过程缺失，不因2999更大就升级正式源；只读两README/yaml，无整库下载。

可追溯小源审阅收据 G:\FutsalMOT_Dataset\real\football_seg_public\derived\DATASET_CANDIDATES_RECHECK_20261104.json SHAde55b6c546ad217a8d2f1959405d8311c80532a31d5da75b3ae14e7e9b6ebfe7，schema football_seg_dataset_candidates_review_v1/status NO_FORMAL_REAL_SEG_DATASET_ADMITTED。只在现derived加一份实际来源证据，不建方案或报告树。所有URL/部分HTTP范围/SHA/类别/7mask数值与未满足条件已登记；范围SHA明确不是全GT摘要。主来源参考：Acme Kaggle原页/API；MV-Soccer原CVF论文+作者GitHub；Simula作者论文 https://home.simula.no/~paalh/students/2025-HiOF-VarunSharman.pdf 第1.5节；https://datasets.simula.no/soccersum/；https://huggingface.co/lifatsastain/soccer-player-segmentation；https://huggingface.co/datasets/Voxel51/Football-Player-Segmentation。

后续数据优先级：先核Acme原格式/全部人物质量与比赛关系，继续保留ihelon/Quantigo而不宣称正式；若仍缺可靠多比赛真值，应另行讨论在已有合法足球原图上建设比赛分离的小规模人工mask面板，不自动用SAM充GT或新增标注任务。本轮仅数据调研和极少抽样，不变训练配方。


## LABEL-CURVE：label50_R_seed29 COMPLETE

UTC 2026-11-04T03:52:17.810096+00:00。公开YOLO11s独立起点seed29，唯一真实1050/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset50_seed17_v1.json", "sha256": "fa1b6729f7eb3438157c7abfb91c561062026f11aea40cd016f0da6f807f799b"}。策略R，1500/1500成功更新，实际曝光{'R': 24000, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1050不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 26232, "create_time": 1793762851.227152, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 24000, "S": 0}, "schema_version": "bbox_label_curve_v1", "job_id": "label50_R_seed29", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_R_seed29_20261104_a03\\label50_R_seed29", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5362147753246989, "AP50": 0.9517460562308222, "AP75": 0.5305804553986875, "AP_small": 0.3582310989149963, "AP_medium": 0.52326241275932, "AP_large": 0.6085786535746444, "AR_max1": 0.05499798873692678, "AR_max10": 0.4568584070796461, "AR_max100": 0.6317980691874497, "AR_small": 0.4232283464566929, "AR_medium": 0.6335012285012285, "AR_large": 0.6576774193548387}, "raw_prediction_bytes": 4275289289, "images": 750, "wall_seconds": 336.5, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5362147753246989, "AP50": 0.9517460562308222, "AP75": 0.5305804553986875, "AP_small": 0.3582310989149963, "AP_medium": 0.52326241275932, "AP_large": 0.6085786535746444, "AR_max1": 0.05499798873692678, "AR_max10": 0.4568584070796461, "AR_max100": 0.6317980691874497, "AR_small": 0.4232283464566929, "AR_medium": 0.6335012285012285, "AR_large": 0.6576774193548387}, "raw_prediction_bytes": 4275289289, "images": 750, "wall_seconds": 336.5, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed29_20261104_a03/label50_R_seed29。


## LABEL-CURVE-BATCH 验收：label50_R_seed29

UTC 2026-11-04T03:52:19.194967+00:00。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R=1050/曝光{'R': 24000, 'S': 0}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={"AP50_95": 0.5362147753246989, "AP50": 0.9517460562308222, "AP75": 0.5305804553986875, "AP_small": 0.3582310989149963, "AP_medium": 0.52326241275932, "AP_large": 0.6085786535746444, "AR_max1": 0.05499798873692678, "AR_max10": 0.4568584070796461, "AR_max100": 0.6317980691874497, "AR_small": 0.4232283464566929, "AR_medium": 0.6335012285012285, "AR_large": 0.6576774193548387}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据G:\FutsalMOT_Training\runs\main_bbox_yolo11s_soccernet_R_seed29_20261104_a03\completion_verification.json；无best/test/新策略。


## LABEL-CURVE：label50_R_plus_S_seed29 STARTED

UTC 2026-11-04T03:52:45.642894+00:00。公开YOLO11s独立起点seed29，唯一真实1050/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset50_seed17_v1.json", "sha256": "fa1b6729f7eb3438157c7abfb91c561062026f11aea40cd016f0da6f807f799b"}。策略R+S，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1050不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed29_20261104_a02/label50_R_plus_S_seed29。


## SEG-ACME-FULL-008 / SEG-TRAIN-SOURCE-008

UTC 2026-11-04T04:11:00.937267+00:00。按用户要求完成Acme100帧全量来源/人物GT质量/跨源核验，并收口主训练候选；原GT/EXPERIMENTS/PAPER/DATASET及活动BBox42不改，零Seg模型/优化器/GPU/训练。Acme固定Kaggle sadhliroomyprime/football-semantic-segmentation v1完整取得303源文件219213825B（原COCO153300437B、100JPEG、100savePNG、100fusePNG、原PDF和说明），archive71990776B/SHA16324f1aa4ad1f30526d4b58b1588d99f554dc8942c98cfed1df47d9aa4dde40保留。取得75.641秒、完整CPU审计170.375秒、额外内区差异诊断12.563秒；单CPU低优先级/4GiB余量，未改变BBox执行器或派发任务。

完整COCO实数100图/915标注，其中TeamA192/TeamB210/GKA38/GKB47/Referee24共511人物实例，非人物stuff与ball不冒充人物。全100JPEG和200PNG尺寸/源清单已核，511人物原COCO光栅零空mask、零越界、零严格proper-crossing、960/ratio2/proto采样零消失；这只证明已列几何与采样结果，不是语义标注全PASS。实际查看四页覆盖全部100原图/人物叠加，另放大4/5/43/86和同图1/74，以及ihelon5904/4261上下文；13张临时诊断展示后按精确SHA退休，原303源文件和archive全部保留。

比赛来源：发布说明和原PDF均指2017Real Madrid-Manchester United欧洲超级杯；100帧观测一致RMA/MU、比分和时钟/黑绿皇马红白曼联，无另场明显画面。UEFA官方记录为2017-08-08斯科普里2-1（https://www.uefa.com/uefasupercup/history/2017/）。原YouTube I8RoMceZ7W8的oEmbed返回404/watch不可读，保留这一缺口，不称已删除或编造上传信息。Kaggle平台声明CCBY4.0与上游权利核验区别记录。Acme100对原ihelon512与Quantigo847复用旧指纹共135900对：0低分辨率RGB精确/0pHash<=8，最小Hamming14。结合ihelon已观测NYRB红/浅蓝场景及Quantigo2/3 JUV-RMA2017决赛场景，支持不同观测面板；无重复检出不等于完整比赛独立性证明，ihelon全比赛/日期/原视频和Q所有clip身份仍未全部确证。

明确质量阻塞：①image1 Frame1(14)与image74 Frame1(71)完整decodedRGB相同、两版人物GT不同；原100只有99唯一RGB。整对隔离后仅量化为98图/497人物/895全部标注候选，不选一个版本充自动去重，更不写98图质量通过。②image43/ann380的原COCO门将轮廓包含身前球，source area26641/raster31480，侵入原fuse Ball类别的腐蚀2px内区5314像素；原轮廓仅outer contour，源PNG区分球。此为明确内部孔洞/遮挡表示冲突，不是普通1像素边界差。③image86右缘小手/前臂原COCO与fuse未标为人物，发布材料未给裁边碎片ignore规则。image4/5顶部截断者实际上已标，不能泛称所有裁边漏标。

由此追加全511人物的COCO内区与原fuse类别诊断：175实例/85图出现最大连通差异>=50像素，涵盖球/背景/其他人物；50只是检查排序阈值，不是新增质量界限，也不等于175个人工错误。与已放大案例合看，不能只删重复对或image43就宣称剩余可做严格实例评价。结论Acme当前NOT_ADMITTED_AS_FORMAL_INSTANCE_SEG_EVALUATION：需要原instance-pixel ID/孔洞导出解释及可信RLE（不能猜fuse类色当实例ID），同RGB异GT与裁边范围裁定、所有方法共用固定GT视图。100帧本质一场highlights/88对pHash近邻（含精确对），不作为100独立样本或首次盲测；尚无任何模型评价分数。

主训练优先级明确为ihelon，Quantigo保留备用而非升级正式评价源。已有7条全字段精确annotation重复的alias/RLE血缘保持；同图异GT images321/451整对、tiny3px image384/ann5904、遮挡疑难image287/ann4261共4整图隔离后，准备了508图/7620人物候选UID/annotationID，状态PROPOSED_NOT_ACTIVATED，不删单个人作背景、不修原GT。4261可解释为前排遮挡后的后排可见区域，仅source bbox与mask不同不足以判错；隔离是保守未定，非新人工坏标注结论。原1688 proper-crossing findings不能通过自动改多边形抹除，现RLE只等价保留原COCO像素。ihelon仍一保守源组，不能随机帧拆成独立比赛；确切日期缺失不是唯一阻塞，实际还需共用GT/极小监督规则和独立dev用途冻结。Quantigo4482人物覆盖不完整与5895极小实例尚待裁定，三视频不当三场独立比赛。

收据：{"SOURCE_RECEIPT.json": {"path": "G:/FutsalMOT_Dataset/real/football_seg_public/acme_football_semantic_v1/SOURCE_RECEIPT.json", "bytes": 106877, "sha256": "36c9e87c790a352c30ac292229c19ae768e0b88fa43ea76a2805460c3d14f9f7"}, "FULL_GT_AUDIT_20261104_A01.json": {"path": "G:/FutsalMOT_Dataset/real/football_seg_public/acme_football_semantic_v1/FULL_GT_AUDIT_20261104_A01.json", "bytes": 1197744, "sha256": "12d5370835e6582af112593bd1f00decb631156d5ad75f3c5dad028987e0c9d9"}, "CPU_FINGERPRINTS_20261104_A01.json": {"path": "G:/FutsalMOT_Dataset/real/football_seg_public/acme_football_semantic_v1/CPU_FINGERPRINTS_20261104_A01.json", "bytes": 36819, "sha256": "fe869af9fbaeb6040ea8c6eb9e66821f47bb67703188baa9f0b1101ec3aebb2a"}, "COCO_FUSE_INTERIOR_DIAGNOSTIC_20261104.json": {"path": "G:/FutsalMOT_Dataset/real/football_seg_public/acme_football_semantic_v1/COCO_FUSE_INTERIOR_DIAGNOSTIC_20261104.json", "bytes": 202069, "sha256": "346d6f3e150163feb6c25b50dc50aabe8631f681a09480714f9db6aa2b95e00c"}, "QUALITY_REVIEW_20261104.json": {"path": "G:/FutsalMOT_Dataset/real/football_seg_public/acme_football_semantic_v1/QUALITY_REVIEW_20261104.json", "bytes": 16204, "sha256": "97ad2c43d61e542082216fe039a24bed7200173fe910abbee103432f258cb66d"}, "REVIEW_VISUAL_CLEANUP_20261104.json": {"path": "G:/FutsalMOT_Dataset/real/football_seg_public/acme_football_semantic_v1/REVIEW_VISUAL_CLEANUP_20261104.json", "bytes": 4136, "sha256": "0805d6ddb8b0d25549c20ba4c846ce1132bb2827fcda84b61f7b1d7f4c4b9dec"}, "decision": {"path": "G:/FutsalMOT_Dataset/real/football_seg_public/derived/REAL_SEG_DATASET_DECISION_20261104.json", "bytes": 216270, "sha256": "49f933c2e0251af70c4660d79ac19f4e48300d389fab7decdc14da997d9d5151"}}

正式真实训练/评价数据仍未准入；本轮完成的是完整数据核验与明确候选处置。下一数据步骤限于可信像素实例导出/疑难GT裁定和共同视图，不自动进行人工重标、改变训练方案或启动Seg。原EXPERIMENTS32c941.../PAPERd48cb.../DATASET4ca512...前后核对保持。


## LABEL-CURVE：label50_R_plus_S_seed29 COMPLETE

UTC 2026-11-04T04:18:41.847232+00:00。公开YOLO11s独立起点seed29，唯一真实1050/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset50_seed17_v1.json", "sha256": "fa1b6729f7eb3438157c7abfb91c561062026f11aea40cd016f0da6f807f799b"}。策略R+S，1500/1500成功更新，实际曝光{'R': 12000, 'S': 12000}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1050不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 7984, "create_time": 1793764352.9550292, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 12000, "S": 12000}, "schema_version": "bbox_label_curve_v1", "job_id": "label50_R_plus_S_seed29", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_RS_seed29_20261104_a02\\label50_R_plus_S_seed29", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5354680179560748, "AP50": 0.9589961333980098, "AP75": 0.528039535953075, "AP_small": 0.33204264941794454, "AP_medium": 0.5240002686728545, "AP_large": 0.610425809044441, "AR_max1": 0.05339903459372486, "AR_max10": 0.4562751407884151, "AR_max100": 0.6343121480289622, "AR_small": 0.39448818897637794, "AR_medium": 0.6369410319410319, "AR_large": 0.6603870967741935}, "raw_prediction_bytes": 4264238750, "images": 750, "wall_seconds": 336.4529999999977, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5354680179560748, "AP50": 0.9589961333980098, "AP75": 0.528039535953075, "AP_small": 0.33204264941794454, "AP_medium": 0.5240002686728545, "AP_large": 0.610425809044441, "AR_max1": 0.05339903459372486, "AR_max10": 0.4562751407884151, "AR_max100": 0.6343121480289622, "AR_small": 0.39448818897637794, "AR_medium": 0.6369410319410319, "AR_large": 0.6603870967741935}, "raw_prediction_bytes": 4264238750, "images": 750, "wall_seconds": 336.4529999999977, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed29_20261104_a02/label50_R_plus_S_seed29。


## LABEL-CURVE-BATCH 验收：label50_R_plus_S_seed29

UTC 2026-11-04T04:18:43.349881+00:00。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R=1050/曝光{'R': 12000, 'S': 12000}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={"AP50_95": 0.5354680179560748, "AP50": 0.9589961333980098, "AP75": 0.528039535953075, "AP_small": 0.33204264941794454, "AP_medium": 0.5240002686728545, "AP_large": 0.610425809044441, "AR_max1": 0.05339903459372486, "AR_max10": 0.4562751407884151, "AR_max100": 0.6343121480289622, "AR_small": 0.39448818897637794, "AR_medium": 0.6369410319410319, "AR_large": 0.6603870967741935}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据G:\FutsalMOT_Training\runs\main_bbox_yolo11s_soccernet_RS_seed29_20261104_a02\completion_verification.json；无best/test/新策略。


## LABEL-CURVE：label25_R_seed29 STARTED

UTC 2026-11-04T04:19:26.321039+00:00。公开YOLO11s独立起点seed29，唯一真实525/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset25_seed17_v1.json", "sha256": "bdac16a1dc34b8ebdd9cc8f9a5c6cc98b1844bacbec4af90b25ffaebe7b95bf0"}。策略R，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1575不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed29_20261104_a04/label25_R_seed29。


## SEG-PNG-RLE-009 / IHELON-COMMON-VIEW-009

UTC 2026-11-04T04:43:54.493846+00:00。用户直接要求继续核Acme原PNG实例ID/孔洞并尝试无损RLE，补齐ihelon候选，保留GT且不启动Seg。本轮只CPU数据准备，无模型/优化器/CUDA/训练，原EXPERIMENTS32c941/PAPERd48cb/DATASET4ca512、Seg后端a3e8与BBox42 c6eaf原SHA前后保持，不向方案追加对话。

官方SuperAnnotate SDK固定commit e8331413b5a99013350c59fe7a45172dce1b890b，common.py blue_color_generator以RGB整数15*(index+1)生成像素part色；官方tests/data_set/pixel_with_holes/1.jpg___pixel.json明确同一实例可含多个part、同类可有不同实例。Acme303源文件缺该instance-to-parts JSON，不能一色一人物。实际全19071个save颜色项中65透明项、非透明编码零不符合15倍数，且每part在fuse上为单一类别。709个图像×类别的source area总和与fuse像素量全部exact。SDK仅只读来源，没有安装/执行外部代码，来源commit/文件Gitblob/URLs已存Acme metadata/SDK_PNG_ENCODING_EVIDENCE_20261104.json。

新独立segmentation_png_rle.py与7项针对CPU检查PASS/0skip0.010秒，覆盖RGBA/alpha全像素往返、孔洞、多part、重叠/混类/歧义/非整数/原面积错误拒绝。先以全part唯一同类COCO外壳包含关系+原面积/原bbox/填满外轮廓逐值exact恢复，明确不用最近邻、IoU容差、删点/桥接或模型。100save+100fuse共200PNG均全RGBA逐像素RLE往返exact，保留全部透明像素/原色号；全类别73帧/647标注严格重建通过，其他27帧未改判。该独立派生目录png_rle_candidate_20261104_a01，398.750秒，原GT不动。

进一步单列人物范围避免背景part问题混入：83帧/406人物满足同样全部exact条件，17帧仍未唯一恢复；原像素RLE全在，不用这些局部未定项生成假实例。person_rle_candidate_20261104_a01实际229.281秒。针对17帧再做一次逐像素同类COCO壳唯一归属的无阈值补充，2.391秒/新增exact0；多数为1—5个源PNG孤立像素未被COCO轮廓覆盖，部分有同类遮挡像素多个owner，保留每帧missing/ambiguous真数，不用1px容差伪称exact。初步inline诊断曾因NumPy2 inverse未展平报objecttoo deep，修正ravel后运行；零模型/零GT写入的准备错误单列。

原图43/ann380实际放大四栏比较：原COCO光栅31480px含身前球，恢复PNG-RLE26641px与原source area精确相同，球心xy50,737由原mask true变为RLE false，球洞和手臂间背景孔洞真实保留。这是取回原PNG像素、不是人工修标，也不冒称与原COCO光栅逐像素一致。已展示174686B诊断PNG/SHA73ff0b29d8a77476820cd1cb57832401b58e2e3093e37612d8002c3aabd4aaf6，按精确路径/SHA退休，原图/PNG/COCO全部保留。孔洞问题因此可部分解决，但不能宣称100帧实例恢复全通过。

Acme把83严格帧中的同RGB异GT1/74和已知右边缘漏手86再整图hold，已生成eval80_candidate_20261104_a01：80图/383实例、原categoryID/area/bbox/iscrowd等非seg字段保持，独立LEDGER绑定原polygon、PNG part code和恢复RLE，所有20个排除frame及理由明列，不按模型分数选图。候选仍training_enabled=false/activated_split=null/formal_eval=false；17帧原关系JSON仍缺，不能完整修复。80帧是同一场已观测highlights，质量筛选会改变遮挡/难度分布；需先固定共同视图/目标/裁边/遮挡与dev或confirmation用途，不能一份数据两用或称首次盲测，未接实际模型评价。

主训练ihelon508图/7620人物的完整共同RLE文件/UID/原GT血缘已实际准备。用原segmentation_data.prepare_sample、identity affine/no flip、960/ratio2和真实RLE，逐图检查监督覆盖；RGB仅零数组占位因mask几何独立，未重新解码原RGB或重做训练CPU预检。75.422秒：54目标被现候选过滤（wh<=2为53，ratio2空12，geometry空1，理由重叠），另有52个通过过滤的目标在proto240 nearest后为空，共82张图受影响。原508/7620全部保留，不能把被删目标当正常背景后称监督完整。

另生成identity_safe_instances.json及IDENTITY_SAFE_MANIFEST.json：把上述82张整图暂hold后为426图/6107人物，引用原覆盖结果，静态identity下过滤0/proto消失0，不重跑、不删个别标注、不改原area/bbox。这是供最小探针的可监督候选，不静默替换主训练池；小目标/遮挡分布会变化，426与508都未激活。随机仿射仍可能裁边/消失，实际ignore/过滤/整图拒绝规则与所有方法共用GT视图、比赛/源级开发/确认用途需冻结；原ihelon一保守组不能帧随机拆独立比赛，原exactmatch/date/video字段空不补造。

实际已补条件为：共同数据文件/UID与完整血缘、疑难整图处置清单、所有7620实例当前几何监督覆盖、保留孔洞的Acme像素RLE与80图候选。仍未闭合的是共同GT/角色与裁边/随机几何监督规则、正式来源级用途；执行方面合成1600实际UID、完整500步提交/STOP/恢复/EMA/native预测/partial/owned流水线、公开s/960真实GPU/资源未准入。本次不启动Seg，也不借数据转换悄改输入/ratio/更新或评价协议。

总收据 {"path": "G:/FutsalMOT_Dataset/real/football_seg_public/derived/SEG_PNG_RLE_READINESS_20261104.json", "bytes": 15049, "sha256": "836a05b92ea112c55fe50946a8c5eed1f4b7307e0730188f3a75da818ff64cd3"}
分项 {"primary_encoding": {"path": "G:/FutsalMOT_Dataset/real/football_seg_public/acme_football_semantic_v1/metadata/SDK_PNG_ENCODING_EVIDENCE_20261104.json", "bytes": 1837, "sha256": "3f21d819e5d119844b26186598894d615bd5cbbc3e0fbde7930ad6e553357877"}, "all_palette_RLE": {"path": "G:/FutsalMOT_Dataset/real/football_seg_public/acme_football_semantic_v1/png_rle_candidate_20261104_a01/SUMMARY.json", "bytes": 57409, "sha256": "0b1c693bd2e64eb00b7473f49224275c4c9513c58a37e71c3c4f6b6536d35bb0"}, "person_recovery": {"path": "G:/FutsalMOT_Dataset/real/football_seg_public/acme_football_semantic_v1/person_rle_candidate_20261104_a01/SUMMARY.json", "bytes": 49125, "sha256": "30c2a3c390c10fcfeef4f68614147058f1ea51522e87b0db198af2017309799c"}, "pixel_unique_complement": {"path": "G:/FutsalMOT_Dataset/real/football_seg_public/acme_football_semantic_v1/PIXEL_UNIQUE_COMPLEMENT_20261104.json", "bytes": 3536, "sha256": "be6b5b7f460e29343143fe793f2e3beff517c304d7b20e26fff40158d68eb20c"}, "Acme80": {"path": "G:/FutsalMOT_Dataset/real/football_seg_public/acme_football_semantic_v1/eval80_candidate_20261104_a01/MANIFEST.json", "bytes": 5583, "sha256": "b72e1c8519c37a80c607b854588af14cbd799887a00ead60d768299e1e0c2e30"}, "ihelon508": {"path": "G:/FutsalMOT_Dataset/real/football_seg_public/derived/ihelon_train_candidate_20261104_a01/READINESS.json", "bytes": 872027, "sha256": "072bf3d3243bf7508358ec13ef6c96bb77409be2e1f3c2d6bc0a1078554a95e6"}, "ihelon426": {"path": "G:/FutsalMOT_Dataset/real/football_seg_public/derived/ihelon_train_candidate_20261104_a01/IDENTITY_SAFE_MANIFEST.json", "bytes": 151090, "sha256": "dd70a949ec7ae5ae6d7ac0b0b1e969e56de47b10659924fda79c252436fd1ed2"}}


## LABEL-CURVE：label25_R_seed29 COMPLETE

UTC 2026-11-04T04:44:37.172341+00:00。公开YOLO11s独立起点seed29，唯一真实525/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset25_seed17_v1.json", "sha256": "bdac16a1dc34b8ebdd9cc8f9a5c6cc98b1844bacbec4af90b25ffaebe7b95bf0"}。策略R，1500/1500成功更新，实际曝光{'R': 24000, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1575不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 23316, "create_time": 1793765953.3759778, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 24000, "S": 0}, "schema_version": "bbox_label_curve_v1", "job_id": "label25_R_seed29", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_R_seed29_20261104_a04\\label25_R_seed29", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.515909088822543, "AP50": 0.9368695287882455, "AP75": 0.5005343658912256, "AP_small": 0.2977955754896677, "AP_medium": 0.5048478906184792, "AP_large": 0.5863068989883495, "AR_max1": 0.053569991954947695, "AR_max10": 0.44663113435237334, "AR_max100": 0.6092819790828641, "AR_small": 0.3476377952755906, "AR_medium": 0.6121744471744471, "AR_large": 0.6382580645161291}, "raw_prediction_bytes": 4274343362, "images": 750, "wall_seconds": 349.4850000000006, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.515909088822543, "AP50": 0.9368695287882455, "AP75": 0.5005343658912256, "AP_small": 0.2977955754896677, "AP_medium": 0.5048478906184792, "AP_large": 0.5863068989883495, "AR_max1": 0.053569991954947695, "AR_max10": 0.44663113435237334, "AR_max100": 0.6092819790828641, "AR_small": 0.3476377952755906, "AR_medium": 0.6121744471744471, "AR_large": 0.6382580645161291}, "raw_prediction_bytes": 4274343362, "images": 750, "wall_seconds": 349.4850000000006, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed29_20261104_a04/label25_R_seed29。


## LABEL-CURVE-BATCH 验收：label25_R_seed29

UTC 2026-11-04T04:44:38.607227+00:00。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R=525/曝光{'R': 24000, 'S': 0}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={"AP50_95": 0.515909088822543, "AP50": 0.9368695287882455, "AP75": 0.5005343658912256, "AP_small": 0.2977955754896677, "AP_medium": 0.5048478906184792, "AP_large": 0.5863068989883495, "AR_max1": 0.053569991954947695, "AR_max10": 0.44663113435237334, "AR_max100": 0.6092819790828641, "AR_small": 0.3476377952755906, "AR_medium": 0.6121744471744471, "AR_large": 0.6382580645161291}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据G:\FutsalMOT_Training\runs\main_bbox_yolo11s_soccernet_R_seed29_20261104_a04\completion_verification.json；无best/test/新策略。


## LABEL-CURVE：label25_R_plus_S_seed29 STARTED

UTC 2026-11-04T04:45:05.283031+00:00。公开YOLO11s独立起点seed29，唯一真实525/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset25_seed17_v1.json", "sha256": "bdac16a1dc34b8ebdd9cc8f9a5c6cc98b1844bacbec4af90b25ffaebe7b95bf0"}。策略R+S，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1575不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed29_20261104_a03/label25_R_plus_S_seed29。


## LABEL-CURVE：label25_R_plus_S_seed29 COMPLETE

UTC 2026-11-04T05:11:15.720318+00:00。公开YOLO11s独立起点seed29，唯一真实525/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset25_seed17_v1.json", "sha256": "bdac16a1dc34b8ebdd9cc8f9a5c6cc98b1844bacbec4af90b25ffaebe7b95bf0"}。策略R+S，1500/1500成功更新，实际曝光{'R': 12000, 'S': 12000}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1575不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 23056, "create_time": 1793767492.5212843, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 12000, "S": 12000}, "schema_version": "bbox_label_curve_v1", "job_id": "label25_R_plus_S_seed29", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_RS_seed29_20261104_a03\\label25_R_plus_S_seed29", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5332803978434328, "AP50": 0.9526158124092253, "AP75": 0.5301506545991193, "AP_small": 0.3667150300010519, "AP_medium": 0.5223430853214251, "AP_large": 0.6010733354439503, "AR_max1": 0.05319790828640386, "AR_max10": 0.4542839903459373, "AR_max100": 0.627815768302494, "AR_small": 0.40393700787401576, "AR_medium": 0.6309213759213759, "AR_large": 0.6486451612903225}, "raw_prediction_bytes": 4268819783, "images": 750, "wall_seconds": 352.34400000000096, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5332803978434328, "AP50": 0.9526158124092253, "AP75": 0.5301506545991193, "AP_small": 0.3667150300010519, "AP_medium": 0.5223430853214251, "AP_large": 0.6010733354439503, "AR_max1": 0.05319790828640386, "AR_max10": 0.4542839903459373, "AR_max100": 0.627815768302494, "AR_small": 0.40393700787401576, "AR_medium": 0.6309213759213759, "AR_large": 0.6486451612903225}, "raw_prediction_bytes": 4268819783, "images": 750, "wall_seconds": 352.34400000000096, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed29_20261104_a03/label25_R_plus_S_seed29。


## LABEL-CURVE-BATCH 验收：label25_R_plus_S_seed29

UTC 2026-11-04T05:11:17.302746+00:00。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R=525/曝光{'R': 12000, 'S': 12000}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={"AP50_95": 0.5332803978434328, "AP50": 0.9526158124092253, "AP75": 0.5301506545991193, "AP_small": 0.3667150300010519, "AP_medium": 0.5223430853214251, "AP_large": 0.6010733354439503, "AR_max1": 0.05319790828640386, "AR_max10": 0.4542839903459373, "AR_max100": 0.627815768302494, "AR_small": 0.40393700787401576, "AR_medium": 0.6309213759213759, "AR_large": 0.6486451612903225}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据G:\FutsalMOT_Training\runs\main_bbox_yolo11s_soccernet_RS_seed29_20261104_a03\completion_verification.json；无best/test/新策略。


## LABEL-CURVE：label10_R_seed29 STARTED

UTC 2026-11-04T05:11:59.946263+00:00。公开YOLO11s独立起点seed29，唯一真实210/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset10_seed17_v1.json", "sha256": "fccc2c998d3b6a6dd4316db0f2a66d97aa5f02571208df360b58911d611245b1"}。策略R，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1890不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed29_20261104_a05/label10_R_seed29。


## LABEL-CURVE：label10_R_seed29 COMPLETE

UTC 2026-11-04T05:37:17.200477+00:00。公开YOLO11s独立起点seed29，唯一真实210/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset10_seed17_v1.json", "sha256": "fccc2c998d3b6a6dd4316db0f2a66d97aa5f02571208df360b58911d611245b1"}。策略R，1500/1500成功更新，实际曝光{'R': 24000, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1890不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 13460, "create_time": 1793769107.3964114, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 24000, "S": 0}, "schema_version": "bbox_label_curve_v1", "job_id": "label10_R_seed29", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_R_seed29_20261104_a05\\label10_R_seed29", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.4831380807984544, "AP50": 0.9189879320335161, "AP75": 0.44796125392354363, "AP_small": 0.3011853655406133, "AP_medium": 0.46958779696846425, "AP_large": 0.5594848458261333, "AR_max1": 0.05121681415929204, "AR_max10": 0.43278358809332274, "AR_max100": 0.5732099758648431, "AR_small": 0.33503937007874013, "AR_medium": 0.5727764127764127, "AR_large": 0.6152903225806452}, "raw_prediction_bytes": 4269552176, "images": 750, "wall_seconds": 353.0619999999981, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.4831380807984544, "AP50": 0.9189879320335161, "AP75": 0.44796125392354363, "AP_small": 0.3011853655406133, "AP_medium": 0.46958779696846425, "AP_large": 0.5594848458261333, "AR_max1": 0.05121681415929204, "AR_max10": 0.43278358809332274, "AR_max100": 0.5732099758648431, "AR_small": 0.33503937007874013, "AR_medium": 0.5727764127764127, "AR_large": 0.6152903225806452}, "raw_prediction_bytes": 4269552176, "images": 750, "wall_seconds": 353.0619999999981, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed29_20261104_a05/label10_R_seed29。


## LABEL-CURVE-BATCH 验收：label10_R_seed29

UTC 2026-11-04T05:37:18.578107+00:00。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R=210/曝光{'R': 24000, 'S': 0}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={"AP50_95": 0.4831380807984544, "AP50": 0.9189879320335161, "AP75": 0.44796125392354363, "AP_small": 0.3011853655406133, "AP_medium": 0.46958779696846425, "AP_large": 0.5594848458261333, "AR_max1": 0.05121681415929204, "AR_max10": 0.43278358809332274, "AR_max100": 0.5732099758648431, "AR_small": 0.33503937007874013, "AR_medium": 0.5727764127764127, "AR_large": 0.6152903225806452}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据G:\FutsalMOT_Training\runs\main_bbox_yolo11s_soccernet_R_seed29_20261104_a05\completion_verification.json；无best/test/新策略。


## LABEL-CURVE：label10_R_plus_S_seed29 STARTED

UTC 2026-11-04T05:37:45.013550+00:00。公开YOLO11s独立起点seed29，唯一真实210/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset10_seed17_v1.json", "sha256": "fccc2c998d3b6a6dd4316db0f2a66d97aa5f02571208df360b58911d611245b1"}。策略R+S，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1890不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed29_20261104_a04/label10_R_plus_S_seed29。


## LABEL-CURVE：label10_R_plus_S_seed29 COMPLETE

UTC 2026-11-04T06:03:55.384145+00:00。公开YOLO11s独立起点seed29，唯一真实210/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset10_seed17_v1.json", "sha256": "fccc2c998d3b6a6dd4316db0f2a66d97aa5f02571208df360b58911d611245b1"}。策略R+S，1500/1500成功更新，实际曝光{'R': 12000, 'S': 12000}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1890不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 15300, "create_time": 1793770652.3460443, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 12000, "S": 12000}, "schema_version": "bbox_label_curve_v1", "job_id": "label10_R_plus_S_seed29", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_RS_seed29_20261104_a04\\label10_R_plus_S_seed29", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5066003565208544, "AP50": 0.9362577906739905, "AP75": 0.4859606818190443, "AP_small": 0.330829044365063, "AP_medium": 0.4940447206752295, "AP_large": 0.5902608823100529, "AR_max1": 0.05242357200321801, "AR_max10": 0.4470635559131134, "AR_max100": 0.5968222043443283, "AR_small": 0.362992125984252, "AR_medium": 0.5944348894348894, "AR_large": 0.648516129032258}, "raw_prediction_bytes": 4263683704, "images": 750, "wall_seconds": 353.9220000000023, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5066003565208544, "AP50": 0.9362577906739905, "AP75": 0.4859606818190443, "AP_small": 0.330829044365063, "AP_medium": 0.4940447206752295, "AP_large": 0.5902608823100529, "AR_max1": 0.05242357200321801, "AR_max10": 0.4470635559131134, "AR_max100": 0.5968222043443283, "AR_small": 0.362992125984252, "AR_medium": 0.5944348894348894, "AR_large": 0.648516129032258}, "raw_prediction_bytes": 4263683704, "images": 750, "wall_seconds": 353.9220000000023, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed29_20261104_a04/label10_R_plus_S_seed29。


## LABEL-CURVE-BATCH 验收：label10_R_plus_S_seed29

UTC 2026-11-04T06:03:57.052875+00:00。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R=210/曝光{'R': 12000, 'S': 12000}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={"AP50_95": 0.5066003565208544, "AP50": 0.9362577906739905, "AP75": 0.4859606818190443, "AP_small": 0.330829044365063, "AP_medium": 0.4940447206752295, "AP_large": 0.5902608823100529, "AR_max1": 0.05242357200321801, "AR_max10": 0.4470635559131134, "AR_max100": 0.5968222043443283, "AR_small": 0.362992125984252, "AR_medium": 0.5944348894348894, "AR_large": 0.648516129032258}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据G:\FutsalMOT_Training\runs\main_bbox_yolo11s_soccernet_RS_seed29_20261104_a04\completion_verification.json；无best/test/新策略。


## LABEL-CURVE：label100_R_seed43 STARTED

UTC 2026-11-04T06:04:39.997907+00:00。公开YOLO11s独立起点seed43，唯一真实2100/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset100_seed17_v1.json", "sha256": "81e363fe89f3b1cda00f967487ad033ee84e00d68654d96ea305ca0680a05f92"}。策略R，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选0不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed43_20261104_a01/label100_R_seed43。


## LABEL-CURVE：label100_R_seed43 COMPLETE

UTC 2026-11-04T06:30:06.334323+00:00。公开YOLO11s独立起点seed43，唯一真实2100/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset100_seed17_v1.json", "sha256": "81e363fe89f3b1cda00f967487ad033ee84e00d68654d96ea305ca0680a05f92"}。策略R，1500/1500成功更新，实际曝光{'R': 24000, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选0不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 22600, "create_time": 1793772267.1908054, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 24000, "S": 0}, "schema_version": "bbox_label_curve_v1", "job_id": "label100_R_seed43", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_R_seed43_20261104_a01\\label100_R_seed43", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5548304304623788, "AP50": 0.9542435597028338, "AP75": 0.577310722160515, "AP_small": 0.3749476411350021, "AP_medium": 0.5452225862087758, "AP_large": 0.6205180484187872, "AR_max1": 0.05559131134352373, "AR_max10": 0.46645213193885765, "AR_max100": 0.64635961383749, "AR_small": 0.42598425196850387, "AR_medium": 0.648931203931204, "AR_large": 0.6694838709677419}, "raw_prediction_bytes": 4276551739, "images": 750, "wall_seconds": 360.48399999999674, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5548304304623788, "AP50": 0.9542435597028338, "AP75": 0.577310722160515, "AP_small": 0.3749476411350021, "AP_medium": 0.5452225862087758, "AP_large": 0.6205180484187872, "AR_max1": 0.05559131134352373, "AR_max10": 0.46645213193885765, "AR_max100": 0.64635961383749, "AR_small": 0.42598425196850387, "AR_medium": 0.648931203931204, "AR_large": 0.6694838709677419}, "raw_prediction_bytes": 4276551739, "images": 750, "wall_seconds": 360.48399999999674, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed43_20261104_a01/label100_R_seed43。


## LABEL-CURVE-BATCH 验收：label100_R_seed43

UTC 2026-11-04T06:30:07.716330+00:00。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R=2100/曝光{'R': 24000, 'S': 0}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={"AP50_95": 0.5548304304623788, "AP50": 0.9542435597028338, "AP75": 0.577310722160515, "AP_small": 0.3749476411350021, "AP_medium": 0.5452225862087758, "AP_large": 0.6205180484187872, "AR_max1": 0.05559131134352373, "AR_max10": 0.46645213193885765, "AR_max100": 0.64635961383749, "AR_small": 0.42598425196850387, "AR_medium": 0.648931203931204, "AR_large": 0.6694838709677419}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据G:\FutsalMOT_Training\runs\main_bbox_yolo11s_soccernet_R_seed43_20261104_a01\completion_verification.json；无best/test/新策略。


## LABEL-CURVE：label100_R_plus_S_seed43 STARTED

UTC 2026-11-04T06:30:35.324075+00:00。公开YOLO11s独立起点seed43，唯一真实2100/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset100_seed17_v1.json", "sha256": "81e363fe89f3b1cda00f967487ad033ee84e00d68654d96ea305ca0680a05f92"}。策略R+S，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选0不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed43_20261104_a01/label100_R_plus_S_seed43。


## LABEL-CURVE：label100_R_plus_S_seed43 COMPLETE

UTC 2026-11-04T06:58:03.557796+00:00。公开YOLO11s独立起点seed43，唯一真实2100/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset100_seed17_v1.json", "sha256": "81e363fe89f3b1cda00f967487ad033ee84e00d68654d96ea305ca0680a05f92"}。策略R+S，1500/1500成功更新，实际曝光{'R': 12000, 'S': 12000}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选0不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 32756, "create_time": 1793773822.4144318, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 12000, "S": 12000}, "schema_version": "bbox_label_curve_v1", "job_id": "label100_R_plus_S_seed43", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_RS_seed43_20261104_a01\\label100_R_plus_S_seed43", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5449718072989591, "AP50": 0.9611659010621143, "AP75": 0.5472771242107806, "AP_small": 0.37888527739834893, "AP_medium": 0.5344701275107009, "AP_large": 0.607649147671987, "AR_max1": 0.055148833467417537, "AR_max10": 0.4597445695897024, "AR_max100": 0.6422767497988737, "AR_small": 0.42559055118110234, "AR_medium": 0.6460319410319411, "AR_large": 0.6583870967741936}, "raw_prediction_bytes": 4262855582, "images": 750, "wall_seconds": 359.59400000000096, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5449718072989591, "AP50": 0.9611659010621143, "AP75": 0.5472771242107806, "AP_small": 0.37888527739834893, "AP_medium": 0.5344701275107009, "AP_large": 0.607649147671987, "AR_max1": 0.055148833467417537, "AR_max10": 0.4597445695897024, "AR_max100": 0.6422767497988737, "AR_small": 0.42559055118110234, "AR_medium": 0.6460319410319411, "AR_large": 0.6583870967741936}, "raw_prediction_bytes": 4262855582, "images": 750, "wall_seconds": 359.59400000000096, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed43_20261104_a01/label100_R_plus_S_seed43。


## LABEL-CURVE-BATCH 验收：label100_R_plus_S_seed43

UTC 2026-11-04T06:58:05.246365+00:00。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R=2100/曝光{'R': 12000, 'S': 12000}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={"AP50_95": 0.5449718072989591, "AP50": 0.9611659010621143, "AP75": 0.5472771242107806, "AP_small": 0.37888527739834893, "AP_medium": 0.5344701275107009, "AP_large": 0.607649147671987, "AR_max1": 0.055148833467417537, "AR_max10": 0.4597445695897024, "AR_max100": 0.6422767497988737, "AR_small": 0.42559055118110234, "AR_medium": 0.6460319410319411, "AR_large": 0.6583870967741936}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据G:\FutsalMOT_Training\runs\main_bbox_yolo11s_soccernet_RS_seed43_20261104_a01\completion_verification.json；无best/test/新策略。


## LABEL-CURVE：label50_R_seed43 STARTED

UTC 2026-11-04T06:58:48.260868+00:00。公开YOLO11s独立起点seed43，唯一真实1050/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset50_seed17_v1.json", "sha256": "fa1b6729f7eb3438157c7abfb91c561062026f11aea40cd016f0da6f807f799b"}。策略R，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1050不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed43_20261104_a02/label50_R_seed43。


## LABEL-CURVE：label50_R_seed43 COMPLETE

UTC 2026-11-04T07:25:13.215047+00:00。公开YOLO11s独立起点seed43，唯一真实1050/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset50_seed17_v1.json", "sha256": "fa1b6729f7eb3438157c7abfb91c561062026f11aea40cd016f0da6f807f799b"}。策略R，1500/1500成功更新，实际曝光{'R': 24000, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1050不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 14888, "create_time": 1793775515.0407577, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 24000, "S": 0}, "schema_version": "bbox_label_curve_v1", "job_id": "label50_R_seed43", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_R_seed43_20261104_a02\\label50_R_seed43", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5421815337435989, "AP50": 0.9523656510477245, "AP75": 0.5454367934941959, "AP_small": 0.32203032072915283, "AP_medium": 0.5323015769290727, "AP_large": 0.6096363054875288, "AR_max1": 0.05496781979082864, "AR_max10": 0.4602373290426388, "AR_max100": 0.6355691874497185, "AR_small": 0.39566929133858264, "AR_medium": 0.6385626535626536, "AR_large": 0.6595483870967741}, "raw_prediction_bytes": 4273923398, "images": 750, "wall_seconds": 353.7029999999977, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5421815337435989, "AP50": 0.9523656510477245, "AP75": 0.5454367934941959, "AP_small": 0.32203032072915283, "AP_medium": 0.5323015769290727, "AP_large": 0.6096363054875288, "AR_max1": 0.05496781979082864, "AR_max10": 0.4602373290426388, "AR_max100": 0.6355691874497185, "AR_small": 0.39566929133858264, "AR_medium": 0.6385626535626536, "AR_large": 0.6595483870967741}, "raw_prediction_bytes": 4273923398, "images": 750, "wall_seconds": 353.7029999999977, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed43_20261104_a02/label50_R_seed43。


## LABEL-CURVE-BATCH 验收：label50_R_seed43

UTC 2026-11-04T07:25:14.694072+00:00。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R=1050/曝光{'R': 24000, 'S': 0}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={"AP50_95": 0.5421815337435989, "AP50": 0.9523656510477245, "AP75": 0.5454367934941959, "AP_small": 0.32203032072915283, "AP_medium": 0.5323015769290727, "AP_large": 0.6096363054875288, "AR_max1": 0.05496781979082864, "AR_max10": 0.4602373290426388, "AR_max100": 0.6355691874497185, "AR_small": 0.39566929133858264, "AR_medium": 0.6385626535626536, "AR_large": 0.6595483870967741}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据G:\FutsalMOT_Training\runs\main_bbox_yolo11s_soccernet_R_seed43_20261104_a02\completion_verification.json；无best/test/新策略。


## LABEL-CURVE：label50_R_plus_S_seed43 STARTED

UTC 2026-11-04T07:25:42.233941+00:00。公开YOLO11s独立起点seed43，唯一真实1050/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset50_seed17_v1.json", "sha256": "fa1b6729f7eb3438157c7abfb91c561062026f11aea40cd016f0da6f807f799b"}。策略R+S，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1050不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed43_20261104_a02/label50_R_plus_S_seed43。


## LABEL-CURVE：label50_R_plus_S_seed43 COMPLETE

UTC 2026-11-04T07:52:21.168431+00:00。公开YOLO11s独立起点seed43，唯一真实1050/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset50_seed17_v1.json", "sha256": "fa1b6729f7eb3438157c7abfb91c561062026f11aea40cd016f0da6f807f799b"}。策略R+S，1500/1500成功更新，实际曝光{'R': 12000, 'S': 12000}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1050不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 2576, "create_time": 1793777128.8982255, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 12000, "S": 12000}, "schema_version": "bbox_label_curve_v1", "job_id": "label50_R_plus_S_seed43", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_RS_seed43_20261104_a02\\label50_R_plus_S_seed43", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5447330806964767, "AP50": 0.9615876372358897, "AP75": 0.5506419372720406, "AP_small": 0.3677742223038705, "AP_medium": 0.5329086038875562, "AP_large": 0.6165896595634737, "AR_max1": 0.0542538213998391, "AR_max10": 0.46037811745776347, "AR_max100": 0.6392598551890587, "AR_small": 0.41338582677165353, "AR_medium": 0.6414496314496315, "AR_large": 0.6655483870967742}, "raw_prediction_bytes": 4267320573, "images": 750, "wall_seconds": 353.53199999999924, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5447330806964767, "AP50": 0.9615876372358897, "AP75": 0.5506419372720406, "AP_small": 0.3677742223038705, "AP_medium": 0.5329086038875562, "AP_large": 0.6165896595634737, "AR_max1": 0.0542538213998391, "AR_max10": 0.46037811745776347, "AR_max100": 0.6392598551890587, "AR_small": 0.41338582677165353, "AR_medium": 0.6414496314496315, "AR_large": 0.6655483870967742}, "raw_prediction_bytes": 4267320573, "images": 750, "wall_seconds": 353.53199999999924, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed43_20261104_a02/label50_R_plus_S_seed43。


## LABEL-CURVE-BATCH 验收：label50_R_plus_S_seed43

UTC 2026-11-04T07:52:22.741672+00:00。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R=1050/曝光{'R': 12000, 'S': 12000}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={"AP50_95": 0.5447330806964767, "AP50": 0.9615876372358897, "AP75": 0.5506419372720406, "AP_small": 0.3677742223038705, "AP_medium": 0.5329086038875562, "AP_large": 0.6165896595634737, "AR_max1": 0.0542538213998391, "AR_max10": 0.46037811745776347, "AR_max100": 0.6392598551890587, "AR_small": 0.41338582677165353, "AR_medium": 0.6414496314496315, "AR_large": 0.6655483870967742}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据G:\FutsalMOT_Training\runs\main_bbox_yolo11s_soccernet_RS_seed43_20261104_a02\completion_verification.json；无best/test/新策略。


## LABEL-CURVE：label25_R_seed43 STARTED

UTC 2026-11-04T07:53:05.025344+00:00。公开YOLO11s独立起点seed43，唯一真实525/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset25_seed17_v1.json", "sha256": "bdac16a1dc34b8ebdd9cc8f9a5c6cc98b1844bacbec4af90b25ffaebe7b95bf0"}。策略R，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1575不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed43_20261104_a03/label25_R_seed43。


## LABEL-CURVE：label25_R_seed43 COMPLETE

UTC 2026-11-04T08:18:57.783485+00:00。公开YOLO11s独立起点seed43，唯一真实525/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset25_seed17_v1.json", "sha256": "bdac16a1dc34b8ebdd9cc8f9a5c6cc98b1844bacbec4af90b25ffaebe7b95bf0"}。策略R，1500/1500成功更新，实际曝光{'R': 24000, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1575不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 15920, "create_time": 1793778772.3622265, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 24000, "S": 0}, "schema_version": "bbox_label_curve_v1", "job_id": "label25_R_seed43", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_R_seed43_20261104_a03\\label25_R_seed43", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5143239769941162, "AP50": 0.9403562200229939, "AP75": 0.4948897820947188, "AP_small": 0.31017365673995106, "AP_medium": 0.5047022134648075, "AP_large": 0.5777117986156047, "AR_max1": 0.05314762670957361, "AR_max10": 0.44583668543845534, "AR_max100": 0.6075321802091713, "AR_small": 0.37559055118110235, "AR_medium": 0.6106019656019657, "AR_large": 0.6301935483870967}, "raw_prediction_bytes": 4276119795, "images": 750, "wall_seconds": 352.625, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5143239769941162, "AP50": 0.9403562200229939, "AP75": 0.4948897820947188, "AP_small": 0.31017365673995106, "AP_medium": 0.5047022134648075, "AP_large": 0.5777117986156047, "AR_max1": 0.05314762670957361, "AR_max10": 0.44583668543845534, "AR_max100": 0.6075321802091713, "AR_small": 0.37559055118110235, "AR_medium": 0.6106019656019657, "AR_large": 0.6301935483870967}, "raw_prediction_bytes": 4276119795, "images": 750, "wall_seconds": 352.625, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed43_20261104_a03/label25_R_seed43。


## LABEL-CURVE-BATCH 验收：label25_R_seed43

UTC 2026-11-04T08:18:59.195320+00:00。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R=525/曝光{'R': 24000, 'S': 0}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={"AP50_95": 0.5143239769941162, "AP50": 0.9403562200229939, "AP75": 0.4948897820947188, "AP_small": 0.31017365673995106, "AP_medium": 0.5047022134648075, "AP_large": 0.5777117986156047, "AR_max1": 0.05314762670957361, "AR_max10": 0.44583668543845534, "AR_max100": 0.6075321802091713, "AR_small": 0.37559055118110235, "AR_medium": 0.6106019656019657, "AR_large": 0.6301935483870967}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据G:\FutsalMOT_Training\runs\main_bbox_yolo11s_soccernet_R_seed43_20261104_a03\completion_verification.json；无best/test/新策略。


## LABEL-CURVE：label25_R_plus_S_seed43 STARTED

UTC 2026-11-04T08:19:25.751041+00:00。公开YOLO11s独立起点seed43，唯一真实525/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset25_seed17_v1.json", "sha256": "bdac16a1dc34b8ebdd9cc8f9a5c6cc98b1844bacbec4af90b25ffaebe7b95bf0"}。策略R+S，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1575不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed43_20261104_a03/label25_R_plus_S_seed43。


## LABEL-CURVE：label25_R_plus_S_seed43 COMPLETE

UTC 2026-11-04T08:46:01.970369+00:00。公开YOLO11s独立起点seed43，唯一真实525/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset25_seed17_v1.json", "sha256": "bdac16a1dc34b8ebdd9cc8f9a5c6cc98b1844bacbec4af90b25ffaebe7b95bf0"}。策略R+S，1500/1500成功更新，实际曝光{'R': 12000, 'S': 12000}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1575不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 29928, "create_time": 1793780353.08281, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 12000, "S": 12000}, "schema_version": "bbox_label_curve_v1", "job_id": "label25_R_plus_S_seed43", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_RS_seed43_20261104_a03\\label25_R_plus_S_seed43", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5284388658707133, "AP50": 0.9588569046877232, "AP75": 0.5138132650198034, "AP_small": 0.340647104354083, "AP_medium": 0.5173298537109937, "AP_large": 0.5983677850999419, "AR_max1": 0.05311745776347545, "AR_max10": 0.45231295253419146, "AR_max100": 0.6251206757843926, "AR_small": 0.39448818897637794, "AR_medium": 0.627960687960688, "AR_large": 0.6486451612903225}, "raw_prediction_bytes": 4266504554, "images": 750, "wall_seconds": 354.4380000000019, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5284388658707133, "AP50": 0.9588569046877232, "AP75": 0.5138132650198034, "AP_small": 0.340647104354083, "AP_medium": 0.5173298537109937, "AP_large": 0.5983677850999419, "AR_max1": 0.05311745776347545, "AR_max10": 0.45231295253419146, "AR_max100": 0.6251206757843926, "AR_small": 0.39448818897637794, "AR_medium": 0.627960687960688, "AR_large": 0.6486451612903225}, "raw_prediction_bytes": 4266504554, "images": 750, "wall_seconds": 354.4380000000019, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed43_20261104_a03/label25_R_plus_S_seed43。


## LABEL-CURVE-BATCH 验收：label25_R_plus_S_seed43

UTC 2026-11-04T08:46:03.747686+00:00。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R=525/曝光{'R': 12000, 'S': 12000}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={"AP50_95": 0.5284388658707133, "AP50": 0.9588569046877232, "AP75": 0.5138132650198034, "AP_small": 0.340647104354083, "AP_medium": 0.5173298537109937, "AP_large": 0.5983677850999419, "AR_max1": 0.05311745776347545, "AR_max10": 0.45231295253419146, "AR_max100": 0.6251206757843926, "AR_small": 0.39448818897637794, "AR_medium": 0.627960687960688, "AR_large": 0.6486451612903225}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据G:\FutsalMOT_Training\runs\main_bbox_yolo11s_soccernet_RS_seed43_20261104_a03\completion_verification.json；无best/test/新策略。


## LABEL-CURVE：label10_R_seed43 STARTED

UTC 2026-11-04T08:46:45.570384+00:00。公开YOLO11s独立起点seed43，唯一真实210/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset10_seed17_v1.json", "sha256": "fccc2c998d3b6a6dd4316db0f2a66d97aa5f02571208df360b58911d611245b1"}。策略R，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1890不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed43_20261104_a04/label10_R_seed43。


## LABEL-CURVE：label10_R_seed43 COMPLETE

UTC 2026-11-04T09:12:04.473592+00:00。公开YOLO11s独立起点seed43，唯一真实210/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset10_seed17_v1.json", "sha256": "fccc2c998d3b6a6dd4316db0f2a66d97aa5f02571208df360b58911d611245b1"}。策略R，1500/1500成功更新，实际曝光{'R': 24000, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1890不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 10880, "create_time": 1793781992.6858492, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 24000, "S": 0}, "schema_version": "bbox_label_curve_v1", "job_id": "label10_R_seed43", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_R_seed43_20261104_a04\\label10_R_seed43", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.4814423149903558, "AP50": 0.9199267170116022, "AP75": 0.4457694882495966, "AP_small": 0.28836352047164937, "AP_medium": 0.4658587062288529, "AP_large": 0.5623148640440634, "AR_max1": 0.051609010458567986, "AR_max10": 0.4345937248592116, "AR_max100": 0.5732099758648431, "AR_small": 0.3401574803149606, "AR_medium": 0.5713022113022113, "AR_large": 0.6223225806451612}, "raw_prediction_bytes": 4270040800, "images": 750, "wall_seconds": 349.125, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.4814423149903558, "AP50": 0.9199267170116022, "AP75": 0.4457694882495966, "AP_small": 0.28836352047164937, "AP_medium": 0.4658587062288529, "AP_large": 0.5623148640440634, "AR_max1": 0.051609010458567986, "AR_max10": 0.4345937248592116, "AR_max100": 0.5732099758648431, "AR_small": 0.3401574803149606, "AR_medium": 0.5713022113022113, "AR_large": 0.6223225806451612}, "raw_prediction_bytes": 4270040800, "images": 750, "wall_seconds": 349.125, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_R_seed43_20261104_a04/label10_R_seed43。


## LABEL-CURVE-BATCH 验收：label10_R_seed43

UTC 2026-11-04T09:12:05.860679+00:00。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R=210/曝光{'R': 24000, 'S': 0}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={"AP50_95": 0.4814423149903558, "AP50": 0.9199267170116022, "AP75": 0.4457694882495966, "AP_small": 0.28836352047164937, "AP_medium": 0.4658587062288529, "AP_large": 0.5623148640440634, "AR_max1": 0.051609010458567986, "AR_max10": 0.4345937248592116, "AR_max100": 0.5732099758648431, "AR_small": 0.3401574803149606, "AR_medium": 0.5713022113022113, "AR_large": 0.6223225806451612}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据G:\FutsalMOT_Training\runs\main_bbox_yolo11s_soccernet_R_seed43_20261104_a04\completion_verification.json；无best/test/新策略。


## LABEL-CURVE：label10_R_plus_S_seed43 STARTED

UTC 2026-11-04T09:12:32.074365+00:00。公开YOLO11s独立起点seed43，唯一真实210/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset10_seed17_v1.json", "sha256": "fccc2c998d3b6a6dd4316db0f2a66d97aa5f02571208df360b58911d611245b1"}。策略R+S，0/1500成功更新，实际曝光{'R': 0, 'S': 0}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1890不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：null。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed43_20261104_a04/label10_R_plus_S_seed43。


## LABEL-CURVE：label10_R_plus_S_seed43 COMPLETE

UTC 2026-11-04T09:38:47.461150+00:00。公开YOLO11s独立起点seed43，唯一真实210/原2100，subset={"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\configs\\bbox_real_subset10_seed17_v1.json", "sha256": "fccc2c998d3b6a6dd4316db0f2a66d97aa5f02571208df360b58911d611245b1"}。策略R+S，1500/1500成功更新，实际曝光{'R': 12000, 'S': 12000}；960FP32/micro8x2/workers2，原warm100/.001完整cosine至.00001。三计划完整cp、两真实750val、全部逐步/原生预测/失败保留，不best/test。未选1890不训练/不伪标签；当前不预设非劣效/标注节约通过线。结果：{"state": "COMPLETE", "pid": 33504, "create_time": 1793783539.5848444, "update": 1500, "updates": 1500, "phase": "final", "phase_update": 1500, "exposures": {"R": 12000, "S": 12000}, "schema_version": "bbox_label_curve_v1", "job_id": "label10_R_plus_S_seed43", "directory": "G:\\FutsalMOT_Training\\runs\\main_bbox_yolo11s_soccernet_RS_seed43_20261104_a04\\label10_R_plus_S_seed43", "terminal": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5071367449052588, "AP50": 0.9416786406022559, "AP75": 0.48217543912399435, "AP_small": 0.32603504638284025, "AP_medium": 0.4931825447811627, "AP_large": 0.5918620213899054, "AR_max1": 0.052725261464199515, "AR_max10": 0.445756234915527, "AR_max100": 0.5953439259855189, "AR_small": 0.3716535433070866, "AR_medium": 0.5932923832923833, "AR_large": 0.6439354838709678}, "raw_prediction_bytes": 4265199119, "images": 750, "wall_seconds": 349.875, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}, "terminal_evaluation": {"tag": "step001500_final_final", "update": 1500, "main_metrics": {"AP50_95": 0.5071367449052588, "AP50": 0.9416786406022559, "AP75": 0.48217543912399435, "AP_small": 0.32603504638284025, "AP_medium": 0.4931825447811627, "AP_large": 0.5918620213899054, "AR_max1": 0.052725261464199515, "AR_max10": 0.445756234915527, "AR_max100": 0.5953439259855189, "AR_small": 0.3716535433070866, "AR_medium": 0.5932923832923833, "AR_large": 0.6439354838709678}, "raw_prediction_bytes": 4265199119, "images": 750, "wall_seconds": 349.875, "evaluated_model": "stage_EMA_at_recorded_update", "official_dataset_benchmark": false}}。证据：G:/FutsalMOT_Training/runs/main_bbox_yolo11s_soccernet_RS_seed43_20261104_a04/label10_R_plus_S_seed43。


## LABEL-CURVE-BATCH 验收：label10_R_plus_S_seed43

UTC 2026-11-04T09:38:49.054604+00:00。原driver已记COMPLETE；batch补验收1500成功更新/真实唯一N_R=210/曝光{'R': 12000, 'S': 12000}，3计划完整cp及额外边界、2真实750val、末步EMA、逐小control SHA/SEALED、owned退出/GPU锁退休通过。末AP={"AP50_95": 0.5071367449052588, "AP50": 0.9416786406022559, "AP75": 0.48217543912399435, "AP_small": 0.32603504638284025, "AP_medium": 0.4931825447811627, "AP_large": 0.5918620213899054, "AR_max1": 0.052725261464199515, "AR_max10": 0.445756234915527, "AR_max100": 0.5953439259855189, "AR_small": 0.3716535433070866, "AR_medium": 0.5932923832923833, "AR_large": 0.6439354838709678}。复用已验证配置准入，fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新数值预检。小收据G:\FutsalMOT_Training\runs\main_bbox_yolo11s_soccernet_RS_seed43_20261104_a04\completion_verification.json；无best/test/新策略。


## LABEL-CURVE-BATCH COMPLETE

UTC 2026-11-04T09:38:49.172438+00:00。seed29/43、100/50/25/10四固定标注量xR/混合共16项已逐项完整验收；复用准入无新CPU/GPU数值预检，原42数值代码保持。全部结果与小控制收据见G:\FutsalMOT_Training\runs\bbox_label_curve_batch_20261104_a02\result.json。本队列不自动启动独立比赛评价或更多训练。


## BBOX-CURVE3-RESULT-001 / BBOX-BATCH-ROOT-CLOSE-001

UTC 2026-11-04T09:52:59.047066+00:00。最新用户要求Seg先搁置、查看YOLO11s三个seed已完结果。Seg从此停止后续准备/训练/GPU，已有原GT/候选/代码/收据全部保留；不把本次BBox收口当Seg启动授权。

seed29/43批量16项于2026-11-04T09:38:49Z（北京17:38:49）全部COMPLETE；连同seed17既有8项，三seed17/29/43×真实标注100/50/25/10%×R/1:1混合共24模型均为独立公开YOLO11s/1500成功更新/末步EMA。原driver已记STARTED/COMPLETE，root不重复这些事件；这里只补24结果核对和批量最终闭合。各项相同960FP32/micro8×2有效16/workers2、warm100 LR0→.001完整cosine至.00001/SGD Nesterov .937/WD.0005/clip10。R24000R0S，混合12000R12000S，同总24000曝光但真实曝光不同。固定真实图池2100/1050/525/210、selectionseed17嵌套不重选；训练seed变化不构成独立比赛或独立子集样本。

root实际核24个job_table的seed/N_R/策略/预算/输入/有效batch、status COMPLETE1500/末曝光、两精确tag750summary与末EMA一致、3计划cp小收据bytes（额外边界保留）。16新组SEALED scope control_snapshot_only的全部小文件逐SHA复核；seed17已闭合8组复核原status/completion/manifest引用，不重复旧全媒体审计。实际batch12512/wrapper15864/host及known children/workers原CT身份全部已退，GPU锁/批量锁退休、无精确脚本argv训练写者/UE；无PT/NPZ/媒体全hash。批量自身control已一次SEALED：30文件211714B。

绝对末步开发集指标（单位%；不是AP95单阈值；全部正负结果保留）：

|真实N|seed|策略|AP50:95|AP50|AP75|
|---:|---:|---|---:|---:|---:|
|2100|17|R|55.163426165|95.876456222|56.872379727|
|2100|17|R+S|54.044657304|95.858181776|54.136439622|
|2100|29|R|54.859228480|95.674777120|55.828641114|
|2100|29|R+S|54.212171489|96.028097690|54.438358241|
|2100|43|R|55.483043046|95.424355970|57.731072216|
|2100|43|R+S|54.497180730|96.116590106|54.727712421|
|1050|17|R|53.911248705|95.293739054|54.372677698|
|1050|17|R+S|54.207372547|95.545265986|54.360785671|
|1050|29|R|53.621477532|95.174605623|53.058045540|
|1050|29|R+S|53.546801796|95.899613340|52.803953595|
|1050|43|R|54.218153374|95.236565105|54.543679349|
|1050|43|R+S|54.473308070|96.158763724|55.064193727|
|525|17|R|51.786728311|94.836833311|49.859485323|
|525|17|R+S|52.411902268|95.018611372|50.443864217|
|525|29|R|51.590908882|93.686952879|50.053436589|
|525|29|R+S|53.328039784|95.261581241|53.015065460|
|525|43|R|51.432397699|94.035622002|49.488978209|
|525|43|R+S|52.843886587|95.885690469|51.381326502|
|210|17|R|48.613034512|91.943867492|44.727624783|
|210|17|R+S|50.158494352|94.065387607|46.371886463|
|210|29|R|48.313808080|91.898793203|44.796125392|
|210|29|R+S|50.660035652|93.625779067|48.596068182|
|210|43|R|48.144231499|91.992671701|44.576948825|
|210|43|R+S|50.713674491|94.167864060|48.217543912|

三seed汇总：均值±样本标准差(ddof1)，误差条不是置信区间；Δ为同训练seed同标注量的混合−R百分点。

|标注量|R AP50:95|混合 AP50:95|Δ17|Δ29|Δ43|平均Δ±SD|
|---:|---:|---:|---:|---:|---:|---:|
|100%|55.168566±0.311939|54.251337±0.228790|-1.118769|-0.647057|-0.985862|-0.917229±0.243230|
|50%|53.916960±0.298379|54.075827±0.477055|+0.296124|-0.074676|+0.255155|+0.158868±0.203289|
|25%|51.603345±0.177492|52.861276±0.458316|+0.625174|+1.737131|+1.411489|+1.257931±0.571662|
|10%|48.357025±0.237371|50.510735±0.306226|+1.545460|+2.346228|+2.569443|+2.153710±0.538454|

当前能说明：在这个已开发game6/750图面板、固定1500步配方下，合成混合对真实标注较少的25%与10%三seed全部正向；主AP平均分别+1.257931/+2.153710pp，且AP50/AP75分别+1.202158/+1.812785及+2.007899/+3.028267pp。50%主AP仅+0.158868pp、两正一负；100%主AP三seed均低、平均−0.917229pp，当前配方没有提升全量标注条件的主AP。全量AP50平均+0.342427但AP75−2.376528，不能用宽松单阈值替代主指标，更不能据此宣称合成普遍有害或普遍有效。

标注效率折中：10%混合vs25%纯R平均−1.092610pp；25%混合vs50%纯R−1.055684pp；50%混合vs100%纯R−1.092738pp。混合用更小独立真实标注池得到较接近的大池纯R结果，但仍有差距；无事前q/容许损失/非劣效或等效界限，不事后设约1pp为PASS，不计算已证明节约多少标注。三seed只覆盖训练随机性，同一固定标注子集的选择不确定性未测；真实曝光减半/减少重复与合成信息的独立机制未拆分。本结果是完整混合配方的开发效用。

下一已批准证据仍为24末1500EMA对game8/game9两场各750图的48次比赛外确认；截至本次未派发/未读取真实确认GT或媒体/未产生独立比赛分数。需新评价后端外围实际审阅和24准确registry/spec/GT来源/资源时间/单GPU生命周期Gate，不能把训练完成或CPU候选通过当评价已准入。现有bbox跟进状态PAUSED保持，不擅自恢复；不扩训练seed/规格/比例/LR/来源，Seg搁置。

本次只新写结果分析工具/原G批量表格JSON/CSV/科学PNG/SVG与一次小control封存，未训练/推理/重做数值预检。初次分析因旧seed17 N50/N10完成收据使用labelcurve_completion_verification_v1而缺global_updates字段停止在写结果前；查原schema后只加显式读分支，读取原last_step/owned/CP/evaluation，不改旧收据或造字段。

证据：{"comparison": {"path": "G:/FutsalMOT_Training/runs/bbox_label_curve_batch_20261104_a02/label_curve_development_comparison.json", "bytes": 95383, "sha256": "6206739a6ce2c4ca25c4a86981917c28e96c41e19d0028301266fc5ffc57ac7b"}, "CSV": {"path": "G:/FutsalMOT_Training/runs/bbox_label_curve_batch_20261104_a02/label_curve_development_results.csv", "bytes": 4790, "sha256": "a74582e07bda953880ce4c28cf1a90111ff1c29e7d3777cc94157875a387087e"}, "PNG": {"path": "G:/FutsalMOT_Training/runs/bbox_label_curve_batch_20261104_a02/label_curve_development.png", "bytes": 133486, "sha256": "c4525571e8944bc0e7fa0755e012ee1ec9a02ac81e52449df7d8e8d837d4d298"}, "batch_completion": {"path": "G:/FutsalMOT_Training/runs/bbox_label_curve_batch_20261104_a02/completion_verification.json", "bytes": 13342, "sha256": "10b2a9017698b5583834d4901cfbe7f0ef84af50bb0c673b3b8848fa1bf3f244"}, "batch_control": {"path": "G:/FutsalMOT_Training/runs/bbox_label_curve_batch_20261104_a02/control/CONTROL_MANIFEST.json", "bytes": 5035, "sha256": "63896816445073bd4f6d8458c34f705aaf70bd551b7d145818a0cdbf17802a51"}, "analysis_code": {"path": "D:/projects/FutsalMOT_SyntheticDatasetResearch/tools/summarize_bbox_label_curve.py", "bytes": 14468, "sha256": "fb101de1af5b5e24e1c75c0c7b426554192d89b67d7cc1eddd6018b565bf04a7"}}


## MONITOR-CONFIRMATION-001 / BBOX-CONFIRMATION-ADMIT-001 / BBOX-CONFIRMATION-FAIL-001 / BBOX-INFERENCE-SPEED-001

UTC 2026-11-04T10:18:54.559520+00:00。用户直接要求把两场比赛外评价加入训练观察台、开始24模型推理并探索加速；Seg仍搁置。原24训练均已闭合，本轮没有训练更新，也没有恢复逐训练CPU/GPU探针。

先核原41数值及24个实际job_table/status/control/checkpoints.jsonl/末1500receipt/sourcebindings，冻结registry后才读取确认GT小文件。固定seed17/29/43×100/50/25/10%×R/RS末EMA及game8/9各750UID，共48项；不选best。两场原test来源与train/dev比赛/UID无交；game8目标8751框/已知非目标1643框，game9目标9208框/已知非目标1744框，true-ignore均0，原GT不改。game9历史observed/game8结构访问史保留，不称首次盲测。完整新增286156535472B另G100GiB、D32MiB另100GiB；起前freeG562771161088B/D121621020672B，48×900+600=43800秒可在原Nov7截止闭合。

独立评价外围新32CPU由root实际PASS/0skip6.520秒（首次模块名调用因tests不是包而零加载，改discover，不改软件）。a01冻43/原observer，suite SHA3230fbad5f77fa99bde3f6da405607be8854f5388f876c41a646a3a3ac59e179于2026-11-04T10:09:58.4055169Z开始唯一隐藏派发，wrapper4960/CT1793786998.4063015、实际CPU supervisor19956/CT1793786998.4231749；本suite只推理/optimizer_updates0，原900秒从Popen前计，不重置。首项seed17_N100_R__game8 child30356/实际20688的原始CT/命令及全部worker见原dispatch/life。

真实首项16.573052167892456秒后FAIL_STOP：reader和预处理保留test正确，但原batches._collate_lineage仍拒绝test，DataLoader第一批组装失败；尚未network.evaluation_output提交任何图，raw目录0文件、无predictions.jsonl/无summary，0/48完成，后47项无派发。不是模型成绩或GPU数值失败。原异常/冻43/请求/partial保持；root核全部owned原CT已退/GPU锁退休，失败job5小control12824B SEALED逐SHA。没有自动retry或resume。

修复在根独立候选bbox_confirmation.py完成，不热改a01冻包/原训练41/42：新增ConfirmationCollator只接受factory授权、固定比赛/UID/R/test/index/无增强请求，机械派生原lineage和collate；原make_loader参数、预取1、workers2不变，仅替换collate回调；原训练collate继续拒绝test。逆AST三接点证据保留，不把test改名val。新增虚构完整batch→CPU prepare数值exact以及实际两个spawn worker/退出回归，原失败场景先RED，修复后全部confirmation+生命周期45CPU PASS/0skip34.240秒（不读真实媒体/权重/GPU）。旧mock只测Dataset未覆盖collate的缺口如实保留。新candidate SHA47cb06501002ca34e2a23c59adf88f616529c2bd97632efad0e1524fbe1b6099；独立a02 suite SHA02cd3cb88efd7b2d98f40a4137195c4d776496eaab15e2740063401adf3b7fbd已冻结PREPARED_NOT_DISPATCHED，完全同24模型/原GT/原41数值，实际free/resources重核。已有明确失败不自动重试规则，待用户新继续指令才派发，不擅自把CPU通过说成GPU成功。

观察台原localhost8765/v8/PID5780/CT1793750441.2193077完全保留；同截止只读评价接口127.0.0.1:8766，实际PID3440/CT1793786972.8624058，原deadline2026-11-07T00:00:41.387415Z不延。静态HTML新增48项/两场完成模型数/最近图像进度/预计耗时/seed与比赛筛选/逐项AP50:95、AP50、AP75/状态/耗时；仍保留九字段训练筛选。当前明确显示a01失败、0/48、后47未执行，不给虚构AP；完成计数需结果+原生退出+封存。仅小JSON/有界stdout，无GT/PT/native预测扫描。59观察台CPU检查PASS/0skip1.148秒、21客户端检查通过，实际IAB已确认面板/失败状态/48行及原筛选同页，HTTP/CORS只允许原本机8765。旧HTTP断言仍v6导致首次48测1fail，按真实升级v9修正断言后通过；原8765进程未重启，v9为根读取实现，原服务仍v8。页面刷新或关闭不控制评价。

原bbox原生跟进仍PAUSED，失败后不擅自恢复；g/生产UE不动。后续用户若明确继续，使用已经准备的a02仅唯一启动，当前CPU修订/资源不得再冒称无需真实GPU终验；任何新失败照原规则停止，不自动a03。方案/PAPER始末SHA不变。精确引用：{"a01_failure": {"path": "G:\\FutsalMOT_Training\\runs\\bbox_confirmation_suite_20261104_a01\\failure_verification.json", "sha256": "fa7eb632ab26142bb32d9ccb743cce5ed2bde916ec1562a7a5845df2da4b9b26"}, "a02_readiness": {"path": "G:\\FutsalMOT_Training\\runs\\bbox_confirmation_suite_20261104_a02\\readiness_verification.json", "sha256": "c626e4f76c335e443b91bc9d49f0b271290263ca2d6b11258794089b998053f5"}, "speed_component": {"path": "G:\\FutsalMOT_Training\\runs\\bbox_confirmation_suite_20261104_a01\\inference_speed_exploration.json", "sha256": "1c5e09eca17f7a7b785d8524cddb053b6d0c16f82b7a9ddda72768fd63eafa65"}, "UI_html": {"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\tools\\training_monitor.html", "sha256": "0e606b48ec824706cb9e2251ac59ea3fee6e8851cf5f85e5b9c31ba1df9c1c31"}, "UI_reader": {"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\tools\\training_monitor.py", "sha256": "0b87223a7112fd0e88003d480294fff316522f91be43ff31fc459b462b872833"}, "UI_companion": {"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\tools\\bbox_confirmation_monitor.py", "sha256": "19acd0035b669de35272499a8b752c9096d118ee9b5425b5b0066ac9e917bac1"}, "confirmation_candidate": {"path": "D:\\projects\\FutsalMOT_SyntheticDatasetResearch\\src\\futsalmot\\bbox_confirmation.py", "sha256": "47cb06501002ca34e2a23c59adf88f616529c2bd97632efad0e1524fbe1b6099"}}


## BBOX-CONFIRMATION-A02-START-001

UTC 2026-11-04T10:37:02.672287+00:00。用户在a01失败已闭合/修复45CPU候选/a02已冻结未派发的具体说明后直接“启动”，现已实际落实。唯一a02于2026-11-04T10:34:01.4002604Z（北京18:34:01）隐藏启动，D runs/bbox_confirmation_suite_20261104_a02，G同名；wrapper13564/CT1793788441.4012089，实际CPU supervisor34072/CT1793788441.419639，完整D observe.py Dsuite --suite。suite SHA02cd3cb88efd7b2d98f40a4137195c4d776496eaab15e2740063401adf3b7fbd，确认模块47cb06501002ca34e2a23c59adf88f616529c2bd97632efad0e1524fbe1b6099/原41不变；root_dispatch登记用户明确继续与a01失败/a02 readiness精确引用，不改旧失败。

首项seed17_N100_R__game8 wrapper31888/CT1793788445.4874516、实际GPU controller25608/CT1793788445.5046494；runtime起10:34:05.480670Z/max900，固定deadline10:49:05.480670Z。root核父/实际child CT+完整argv/GPU锁正确、已实际写原生000001.npz5683892B及非空predictions.jsonl，越过旧collate拒test点；本次真实推理已执行，不仅是CPU或初始化。首次stdout进度颗粒约200图，0表示尚无进度行而非没有推理。当前最近图数200/750、完整验收0/48；未冒称全组或首项750完成，启动时native/PT不全hash。process_lifetime_isolation_verified=false，不承诺Codex重启隔离。

固定24末1500EMA×game8/game9各750=48项原FP32/960/micro8/workers2无增强推理，完整native/JSON/AP/失败保留，test来源不改名，原GT/原NMS/COCO/所有冻结数值保持；不训练/optimizer_updates0、不重复CPU/GPU训练预检，不把局部CPU加速候选接入正式执行。实际空间再次核freeG562771091456B/D121617047552B，完整286156535472B另G100GiB/D32MiB+100GiB，43800秒可在原Nov7全局截止闭合。任何新失败或人工停止即停，不自动a03/retry/resume；唯一监督器顺序派发，无需聊天逐项启动。

原8765/v8/5780与原截止不动；原子切换D .bbox_confirmation_monitor.json到a02，G monitor_handoff.json保留旧a01指针。实际8766接口已返回a02 RUNNING/48项，原训练观察台同页自动跟随。按本次用户明确继续恢复已有bbox有限20min跟进ACTIVE，原最多504轮7天/更早2026-11-07T23:55Z不延；健康/正常交接安静，仅整批完成/新失败/需处理通知。Seg继续搁置，g和生产UE不动。启动真实证据：{"path": "G:\\FutsalMOT_Training\\runs\\bbox_confirmation_suite_20261104_a02\\startup_verification.json", "sha256": "b1277bdf0a44250c8e1dba7b43c70cb3c034a492f6a70cdd973dd598892a831b"}。EXPERIMENTS/PAPER始末只读SHA保持。


## RESEARCH-TASK-HANDOFF-001：BBox / Seg / Pose职责拆分及两个新聊天首轮只读交接

UTC 2026-11-04T10:44:01.055549+00:00。用户明确决定原聊天此后只负责BBox方案和实验，Seg、Pose分别转给两个新的用户聊天agent，并要求各新聊天首轮只了解现状、复述给用户。已通过原生create_thread实际创建并触发：Seg「Seg方案与实验｜现状交接」chat_id=01a1a627-79a8-7af1-b59e-abb69fd9fa2b、Pose「Pose方案与实验｜现状交接」chat_id=01a1a627-8a05-7c63-a9ba-c90ae819a89e，host=local，现登记项目D:/projects/FutsalMOT_UEDataset，交接提示明确实际研究根D:/projects/FutsalMOT_SyntheticDatasetResearch。没有创建worktree/新分支或改变项目源码位置；未指定模型覆盖，沿用户默认。

两份可见首轮提示均明示：只读研究AGENTS/README/TODO/唯一实验日志及相关小收据/代码，了解后向用户复述并停止；不下载、不测试、不补实现、不改任何文件/共享文档/方案/GT、不训练/推理/初始化CUDA，不创建自动化或子代理，不访问/清除/停止BBox锁与进程。历史“继续准备”不构成新行动授权。未来工作由用户在各新聊天另行安排；本次交接本身不解除暂不训练限制。现GPU固定BBox 24末EMA两比赛48评价仍由本聊天独立负责，D/G bbox_confirmation_suite_20261104_a02保持，原有限bbox跟进不改，g/UE生产不动。

Seg交接覆盖ihelon512原GT/508与426训练候选、Quantigo备用、Acme100原PNGpart编码/200RGBA RLE exact与83帧406人物身份恢复/80图383人物候选、来源独立性与许可/疑难/排除偏差，及CPU-only backend和缺正式训练/评价生命周期的事实。明确所有候选非正式准入，不把identity或CPU夹具当公开s/960/GPU通过。优先小总收据G:/FutsalMOT_Dataset/real/football_seg_public/derived/SEG_PNG_RLE_READINESS_20261104.json与SEG-PNG-RLE-009等，不重新全媒体审计。

Pose交接覆盖3DSP原train4000有2D标签帧/20比赛，test200无2D标签且官方比赛重叠、17非COCO17/缺逐点visibility/3D为lifting、12严格共有身体点、尺寸/路径/边界/时间差异、14/3/3场2820/660/520帧候选非盲测，以及单人crop辅助PDJ/PCK与完整COCO17 OKS主任务的边界。已核现存pose_3dsp.py SHA2dc231272b664974349017b94f746b3558dd2836124648d517ce9a77b48f909f/tests8d79e2ec7035da0ee3f77b5438cf723c9730b66964580737a0ffe91bd474c198，仅读原SHA不重跑9CPU。历史ZIP已清理/未形成正式媒体池与训练入口，不能假设仍可运行。

此条只登记实际创建与首轮授权边界；各新聊天复述是否完成由原生wait_threads后续实读确认，不提前冒称已读完。现行EXPERIMENTS/PAPER始末SHA不变，未修改方案内容或历史证据。


## RESEARCH-TASK-HANDOFF-001-CLOSED：两新聊天已完成首轮只读复述

UTC 2026-11-04T10:45:34.648333+00:00。原生wait_threads已确认两新聊天首次turn均completed/error=null且聊天idle：Seg 01a1a627-79a8-7af1-b59e-abb69fd9fa2b（turn耗时114893ms）、Pose 01a1a627-8a05-7c63-a9ba-c90ae819a89e（105874ms）。两者已分别向用户提交完整中文现状复述并停止等待后续直接安排，均报告本轮无修改/测试/下载/训练/模型推理/CUDA/BBox巡查，不把交接当继续准备授权。

Seg复述区分ihelon508/426、Acme80/383等未激活候选与正式数据准入，保留来源独立性、极小监督和排除偏差，明确CPU-only/小夹具不等于公开s/960/GPU完整管线，训练500步生命周期仍缺。Pose复述区分3DSP H36M17/12语义共有点与完整COCO17、test无2DGT/比赛重叠、候选14/3/3比赛分组和已观察GT史，明确单人crop辅助指标非标准整图OKS且正式训练未准入。具体后续问题留各聊天由用户安排。

现有原生bbox跟进已通过automation_update保持ACTIVE/原20min/Nov7截止，并前置最新职责交接：原聊天只推进BBox，不再准备/委派/推进Seg或Pose；两新聊天ID与首轮只读限制已写入。未创建新的Seg/Pose自动化，也未触碰BBox冻结包/运行控制或GPU锁，g/UE生产不动。此为行政交接完成，不是Seg/Pose技术准入完成。


## BBOX-DOMAIN-GAP-001：现有训练/开发标注域差，CPU只读统计

UTC 2026-11-04T11:33:52.660518+00:00。用户要求在当前比赛外评价继续期间，仅以现有训练/开发集统计合成与真实每帧人数、目标尺寸及遮挡差异。已独立完成tools/analyze_bbox_domain_gap.py，G原位bbox_domain_gap_train_dev_20261104_a01，未更改任何GT/索引/缓存/权重/活动评价冻包，不读取game8/game9 GT、媒体或预测，不初始化CUDA/不import torch。固定统计范围S_train24149图229727有效框、S_dev6030图57061框、R_train2100图31231框、R_dev750图9944框，真实只train/val。原始player/goalkeeper且GTflag1的有效裁边框；已知非目标单记，合成无mask像素人物单记，不将固定十个场上人物全部视为图内目标。数据池/几何/阈值及资源规格在结果前冻结analysis_spec.json。

依次S_train/S_dev/R_train/R_dev：每帧人数均值9.512899084848234/9.46285240464345/14.871904761904762/13.258666666666667；长边960、无随机增强的框高中位36.5/36.25/48.5/48.0像素（p10..p90分别24.25..58.5/24.5..58.25/32.5..73.5/32..72）。与其它目标框交集并集占自身框面积>=0.3的目标比例13.825105451253%/11.252869735897%/9.894015561461%/21.761866452132%。人数按帧、尺寸/重叠按目标加权，另有seed/match/机位分层及等组权重摘要，不把相关帧或对象当独立样本计算显著性。960面积小于32平方像素的目标占72.3572%/73.6931%/44.6992%/44.6702%，仅输入尺度诊断，不是官方原图COCO small分组。

重叠是potential occlusion/crowding proxy，不是人物真实遮挡比例；精确并集面积避免重复累计。合成visible_mask/bbox是形状填充率，不可当未遮挡比例；真实最后3列全部-1不提供可比遮挡真值。合成visible-mask紧框与真实官方框遮挡范围语义未证等同，结果不能消除该标注域差。真实训练每帧人更多却框重叠更低、真实开发人数更少却重叠更高，说明人数不足以单独描述拥挤；不能据此证明人数因果影响小或尺寸为当前YOLO11s误差主因。当前是已有S5→R11数据分布描述，不是固定相机/尺度/姿态后的赛制单因素对照，不给域差PASS或统一百分比分数。

执行242.25秒/CPU27.375秒/峰值192860160B；单核19/BelowNormal/VeryLow I/O、G标注限速2MiB/s/D8MiB/s。只读注释文本及元数据，不扫描RGB/mask/NPY/PT。5项解析几何自检通过；root再次核10个结果文件全部bytes/SHA、33029逐帧行数/UID唯一/目标数/按目标加权重叠一致，已实际查看PNG三联图。10小文件6271431B单次SEALED，summary SHA97b1f58fa141578c663abafba4c62f44e95a54e3bfe92fefade04898610b83e6；manifest SHAe0d971ff4b832c91874324deecaba84946f9cc6adef79080c674c64631e62320。frames.csv/groups.csv/distribution_bins.csv/PNG/SVG/源码/资源收据/输入引用完整保留。不重跑无变化统计。活动a02 suite_request起末SHA02cd3cb88efd7b2d98f40a4137195c4d776496eaab15e2740063401adf3b7fbd不变；未改调度/推理/阈值/GPU锁，未测量端到端吞吐差，故不声称零可测性能影响。方案/PAPER始末SHA不变。

## BBOX-COUNT-SCALE-LITERATURE-001：人数、尺度与拥挤的证据边界

同轮按用户追加问题检索原始研究。Hoiem等ECCV2012《Diagnosing Error in Object Detectors》（官方项目https://dhoiem.cs.illinois.edu/projects/detectionAnalysis/；作者讲座摘要https://videolectures.net/eccv2012_hoiem_detectors）在所研究旧检测器中强调尺寸敏感性、定位和相似对象混淆；原PDF抓取超时，未假称完整阅读。Dollár等TPAMI2012《Pedestrian Detection: An Evaluation of the State of the Art》（https://pdollar.github.io/files/papers/DollarPAMI12peds.pdf）16检测器/6数据集分尺度、遮挡评价支持低分辨率和部分遮挡困难。Zhang等CVPR2017 CityPersons（https://arxiv.org/abs/1702.05693）支持小尺度/重遮挡及训练多样性的重要性。Liu等CVPR2019 Adaptive NMS（https://arxiv.org/abs/1904.03629）与Chu等CVPR2020 Detection in Crowded Scenes（https://arxiv.org/abs/2003.09163）强调局部密度、目标重叠与NMS误抑制，不支持忽略拥挤。只引用原作者/原论文，不把跨旧数据/模型结论的效应排序直接搬到YOLO11s足球。

未找到直接控制其它条件、证明五到十一人制对YOLO11s bbox影响可忽略的适用证据；不能用上述文献声称人数无影响、S5等价S11或无需S11对照。可在论文中以尺度/遮挡/局部密度分布及真实迁移评价回应，但剩余赛制、相机、外观与标注差异保留为局限。当前未追加训练/模型推理/阈值调整；若将来判断主要误差来源，须另行同尺度/重叠条件的分层性能或受控因素研究，当前分布统计不替代该证据。


## BBOX-USER-SCHEDULED-STOP-001：按用户安排今晚21:30停止，余项明天

UTC 2026-11-04T12:39:30.246764+00:00。用户最新直接要求“当前的验证到21：30停止，剩下的内容明天再验证”。当前北京时间20:36起落实定时安排，目标2026-11-04T13:30:00Z/北京时间21:30:00；未提前停止、未改活动评价43/原900秒deadline/范围。独立CPU隐藏定时器位于G bbox_confirmation_suite_20261104_a02/scheduled_stop_2130，wrapper36468/CT1793795879.2283459、实际34756/CT1793795879.2487752，完整命令该目录stop_at_2130.py。helperSHA7306a3fc4c7c864da27c808823c2e36b1b3e40c2545bf02e3c74043f0da5620b；requestSHA97ce14804fa91ae1efc2acdc1be5044426fc12980316ac75b0df370cc0ea1ceb。已独立核startup ARMED、准确CT+cmd、stderr0；当前D STOP未写/timer result未生成，不能提前称已暂停或owned退出。一次CPU语法编译仅验证定时helper，不运行任何重复训练CPU/GPU预检或数值探针。

定时器不CUDA、不持GPU锁、不读GT/媒体/权重/AP或图数；只核准确a02父34072/CT1793788441.419639和完整argv、原suite_requestSHA02cd3...，到时写原D suite/STOP.json现有受控路径。父提前已有终态则记录不写；原child在批次边界停止、原最外worker shutdown和owned退出/锁退休路径照旧，绝不强杀。21:30是发出停止请求时间，安全清理可能稍晚。已完成结果和受控partial保留，不把partial算750完成或删除后重跑。原实现会将人工停止child记USER_STOP/父suite记FAIL_STOP，此为用户计划停止而非新增软件故障，不篡改原状态为PAUSED/COMPLETE。

原生automation_update已前置该用户停止指令，保持ACTIVE/原20分钟/Nov7截止，仅用于今晚停前观察与停后安全闭合。21:30后核timer result/STOP和准确active身份，如timer失效且原suite确活可直接按本用户授权写STOP；不能重派timer/评价。停后root应核已完成/受控partial/未派发清单、小control/真实owned全退和GPU锁退休，然后事实日志/TODO/README并原生PAUSED过夜。用户未指定明天具体时刻，不猜时自动启动，不自动恢复a02/a03或partial；明天继续时只处理余项，已完成不重复48。旧失败、GT、预测、冻结、g/UE及Seg/Pose交接均保持。方案/PAPER始末SHA不变。


## BBOX-USER-SCHEDULED-STOP-CLOSED-001：昨晚定时停止已完整核验

UTC 2026-11-05T01:44:27.288191+00:00在用户明确继续前根独立审阅。原a02于北京时间11月4日21:30:00.000500写STOP，child21:30:03.180911 USER_STOP，父21:30:03.769325原FAIL_STOP；是用户计划停止，非新软件错误。29完整model-match已各750/末1500EMA/summary引用/CP bytes/原900全life/原生退出与小control逐SHA验收，另seed29_N10_R__game9被中断102.4089233875秒，18未派发。150条knownowned实际CT全退、父wrapper/timer全部真实退出、无未知exact-script writer/GPU锁退休已核；无强杀/删partial/改GT/旧结果重写。206小control1194312B单次SEALED SHA f9795109f731a84b36e1854e45d66ff3f75d9cb44c676b0799f052f2aad4b24d；原G user_stop_verification SHAe14a083ca18f447b8c2eb9db20461ece70f431e60f28cf5e88cc4b7065002085。昨晚原停止结果未日常补造日志，今晨核实后如实登记。

## BBOX-CONFIRMATION-REMAINING-START-001：用户明确继续，仅新执行余19项

用户2026-11-05“完成剩下部分的验证”撤销昨晚过夜暂停，已实际于UTC01:40:14.214934/北京时间09:40:14唯一隐藏启动D/G bbox_confirmation_suite_20261105_a01；wrapper23492/CT1793842814.2175884，实际CPU supervisor13004/CT1793842814.234566，完整G本组continue_remaining.py --suite。新suiteSHA996d91a9d2cd38827aa26bbbe8fb4b3e97db9b5942663a13c41ce1514d5ead2f、控制器SHAcb4aa1258f1cc35faf0b7ac3838ded22d5b95a2a38d5db3f0b702c0c0cca5fc2、handoff66ba5ffc3951d37af4b10612de1d1ec8844d1fb519883c12aa27ad812c6d7934/readiness9ed10be83c8f9a36c21083582b3defca2a4a2cb7d5e9dad5f91fda066f53897d。没有等待新许可或重跑已验证训练CPU/GPU探针。

独立接续父只负责从原remaining_jobs取得余19并调用原prepare_job/observe_job/seal_suite；43执行源码与原observe.py逐SHA完全相同，FP32/960/micro8/workers2/原test身份/原GT/NMS/COCO/完整native和JSON/末EMA/每项900秒不变。29已完成项仅将原result/lifetime/SEALED等小收据逐byte/SHA复制为继承证据，所有旧真实路径/时间/summary_reference原值保留；G handoff逐文件source_reference+destination证明它们是旧执行的真实结果，不是29新推理，不复制或声称新组有29套native/PT。新suite保留完整48身份矩阵，原remaining_jobs实际识别29完成并只准下一项；新父仅遍历剩19。旧partial原位保留，在新组对seed29_N10_R__game9按用户新明确继续独立重新评价，旧900runtime/USER_STOP不重置或覆盖，后18原未派发按原顺序接续。任一新失败/人工STOP/未知提交停止、不自动新attempt/隐式retry。最终29旧+19新共48唯一model-match，来源表须保留每项实际执行组。

首新项wrapper15444/CT1793842818.5872593，实际GPU controller11380/CT1793842818.6053848，完整D新observe.py Dnew --child seed29_N10_R__game9；runtime01:40:18.581821Z/max900/deadline01:55:18.581821Z。根真实核首native000001.npz5689648B与JSON已生成、父/child准确身份/锁及两stderr0；已执行真实推理，尚不冒称19或首750全部完成。G startup_verification SHAfe8d678aca29e404f87c195ee15a894738dbb07a2319eb20a1821f01b8327d8b。process_lifetime_isolation_verified=false，不承诺重启隔离。

余19完整native+JSON109709610000B，另一次partial5774190000B/瞬态3221225472B，共118705025472B再G100GiB，D32MiB再100GiB；启动实际freeG437084377088/D121562931200B，17700秒保守上界能在原Nov7T23:55Z闭合。原prepare_suite还实际通过保守48完整资源公式，不代表重跑48。数据、已完成24训练、不变域差统计、Seg/Pose新聊天职责和g/UE均不动；方案/PAPER始末SHA保持。有限bbox原生跟进已更新为ACTIVE/原20min/原Nov7上限，清除旧21:30暂停优先级；健康安静、监督器自行衔接，不等聊天，完成全部或新失败才通知。

只读观察台启动核发现旧PID5780/3440已不存在且接口拒绝/超时，根因是旧服务已退出，退出原因未确证，不凭此重启评价。按原绝对截止2026-11-07T00:00:41.387415Z仅恢复两个隐藏只读服务：8765 v9实际PID11800/CT1793842984.5408287（wrapper13472），8766 v1实际9444/CT1793842985.179317（wrapper21808），均明确--deadline-utc原值，未延72h。旧登记在G monitor_restoration留存，原指针保存在monitor_handoff，当前已指新组；实际health身份+截止通过、/api/confirmation真实返回RUNNING/29已完成/48总项/首余项active/errors[]，无训练或评价重启，无UI数值代码改动。此项恢复按systematic-debugging只核必要生命周期/接口，不跑训练预检。后续以新实际身份和G monitor_restoration/verification.json为准，旧服务仍活描述作废。


## BBOX-CONFIRMATION-FINAL-001：固定24末EMA两比赛48项全部完成并根终验

UTC 2026-11-05T03:50:53.925198+00:00。用户2026-11-05“完成剩下部分的验证”已完整完成。新接续19项于UTC03:31:48.217920/北京时间11:31:48结束，原29完整结果逐原bytes继承且不重算，合计24模型×game8/game9共48唯一model-match/36000图，三seed17/29/43×100/50/25/10%×R/RS均末1500EMA，原FP32/960/micro8/workers2/原NMS/COCO/roleignore/原GT和源test身份不变，无新训练/参数搜索。所有a01collate失败、昨晚USER_STOP partial与原停止/FAIL_STOP保持原值，不用partial充750，不删旧预测或覆盖旧seal。

root独立核48条实际执行来源的result和750summary/末1500EMA/原checkpoint receipt字典和PT stat bytes、数值及iterator inverseAST exact、精确GetProcessTimes全进程<=900、owned退出/worker cleanup/GPU锁退休、各job SEALED逐小SHA。29继承348个source/destination文件逐bytes/SHA exact，所有原路径/实际运行时间保留，原summary_reference指向旧G真实输出；剩19新结果指新G，未搬native。48项240条owned收据按真实CT全部退出，CPU supervisor13004/wrapper23492已真实退出，无未知exact-script writer，GPU锁退休。新suite原288小control1429446B SEALED SHA22ac3243bb5d8c00ed7afd1fa691af48a62b72ac5ed87dcb6b88a78d93fd622d保持，不覆清单。

合计48已接受项的原生全进程秒数min342.1586401462555/max372.10487508773804/sum17069.23971748352；只指已接受48，不包含原collate失败16.573052167892456秒和人工停止partial102.40892338752747秒等历史成本，不称完整总GPU消耗。原summary raw_prediction_bytes累计204765407866B，非全输出容量实扫。无需重哈/解码PT/NPZ/媒体或再次推理。

原G输出confirmation_absolute_results.csv完整48行AP50:95/AP50/AP75绝对百分值及模型/seed/标注量/真实执行组/summary与CP引用；paired_differences_per_seed.csv、grouped_summary.csv、confirmation_equal_match_per_seed.csv和confirmation_comparison.json保留全部正负及开发game6比较。固定跨场口径先同一模型game8/game9 AP算术等权，再按三seed配对与mean/sampleSD(ddof1)；不是合并预测重算COCO AP，不把6个seed-match当独立比赛、不把SD当CI。root从原result独立fsum/sqrt复核144组均值/样本SD和48CSV全部3指标，已实际查看标准PNG曲线；9派生小结果861123B单次RESULTS_MANIFEST SEALED SHA60f150dc79acee50375abf0d7d44b3af4a6f332f815fe4f291b3c0e7ffc0fc50。根completion_verification SHA2137004730c8f49c864cc92db729f72e272de62becebb2854f23a8d6edb99de1；comparison SHAa4b108adc407141349a38c950432cf3fc5fb1548ab3bc6fa04f0ece8fd42cc19。

主AP50:95三seed均值，100/50/25/10顺序：game8 R60.7025/60.2598/57.8013/54.0746，RS61.3258/60.6621/59.4340/56.1955，配对差+0.6234/+0.4023/+1.6327/+2.1209pp；game9 R56.8367/56.1436/53.4561/50.3392，RS56.4967/56.2762/55.6729/52.2287，差−0.3400/+0.1326/+2.2168/+1.8895pp。固定两场等权R58.7696/58.2017/55.6287/52.2069，RS58.9113/58.4691/57.5534/54.2121，配对差+0.1417/+0.2674/+1.9247/+2.0052pp。

解释边界：25%与10%在两场各三seed均正，与开发game6低标注方向一致，是这套整体配方的重复观察证据；50%存在负seed，100%跨比赛方向相反（game8三正、game9三负；开发game6三负），不概括为普遍提升/稳定副作用或仅凭差值大小贴结论。25%混合跨场均值57.5534仍低于100%纯R58.7696约1.2162pp；不拿弱长R作基线，不声称节省75%/90%标注、等效或非劣效PASS，用户未定允许损失界限。R24000R vsRS12000R12000S同总24000，不能分离纯合成机制因果。只有两场，不把相关帧/clip/训练seed当独立比赛；game9已观察和game8先前下载/GT结构访问史保留，不称首次盲测、不按test选方法。

全部授权有限24训练和48确认及小control/owned现已闭合，不追加训练/seed/比例/型号/来源/test范围。只读观察台保留至原Nov7T00:00:41.387415Z，g/UE生产及另聊天Seg/Pose不动；本bbox跟进将按既定完成规则原生删除，删除实际结果另记。EXPERIMENTS32c941.../PAPERd48cb始末SHA保持，不改/重封方案。

完整固定统计（百分值/百分点）：
[{"scope":"confirmation_game8","real_fraction":1.0,"N_R":2100,"metrics":{"AP50_95":{"R_mean_percent":60.70247148759535,"R_sample_SD_pp":0.1778230015675639,"RS_mean_percent":61.32582715761774,"RS_sample_SD_pp":0.2983931071573317,"paired_RS_minus_R_mean_pp":0.6233556700223852,"paired_sample_SD_pp":0.17604872412412245,"paired_seed_values_pp":[0.5478373130778591,0.4976648971669384,0.8245647998223582]},"AP50":{"R_mean_percent":95.03529939947657,"R_sample_SD_pp":0.08317620657499214,"RS_mean_percent":95.34192661415976,"RS_sample_SD_pp":0.21873570696765035,"paired_RS_minus_R_mean_pp":0.3066272146831861,"paired_sample_SD_pp":0.26898079178411666,"paired_seed_values_pp":[0.4915311057652527,0.4302969470133089,-0.0019464087290033127]},"AP75":{"R_mean_percent":69.5880287680749,"R_sample_SD_pp":0.39632577384765205,"RS_mean_percent":70.60066042836071,"RS_sample_SD_pp":0.6579353878334813,"paired_RS_minus_R_mean_pp":1.0126316602858048,"paired_sample_SD_pp":0.7260474979689148,"paired_seed_values_pp":[1.298543258596896,0.18715414882337633,1.5521975734371418]}}},{"scope":"confirmation_game8","real_fraction":0.5,"N_R":1050,"metrics":{"AP50_95":{"R_mean_percent":60.259785965808256,"R_sample_SD_pp":0.09920040865125607,"RS_mean_percent":60.66206261791106,"RS_sample_SD_pp":0.6004209049791472,"paired_RS_minus_R_mean_pp":0.4022766521028065,"paired_sample_SD_pp":0.6352756272040105,"paired_seed_values_pp":[-0.31437726175148617,0.6250289125064423,0.8961783055534633]},"AP50":{"R_mean_percent":94.92217273906869,"R_sample_SD_pp":0.17494915485781165,"RS_mean_percent":95.27941107566791,"RS_sample_SD_pp":0.2814964582712857,"paired_RS_minus_R_mean_pp":0.3572383365992156,"paired_sample_SD_pp":0.45641596573216403,"paired_seed_values_pp":[-0.16969229425951582,0.6121236478130565,0.6292836562441062]},"AP75":{"R_mean_percent":68.60870288760171,"R_sample_SD_pp":0.10225064583036622,"RS_mean_percent":69.80547068651437,"RS_sample_SD_pp":0.6369897274362446,"paired_RS_minus_R_mean_pp":1.196767798912653,"paired_sample_SD_pp":0.6438501387874378,"paired_seed_values_pp":[0.520149992526242,1.268272917880009,1.8018804863317084]}}},{"scope":"confirmation_game8","real_fraction":0.25,"N_R":525,"metrics":{"AP50_95":{"R_mean_percent":57.80126918958614,"R_sample_SD_pp":0.1934247452673513,"RS_mean_percent":59.43399409210693,"RS_sample_SD_pp":0.19978554026305112,"paired_RS_minus_R_mean_pp":1.6327249025207917,"paired_sample_SD_pp":0.3571512238647289,"paired_seed_values_pp":[1.3876954446687684,2.042515492417807,1.4679637704757997]},"AP50":{"R_mean_percent":93.38645450946981,"R_sample_SD_pp":0.14976198025775309,"RS_mean_percent":94.07048653147413,"RS_sample_SD_pp":0.12111005474320216,"paired_RS_minus_R_mean_pp":0.684032022004318,"paired_sample_SD_pp":0.12391585231060112,"paired_seed_values_pp":[0.8133096230916292,0.5662837604702209,0.6725026824511038]},"AP75":{"R_mean_percent":64.99758460787764,"R_sample_SD_pp":0.656370131108571,"RS_mean_percent":67.83663512670921,"RS_sample_SD_pp":0.37333713822713477,"paired_RS_minus_R_mean_pp":2.839050518831565,"paired_sample_SD_pp":0.9948711106550148,"paired_seed_values_pp":[1.7892690183839655,3.7679680352533884,2.9599145028573415]}}},{"scope":"confirmation_game8","real_fraction":0.1,"N_R":210,"metrics":{"AP50_95":{"R_mean_percent":54.07457714809283,"R_sample_SD_pp":0.4011033530944994,"RS_mean_percent":56.19551147413721,"RS_sample_SD_pp":0.17488762369450442,"paired_RS_minus_R_mean_pp":2.1209343260443845,"paired_sample_SD_pp":0.5683340340969596,"paired_seed_values_pp":[1.9714881052841582,1.6422561853974615,2.7490586874515333]},"AP50":{"R_mean_percent":91.5224158256466,"R_sample_SD_pp":0.14847001352535363,"RS_mean_percent":92.00997568826429,"RS_sample_SD_pp":0.20526850908780053,"paired_RS_minus_R_mean_pp":0.48755986261768136,"paired_sample_SD_pp":0.08190809589930977,"paired_seed_values_pp":[0.40654585608392857,0.5703336877767029,0.4858000439924126]},"AP75":{"R_mean_percent":58.64186248253545,"R_sample_SD_pp":0.8330129460045699,"RS_mean_percent":62.093637531698,"RS_sample_SD_pp":0.2656481614181422,"paired_RS_minus_R_mean_pp":3.4517750491625496,"paired_sample_SD_pp":0.8760000136249334,"paired_seed_values_pp":[2.9325873777059086,2.959564957509231,4.463172812272509]}}},{"scope":"confirmation_game9","real_fraction":1.0,"N_R":2100,"metrics":{"AP50_95":{"R_mean_percent":56.83672877394371,"R_sample_SD_pp":0.28367647542060437,"RS_mean_percent":56.49674654676644,"RS_sample_SD_pp":0.12071890263774357,"paired_RS_minus_R_mean_pp":-0.3399822271772712,"paired_sample_SD_pp":0.2653290019534394,"paired_seed_values_pp":[-0.04763183926819181,-0.40679800269476374,-0.5655168395688581]},"AP50":{"R_mean_percent":95.00033467817964,"R_sample_SD_pp":0.3131394461438262,"RS_mean_percent":94.6963758302907,"RS_sample_SD_pp":0.0647678660738407,"paired_RS_minus_R_mean_pp":-0.303958847888941,"paired_sample_SD_pp":0.3669223984076495,"paired_seed_values_pp":[-0.07297881282813945,-0.11184812743956218,-0.7270496033991214]},"AP75":{"R_mean_percent":61.27207177681877,"R_sample_SD_pp":0.4359434108267115,"RS_mean_percent":60.33284415542446,"RS_sample_SD_pp":0.24599257721233075,"paired_RS_minus_R_mean_pp":-0.9392276213943082,"paired_sample_SD_pp":0.5682457836524561,"paired_seed_values_pp":[-0.43315845655342855,-0.8305665155120878,-1.553957892117408]}}},{"scope":"confirmation_game9","real_fraction":0.5,"N_R":1050,"metrics":{"AP50_95":{"R_mean_percent":56.14361332860121,"R_sample_SD_pp":0.08984848937359234,"RS_mean_percent":56.27618751973724,"RS_sample_SD_pp":0.6777731290368029,"paired_RS_minus_R_mean_pp":0.13257419113602564,"paired_sample_SD_pp":0.7615694175889159,"paired_seed_values_pp":[-0.38480686897092165,1.0070776853179586,-0.22454824293896003]},"AP50":{"R_mean_percent":94.74989275344194,"R_sample_SD_pp":0.14792253082699183,"RS_mean_percent":94.7146224864012,"RS_sample_SD_pp":0.3288646500558247,"paired_RS_minus_R_mean_pp":-0.03527026704074343,"paired_sample_SD_pp":0.2552204169133506,"paired_seed_values_pp":[-0.22601180724264314,0.2546530919416701,-0.13445208582125723]},"AP75":{"R_mean_percent":60.19849557214323,"R_sample_SD_pp":0.45650512978167007,"RS_mean_percent":60.329940295765255,"RS_sample_SD_pp":0.8973529003713857,"paired_RS_minus_R_mean_pp":0.13144472362201695,"paired_sample_SD_pp":1.3062977924650498,"paired_seed_values_pp":[-0.17843435501133653,1.5648189073644736,-0.9920503814870862]}}},{"scope":"confirmation_game9","real_fraction":0.25,"N_R":525,"metrics":{"AP50_95":{"R_mean_percent":53.456112674941814,"R_sample_SD_pp":0.39072048279839666,"RS_mean_percent":55.672871909675735,"RS_sample_SD_pp":0.3482491774330806,"paired_RS_minus_R_mean_pp":2.216759234733915,"paired_sample_SD_pp":0.2848080809700068,"paired_seed_values_pp":[1.9256525616659026,2.229804753923914,2.494820388611929]},"AP50":{"R_mean_percent":93.31503302072967,"R_sample_SD_pp":0.2618048888419893,"RS_mean_percent":94.15843408094155,"RS_sample_SD_pp":0.45034223229823206,"paired_RS_minus_R_mean_pp":0.8434010602118791,"paired_sample_SD_pp":0.6166245756701867,"paired_seed_values_pp":[0.3658900930421982,0.6247581130003899,1.5395549745930492]},"AP75":{"R_mean_percent":55.2710771861062,"R_sample_SD_pp":0.6714024157777604,"RS_mean_percent":59.02662814974989,"RS_sample_SD_pp":0.6063336567166953,"paired_RS_minus_R_mean_pp":3.7555509636436946,"paired_sample_SD_pp":0.37069015114668513,"paired_seed_values_pp":[4.09864825937003,3.805642502583617,3.3623621289774364]}}},{"scope":"confirmation_game9","real_fraction":0.1,"N_R":210,"metrics":{"AP50_95":{"R_mean_percent":50.339181645479755,"R_sample_SD_pp":0.20907721985001346,"RS_mean_percent":52.22868876476583,"RS_sample_SD_pp":0.3944074821195168,"paired_RS_minus_R_mean_pp":1.8895071192860713,"paired_sample_SD_pp":0.43462402391956206,"paired_seed_values_pp":[2.1523312208370227,1.387837508289358,2.1283526287318333]},"AP50":{"R_mean_percent":91.61444017904329,"R_sample_SD_pp":0.12276390358457097,"RS_mean_percent":93.1284973325521,"RS_sample_SD_pp":0.20777371749482387,"paired_RS_minus_R_mean_pp":1.5140571535088225,"paired_sample_SD_pp":0.15208009115027307,"paired_seed_values_pp":[1.668571751615111,1.3645344868629339,1.5090652220484224]},"AP75":{"R_mean_percent":49.85892524047309,"R_sample_SD_pp":0.38324561453943073,"RS_mean_percent":52.2837094600835,"RS_sample_SD_pp":0.6480814760279412,"paired_RS_minus_R_mean_pp":2.424784219610408,"paired_sample_SD_pp":1.0307362360003847,"paired_seed_values_pp":[2.815126313528138,1.2558873511334,3.2033389941696857]}}},{"scope":"confirmation_equal_match_mean","real_fraction":1.0,"N_R":2100,"metrics":{"AP50_95":{"R_mean_percent":58.76960013076953,"R_sample_SD_pp":0.22976682563303868,"RS_mean_percent":58.91128685219209,"RS_sample_SD_pp":0.13828300905550356,"paired_RS_minus_R_mean_pp":0.14168672142256256,"paired_sample_SD_pp":0.10287530675652919,"paired_seed_values_pp":[0.25010273690484475,0.04543344723608733,0.1295239801267556]},"AP50":{"R_mean_percent":95.0178170388281,"R_sample_SD_pp":0.18361537774476008,"RS_mean_percent":95.01915122222522,"RS_sample_SD_pp":0.13482971918884204,"paired_RS_minus_R_mean_pp":0.0013341833971169947,"paired_sample_SD_pp":0.31780683948065597,"paired_seed_values_pp":[0.20927614646855108,0.15922440978686225,-0.36449800606406235]},"AP75":{"R_mean_percent":65.43005027244683,"R_sample_SD_pp":0.40888996096363095,"RS_mean_percent":65.46675229189259,"RS_sample_SD_pp":0.20819013701496816,"paired_RS_minus_R_mean_pp":0.036702019445751986,"paired_sample_SD_pp":0.3786008731436189,"paired_seed_values_pp":[0.4326924010217392,-0.3217061833443502,-0.0008801593401330798]}}},{"scope":"confirmation_equal_match_mean","real_fraction":0.5,"N_R":1050,"metrics":{"AP50_95":{"R_mean_percent":58.201699647204734,"R_sample_SD_pp":0.023817632574561952,"RS_mean_percent":58.469125068824155,"RS_sample_SD_pp":0.5669802962953201,"paired_RS_minus_R_mean_pp":0.2674254216194198,"paired_sample_SD_pp":0.5858243190991315,"paired_seed_values_pp":[-0.3495920653612039,0.8160532989122005,0.33581503130726276]},"AP50":{"R_mean_percent":94.83603274625531,"R_sample_SD_pp":0.03585793172657285,"RS_mean_percent":94.99701678103455,"RS_sample_SD_pp":0.29759850142745603,"paired_RS_minus_R_mean_pp":0.1609840347792361,"paired_sample_SD_pp":0.3243747104366683,"paired_seed_values_pp":[-0.19785205075107948,0.4333883698773633,0.2474157852114245]},"AP75":{"R_mean_percent":64.40359922987247,"R_sample_SD_pp":0.18354620235659944,"RS_mean_percent":65.0677054911398,"RS_sample_SD_pp":0.5821388725427498,"paired_RS_minus_R_mean_pp":0.6641062612673387,"paired_sample_SD_pp":0.6620572247593359,"paired_seed_values_pp":[0.17085781875745276,1.4165459126222468,0.4049150524223166]}}},{"scope":"confirmation_equal_match_mean","real_fraction":0.25,"N_R":525,"metrics":{"AP50_95":{"R_mean_percent":55.62869093226398,"R_sample_SD_pp":0.2709012532594605,"RS_mean_percent":57.55343300089133,"RS_sample_SD_pp":0.12968495739958363,"paired_RS_minus_R_mean_pp":1.9247420686273535,"paired_sample_SD_pp":0.24471136594289872,"paired_seed_values_pp":[1.6566740031673355,2.1361601231708605,1.9813920795438644]},"AP50":{"R_mean_percent":93.35074376509974,"R_sample_SD_pp":0.14427952526548388,"RS_mean_percent":94.11446030620783,"RS_sample_SD_pp":0.1695575487123146,"paired_RS_minus_R_mean_pp":0.7637165411081023,"paired_sample_SD_pp":0.29646591942260386,"paired_seed_values_pp":[0.5895998580669137,0.5955209367353165,1.1060288285220765]},"AP75":{"R_mean_percent":60.134330896991926,"R_sample_SD_pp":0.6145991528120128,"RS_mean_percent":63.431631638229554,"RS_sample_SD_pp":0.20753122653953743,"paired_RS_minus_R_mean_pp":3.2973007412376263,"paired_sample_SD_pp":0.4376102889499966,"paired_seed_values_pp":[2.943958638876998,3.7868052689185028,3.161138315917378]}}},{"scope":"confirmation_equal_match_mean","real_fraction":0.1,"N_R":210,"metrics":{"AP50_95":{"R_mean_percent":52.20687939678629,"R_sample_SD_pp":0.3030348544625033,"RS_mean_percent":54.21210011945152,"RS_sample_SD_pp":0.2533431574002181,"paired_RS_minus_R_mean_pp":2.0052207226652277,"paired_sample_SD_pp":0.4644315093439616,"paired_seed_values_pp":[2.0619096630605904,1.5150468468434042,2.438705658091689]},"AP50":{"R_mean_percent":91.56842800234494,"R_sample_SD_pp":0.1281669500296833,"RS_mean_percent":92.56923651040819,"RS_sample_SD_pp":0.12671584476940345,"paired_RS_minus_R_mean_pp":1.0008085080632556,"paired_sample_SD_pp":0.03518403567417408,"paired_seed_values_pp":[1.0375588038495254,0.9674340873198184,0.9974326330204231]},"AP75":{"R_mean_percent":54.25039386150427,"R_sample_SD_pp":0.5904578411848111,"RS_mean_percent":57.188673495890754,"RS_sample_SD_pp":0.27412450895664525,"paired_RS_minus_R_mean_pp":2.9382796343864794,"paired_sample_SD_pp":0.864566914917035,"paired_seed_values_pp":[2.8738568456170177,2.107726154321321,3.8332559032211]}}},{"scope":"development_game6","real_fraction":1.0,"N_R":2100,"metrics":{"AP50_95":{"R_mean_percent":55.16856589695974,"R_sample_SD_pp":0.31193904214280577,"RS_mean_percent":54.25133650764026,"RS_sample_SD_pp":0.2287898308487992,"paired_RS_minus_R_mean_pp":-0.9172293893194813,"paired_sample_SD_pp":0.24323009890114677,"paired_seed_values_pp":[-1.1187688606661639,-0.6470569909503121,-0.9858623163419677]},"AP50":{"R_mean_percent":95.65852977050467,"R_sample_SD_pp":0.2264876190729209,"RS_mean_percent":96.00095652402548,"RS_sample_SD_pp":0.13132478215723614,"paired_RS_minus_R_mean_pp":0.3424267535208229,"paired_sample_SD_pp":0.35537954006406136,"paired_seed_values_pp":[-0.018274445486843316,0.3533205701212627,0.6922341359280493]},"AP75":{"R_mean_percent":56.81069768582531,"R_sample_SD_pp":0.9527142962502857,"RS_mean_percent":54.4341700948991,"RS_sample_SD_pp":0.2956586478176204,"paired_RS_minus_R_mean_pp":-2.3765275909262105,"paired_sample_SD_pp":0.8645156480585787,"paired_seed_values_pp":[-2.735940105056145,-1.390282872749038,-3.003359794973448]}}},{"scope":"development_game6","real_fraction":0.5,"N_R":1050,"metrics":{"AP50_95":{"R_mean_percent":53.916959870480014,"R_sample_SD_pp":0.2983789170425418,"RS_mean_percent":54.07582747066943,"RS_sample_SD_pp":0.4770550798810996,"paired_RS_minus_R_mean_pp":0.15886760018941257,"paired_sample_SD_pp":0.20328916709820705,"paired_seed_values_pp":[0.29612384214285914,-0.07467573686240625,0.2551546952877848]},"AP50":{"R_mean_percent":95.2349699271322,"R_sample_SD_pp":0.05958273245707939,"RS_mean_percent":95.86788101632982,"RS_sample_SD_pp":0.3079773918556446,"paired_RS_minus_R_mean_pp":0.6329110891976261,"paired_sample_SD_pp":0.34469039652358724,"paired_seed_values_pp":[0.2515269320575997,0.7250077167187619,0.9221986188165165]},"AP75":{"R_mean_percent":53.99146752899965,"R_sample_SD_pp":0.8128762812178804,"RS_mean_percent":54.07631099786847,"RS_sample_SD_pp":1.1566614644047581,"paired_RS_minus_R_mean_pp":0.08484346886881895,"paired_sample_SD_pp":0.39626008595317647,"paired_seed_values_pp":[-0.01189202661675548,-0.25409194456125217,0.5205143777844645]}}},{"scope":"development_game6","real_fraction":0.25,"N_R":525,"metrics":{"AP50_95":{"R_mean_percent":51.60334496423032,"R_sample_SD_pp":0.17749235950532335,"RS_mean_percent":52.86127621302555,"RS_sample_SD_pp":0.45831625182382624,"paired_RS_minus_R_mean_pp":1.2579312487952337,"paired_sample_SD_pp":0.5716616337265773,"paired_seed_values_pp":[0.6251739566370151,1.7371309020889791,1.4114888876597065]},"AP50":{"R_mean_percent":94.18646939750495,"R_sample_SD_pp":0.589595161767782,"RS_mean_percent":95.38862769396759,"RS_sample_SD_pp":0.44728306568462356,"paired_RS_minus_R_mean_pp":1.2021582964626447,"paired_sample_SD_pp":0.8943425975243995,"paired_seed_values_pp":[0.1817780608170172,1.574628362097985,1.8500684664729317]},"AP75":{"R_mean_percent":49.80063337400998,"R_sample_SD_pp":0.2867943157610599,"RS_mean_percent":51.613418726229824,"RS_sample_SD_pp":1.3012182978141937,"paired_RS_minus_R_mean_pp":1.812785352219844,"paired_sample_SD_pp":1.1906204516534584,"paired_seed_values_pp":[0.5843788933617033,2.961628870789368,1.8923482925084611]}}},{"scope":"development_game6","real_fraction":0.1,"N_R":210,"metrics":{"AP50_95":{"R_mean_percent":48.35702469687874,"R_sample_SD_pp":0.23737064514088482,"RS_mean_percent":50.51073483148636,"RS_sample_SD_pp":0.30622589353178187,"paired_RS_minus_R_mean_pp":2.1537101346076226,"paired_sample_SD_pp":0.5384538944338886,"paired_seed_values_pp":[1.5454598400925557,2.3462275722400063,2.5694429914903063]},"AP50":{"R_mean_percent":91.94511079883955,"R_sample_SD_pp":0.046951596848352734,"RS_mean_percent":93.95301024495583,"RS_sample_SD_pp":0.28798530956957114,"paired_RS_minus_R_mean_pp":2.0078994461162902,"paired_sample_SD_pp":0.24475397419139897,"paired_seed_values_pp":[2.121520115236064,1.7269858640474367,2.17519235906537]},"AP75":{"R_mean_percent":44.70023299999468,"R_sample_SD_pp":0.11212637621153801,"RS_mean_percent":47.7284995191083,"RS_sample_SD_pp":1.1900081485481109,"paired_RS_minus_R_mean_pp":3.028266519113613,"paired_sample_SD_pp":1.2012285243993759,"paired_seed_values_pp":[1.6442616803510002,3.799942789550065,3.640595087439774]}}}]

BBOX-CONFIRMATION-FINAL-001收尾：UTC 2026-11-05T03:51:22.386197+00:00，原生automation_update(mode=delete,id=bbox)实际返回deleteStatus=deleted，本机bbox automation.toml已不存在；仅删除本有限跟进，g未操作，只读观察台仍原截止，不自动新增实验。


## BBOX-REPOSITORY-CLEANUP-001：用户明确阶段收尾、删除非正式产物并提交

2026-11-05用户明确“bbox实验暂时到此为止，清理仓库，清点值得纳入的训练记录，删除非正式实验记录和内容并提交一次Git”。此新直接授权覆盖旧弃用S3000开发诊断继续保留原始产物的条款，仅限本次精确清单；不授权删除正式负结果、GT、公开基座、共享缓存或Seg/Pose/UE内容。实际Git根为D:/projects/FutsalMOT_SyntheticDatasetResearch，UE工程独立且有大量无关WIP，本次不暂存或清理UE文件。研究库当前未配置remote，只本地提交，不创建远端/推送。

正式索引docs/exp/BBOX_RESULTS.md、机器清单bbox_formal_registry.json已逐已有收据清点：主表24项1500步、6项正式事前批准策略/预算对照、2项端点预算校准；另48比赛外确认及训练/开发域差分析。主表/辅助/校准按不同预算和观察范围单列，保留所有正负，不将正式失败/中断/准入小收据因非COMPLETE而删除。完整48绝对AP50:95/AP50/AP75、配对差/样本SD/同模型两场等权与图作为小结果原样复制进docs/exp；原模型、native预测和完整运行证据仍在实际D/G组，不提交Git。

精确删除清单runs/bbox_cleanup_20261105_a01/delete_manifest.json：54目标、2398文件、14737462758逻辑字节（约13.73GiB）；PowerShell原生Remove-Item -LiteralPath完成。执行前核最终D/G绝对路径、全树无junction/reparse/活动writer，逐文件stat大小；不读/解码/全hashPT/NPZ。主要是方案明确弃用的S3000→R截断前缀开发诊断、其准入临时组、无引用临时预检step/stdout/stderr、旧UI v7/v8日志和未进入正式数值路径的CPU筛选微基准。旧前缀D组只保留正式budget1500/short_pair/fixed_mix依赖的40不可变源码祖先及code_bindings，附SOURCE_ONLY，不留旧job_table/启动器/权重或预测；不是隐藏可运行archive。冻结正式源码和所有正式原始结果未改，科学三方案SHA不变。

README/AGENTS改为当前阶段结束与正式索引，TODO移除已过期BBox RUNNING/授权接力；6个Seg/Pose相关独立待办块原文保留。唯一日志删除36个非正式准备/CPU/安装观察/弃用诊断章节，11个Seg/Pose独立日志块逐原文保留；不新建history或平行报告树。保留的正式历史负结果与失败小收据仍可审阅；本条和bbox_cleanup_20261105.json登记退休范围，不能把已删微基准或诊断称为仍完整留存。

清理后独立核280个正式小引用、主表24/辅助6/校准2/48确认summary和全部主表SEALED小SHA；40依赖源码祖先仍可读取、82个根Python源码语法编译不执行，原PT只stat不全hash。必要确认CPU回归45 PASS/0skip/37.116秒，虚构CPU夹具，无真实GT/媒体/权重/CUDA，不重训或重跑48评价。本次45检查属于删除依赖后的代码回归，不是恢复逐模型训练预检。Git忽略环境/运行/模型/缓存/本机服务与本地PDF；.gitattributes禁自动换行转换以保留科学SHA字节。Seg/Pose未提交源码和资料、UE WIP保留；旧v1架构文档/12点schema等已在工作区淘汰的跟踪删除同步纳入本次仓库收敛，不恢复旧入口。实际commit与暂存检查完成后登记到本机清理收据，避免自引用commit哈希。

## BBOX-MONITOR-RESTORE-002：系统重启及Windows保留端口后的只读观察台恢复

2026-11-05用户报告训练监控无法打开。只读核验显示系统本次开机为UTC08:06:02（北京16:06），上午09:43启动的8765/8766实际服务与wrapper四个身份均已退出，旧登记仍在。随后原8765恢复在socket.bind报WinError10013；netsh只读查询确认Windows TCP排除范围8712–8811同时覆盖8765/8766，无监听者，不能将其归因于实验清理或数据丢失。

改用18765，只读服务沿原绝对截止2026-11-07T00:00:41.387415Z；没有修改系统端口保留/防火墙。页面评价请求改用训练监控原已支持的同源/api/confirmation，因而不再依赖单独8766接口；更新打开训练监控.cmd及README地址。训练/评价数值源、正式冻包、GT与结果不改，不重新执行实验。原8765失败日志与新派发/复核保留runs/bbox_monitor_restore_20261105_a01。

新实际服务PID9892/CT1793882918.9151495、wrapper24700，完整argv绑定tools/training_monitor.py --port18765及原deadline；真实页面、JS、health、status、confirmation HTTP200，确认48/48、36000图、errors[]。13项HTTP/确认监视CPU测试通过（0.459s），原EXPERIMENTS/PAPER SHA未变。恢复收据verification.json保留；不承诺系统重启后进程继续存在，不增加自启动或延长服务。
