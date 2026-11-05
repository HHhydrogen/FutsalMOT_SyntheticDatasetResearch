# 当前待办

## BBox：阶段结束并归整正式证据

- [x] 24主表训练和48两比赛确认已完整终验；原29/新19执行来源与全部正负结果保留，有限bbox自动跟进已删除。
- [x] 清点24主表、6正式辅助对照、2预算校准；完整清单与绝对结果见[正式结果索引](../exp/BBOX_RESULTS.md)。
- [x] 按用户授权删除弃用S3000诊断和无引用临时记录；保留必要冻结代码祖先、正式准入/失败/中断小收据、原GT和模型预测。
- [x] 本次BBox/共享依赖与小结果已通过引用、源码与45项CPU回归检查，按逐文件清单纳入阶段收尾Git提交；无远端不推送。
- [ ] 后续BBox训练或评价等用户新指令，不自动扩范围。

## 其它聊天职责

以下保留既有Seg/Pose独立条目原文，仅供各负责聊天接续；本聊天不推进或启动其工作。混合标题中的旧BBox运行状态不构成当前待办。

## 职责交接：原聊天仅BBox，Seg/Pose新聊天首轮只读了解

- [x] 用户明确将Seg与Pose分别交给两个新聊天；已实际创建并触发，各提示含完整现状/证据/缺口和首轮只读复述限制。
- [x] 两新聊天均已完成首轮只读复述并停止等待用户；没有开始准备/测试/训练。后续Seg/Pose待办归各新聊天由用户安排，原BBox聊天不自行推进。
- [x] BBox a02固定48项评价和有限跟进归原聊天；共享GPU互斥/原方案/原GT/冻结代码保持。交接不修改已有科学方案或授权训练。

## 最新：三seed BBox结果已收口；Seg按用户要求搁置

- [x] seed17/29/43×100/50/25/10%×R/混合共24个1500步末EMA已核；新16批量全部收口，原owned/两锁退出，批量小control30文件211714B SEALED。
- [x] 三seed平均混合−R主AP：100%−0.917229、50%+0.158868、25%+1.257931、10%+2.153710pp；25/10%各三seed均正，50%两正一负。完整绝对表/标准差/图在G原batch，唯一日志BBOX-CURVE3-RESULT-001。
- [ ] 原批准24模型×game8/game9共48独立评价尚未派发；先完成新评价执行器/24registry/spec/来源和实际资源Gate。现有bbox跟进PAUSED保持，不自动恢复或增加训练。
- [ ] Seg任务由最新用户明确搁置：停止后续CPU准备、GPU预检和训练；全部已有原GT/RLE候选/代码与证据保留，待用户直接恢复。

## Seg原PNG/RLE与主训练池推进（SEG-PNG-RLE-009 / IHELON-COMMON-VIEW-009）

- [x] 官方SDK确认save是part色、同实例可含多part；全19071色号核验、200原PNG全RGBA无损RLE往返完成，原GT保留。
- [x] Acme人物83帧/406实例严格恢复（唯一归属+原area/bbox/外轮廓exact），17帧保留未定；逐像素补充新增exact0。孔洞可恢复，不等于原100实例全恢复。
- [x] 排除17未定和1/74重复对、86边缘漏标后，Acme80图/383人物RLE评价候选+原GT血缘已备，未正式激活/未评价，保留全部原100和20排除理由。
- [x] ihelon共同508图/7620人物文件/UID/血缘与原适配器identity覆盖已备：54目标过滤、另52proto消失，共82图。原508完整保留。
- [x] 另备426图/6107人物静态identity监督安全候选，整图hold82、不删单个GT；分布会偏向较易监督目标，尚未替换主训练集或启用split。
- [ ] 冻结共同508+合法过滤/ignore，或采用426最小候选的范围；同步冻结随机几何裁边/微小实例监督规则，不能将identity结果当全部增强通过。
- [ ] 明确来源/比赛级train/dev/confirmation用途与Acme角色映射/真实MaskAP接线；不能逐帧随机拆ihelon或把Acme同场80图当多场独立证据。
- [ ] 合成1600固定UID、完整500步生产提交/STOP/恢复/native/partial/owned、公开s/960实际GPU/空间时间仍未准入；本次继续不训练/不GPU预检。

## Seg真实数据确认（SEG-ACME-FULL-008 / SEG-TRAIN-SOURCE-008）

- [x] Acme固定v1完整303源文件219213825B已取得；100图/915标注/511人物全量CPU审计，全部100原图+GT概览和关键疑难放大已查看，无Seg模型/GPU。
- [x] 比赛指向2017-08-08皇马—曼联欧洲超级杯；原PDF与全100画面一致。对ihelon512/Quantigo847共135900对未检出精确低分辨率/pHash<=8重复，保留原YouTube元数据不可读及旧源完整比赛链未知的局限。
- [x] 定位同RGB异GT image1/74、image43门将mask含球/PNG孔洞冲突、image86边缘手臂未标。全511内区对fuse诊断175实例/85图有>=50px连通差异（仅排查阈值，不等于175坏标注）。Acme未准入正式实例评价；仅去重对后98图/497人物也不是质量PASS。
- [x] 主训练优先ihelon：508图/7620人物整图隔离候选UID已记录，未激活；原512/GT与RLE保持。321/451同图异GT、384/5904碎片、287/4261遮挡疑难整图保守hold，4261未被判为错误。Quantigo仍备用。
- [ ] 解决Acme原instance-pixel ID/孔洞的可信RLE表示、裁边/遮挡范围与重复GT版本裁定；不能用fuse类色猜实例、删背景/单人或自动修GT绕过。
- [ ] 冻结所有方法共用GT/极小监督规则与比赛/源级train/dev/评价用途；ihelon一保守组不能逐帧随机拆，尚无正式真实Seg数据准入。
- [x] 原来源/完整审计/两份新决策收据及精确诊断清理登记在唯一日志；EXPERIMENTS/PAPER/DATASET与活动BBox42保持。不启动Seg训练/推理/GPU预检。

## Seg本次CPU推进（SEG-READINESS-006；未开训）

- [x] 新只读来源/GT约束核验器：实际1359 UID/2保守组、8精确重复、4疑难标注、同RGB异GTmust-link全部追溯；返回未准入。6项CPU检查通过。
- [x] 原生输出NMS、记录LetterBox逆映射、逐实例有界mask/RLE/原图Mask AP和native+SHA证据候选；10项新CPU检查通过，零模型/优化器/GPU。
- [ ] 原两源完整比赛来源链仍未全核；Acme原PDF与全100画面已完成来源核验，平台许可声明与上游链区别记录。正式split与独立确认分组待定。
- [ ] Q4482/5895仍待裁定；ihelon5904/4261上下文已查，508图整图hold候选已记录待正式共同视图，不能将候选当准入或删单个人作背景。原GT不改。
- [ ] 共用最终GT视图、监督覆盖/极小实例规则、正式逆几何/NMS/Mask与BBox评价口径待冻结。
- [ ] 合成1600帧实际清单、500步生产生命周期、公开s/960真实全路径与完整资源仍未准入；保持不训练/不GPU预检。

# TODO与工作顺序

固定16组批量a02已于北京10:35:57唯一启动：G bbox_label_curve_batch_20261104_a02，CPU supervisor12512/CT1793759757.6260707、wrapper15864/CT1793759757.6090045。首项seed29/N100 R实际RUNNING，D/G main_bbox_yolo11s_soccernet_R_seed29_20261104_a02，controller27912/CT1793759774.6515486；runtime {"started_at_utc": "2026-11-04T02:36:23.680750+00:00", "deadline_unix": 1793765183.6807508, "max_run_seconds": 5400}，0为PT0验证。队列自验收衔接seed29再43各四档R/混合，无需聊天逐项启动；复用已验证配置，新诊断0/新CPU和GPU预检false，原42/1500/90min/3cp2val/完整记录不变。轻量互斥/空间/截止/完工检查保留；任一失败/STOP停队列，不自动retry/resume。首a01为JSON元组/列表准备差异零派发FAIL，原字节保留，a02只做一次明确机械修订。方案SHA32c9415b2d81de7fce6a9221af72495822ffe3b70e537004b7bb64af02b065a2，PAPER原d48cb；旧5dcb冻结合法，Seg/Pose不训练/GPU。


最新用户直接授权批量执行seed29/43全部16组，取消逐项重复CPU测试/恢复探针/GPU机械预检；复用已验证同配置的真实证据，使用独立复用准入schema明示fresh_cpu_tests=false/fresh_gpu_probe=false/diagnostic_updates=0，不冒称新测量PASS。保留原42训练数值/完整记录/公开独立起点、单GPU互斥、存储/截止和前项完整收据闭合；任一失败/人工STOP即停队列，无自动retry/resume。EXPERIMENTS新SHA32c9415b2d81de7fce6a9221af72495822ffe3b70e537004b7bb64af02b065a2，PAPER d48cb原值，旧5dcb run合法历史原bytes不回写。N10混合已于北京10:16:48 COMPLETE1500/R12000S12000并root终验，主AP50.1585%/+1.54546pp；seed17四新增项全部闭合，curve4_completion SHA024a4606b7a24aa1bae1fb3fa955fbbaeb34fbb55f444d530e975dabedb3a064/实际准备214.19832253456116含旧FAIL。批量工具当前正在实现，尚未seed29派发，不运行旧逐项预检入口。


10%纯R已于北京09:47:27 COMPLETE1500/R24000S0，root终验3cp/2val/51小control818847B SEALED/owned退出/锁退，末AP50:95 48.6130%。同210真实图的10%混合于北京09:51:04唯一启动D/G main_bbox_yolo11s_soccernet_RS_seed17_20261104_a02，controller7196/CT1793757064.340304、wrapper13608/CT1793757064.3224485；runtime {"started_at_utc": "2026-11-04T01:51:13.171241+00:00", "deadline_unix": 1793762473.1712415, "max_run_seconds": 5400}，初始0为PT0非训练。独立机械PASS33.063s/全36.4252002239s，四项含旧FAIL总214.1983225346/240，原额度未增加。待本混合全收口后才写四项完工+失败历史receipt并开始seed29/43各8共16，之后24末EMA两比赛48确认；原Nov7截止/Gate，Seg/Pose不训练/GPU。方案5dcb/PAPERd48cb原bytes保持。


观察台记录筛选已上线：刷新localhost8765后，可按用途/任务/模型/来源/策略/seed/真实标注量/总更新预算/状态组合筛选，支持关键词、数量、清空与浏览器记忆；当前所选筛选外记录保留。21客户端检查与实际浏览器交互通过。仅更新静态界面，服务PID/截止与活动训练/42源/封存方案不变，见MONITOR-FILTER-001。


50%标注配对已完整验收：R53.9112%、R+S54.2074%，混合主AP+0.2961pp，仍低于全量R0.9561pp；同1050池/1500步。10%纯R(210图)于北京09:22:17唯一启动D/G main_bbox_yolo11s_soccernet_R_seed17_20261104_a03，controller11576/CT1793755337.0921447、wrapper26308/CT1793755337.0750775；runtime {"started_at_utc": "2026-11-04T01:22:25.805835+00:00", "deadline_unix": 1793760745.8058355, "max_run_seconds": 5400}，初始0是PT0。N10独立机械PASS46.953s/全50.4237632751s，含旧FAIL本4累计177.7731223106/240；余62.2268776894s，仅足下一项60s许可，不加预算。随后10混合→seed29/43各8→24模型两场48确认，原Gate/Nov7截止，Seg/Pose不训练/GPU，方案5dcb原bytes保持。


当前50%纯R已于北京08:31:14 COMPLETE1500/R24000S0并终验，末AP50:95 53.9112%；3cp/2val/51小control818699B SEALED/owned退出/锁退。50%同1050混合于北京08:37:26唯一启动D/G main_bbox_yolo11s_soccernet_RS_seed17_20261104_a01，controller14132/CT1793752646.1709087、wrapper26616/CT1793752646.1537173；固定runtime={"started_at_utc": "2026-11-04T00:37:35.052315+00:00", "deadline_unix": 1793758055.0523157, "max_run_seconds": 5400}，0为PT0验证。独立GPU机械PASS33.610s，全进程37.0342175961s；本四项含旧FAIL累计127.3493590355/240。继续10R→10混合→seed29/43共16→两场48确认，全部原Gate/Nov7截止，Seg不训练/GPU。方案5dcb2c911a5e1a985e2f64c4e3157b728c537c2f2f42989dac8fec7f6cbfb691原bytes保持。


当前新标注曲线首项50%纯R(seed17/1050图/1500步)已于北京11月4日08:04:44唯一启动，D/G R_seed17_20261104_a02/controller23652/CT1793750684.882066，固定截止09:34:53；初始0为PT0真实验证。首a01预检协作超时原FAIL保留，一次同协议retry34.294秒全退出PASS，失败+PASS累计90.315秒/原240，本次不加预算。继续50混合→10纯R→10混合，再seed29/43各四档八项共16，最后两场比赛确认；逐项完整Gate/原Nov7截止，Seg不训练/GPU。方案SHA5dcb2c911a5e1a985e2f64c4e3157b728c537c2f2f42989dac8fec7f6cbfb691。

最新直接批准BBox标注效率曲线：seed17新增50%/10%各R与1:1混合共4项1500步，其后seed29/43各复现100/50/25/10全部8项共16训练，最后24末EMA在game8/9两场各750图一次固定评价。子集10⊂25⊂50⊂100已冻结，训练seed不重选UID。EXPERIMENTS新SHA5dcb2c911a5e1a985e2f64c4e3157b728c537c2f2f42989dac8fec7f6cbfb691，PAPER原SHA保持；旧run绑定合法，不回写。新20训练各机械全60秒/正式90min且逐项完整Gate；未通过不派发，不延Nov7。Seg/Pose不训练/GPU。

有限BBox N25配对已全部验收闭合：纯R525/1500步AP50:95 51.7867%，混合相同525/1500步52.4119%，+0.6252pp；仍低于全2100 R1500的55.1634%达2.7515pp，补回18.5%性能缺口，不是已证明节省75%标注/等效。混合3cp/2val/51小control816420B SEALED、owned全退/GPU锁退休已核，原生bbox-n25已删除，g不动，无自动后续训练。四组对比见LOWLABEL25-RS-RESULT-001 / BBOX-N25-CLOSE-001与G label_budget_comparison.json；Seg仅CPU，不训练/GPU。

当前N25纯R已于北京22:13:38终验COMPLETE1500/R24000S0，末AP50:95 51.7867%；3cp/2val/51小control817423B SEALED/owned退出/锁退休均核。同525混合1500于北京22:18:18唯一启动（RS组20261103_a03/controller29744/CT1793715498.6285424），R12000S12000，固定截止23:48:30。独立机械PASS/41.172秒全退出/完整资源已核，bbox-n25更新只跟进此最后混合终验，无自动后续训练，Seg不训练/GPU。见LOWLABEL25-R-RESULT-001 / LOWLABEL25-RS-ADMIT-001。

当前N25=525低标注纯R1500已于北京21:47:42唯一启动（R组20261103_a02/controller34940/CT1793713662.657884），初始0是PT0验证；固定90min截止23:17:52。实际全状态GPU恢复PASS、完整22.6267GiB+100GiB资源通过，42数值代码不改。bbox-n25每30min静默接续，完整闭合后独立准入同525 mixed1500；无分数条件，不扩50%/seed/型号，Seg禁止训练/GPU。观察台v7已实际显示预算和真实图525/2100。详见LOWLABEL25-R-ADMIT-001。

最新Seg准备：ihelon512/512已补齐，跨源全433664对无exact或pHash<=8候选，但比赛源链仍未证；内部4.jpg/517.jpg同RGB不同GT必须同组。1359 UID仅保守两组split=null不准入。CPU Seg后端根11 PASS/0skip，真实公开s/960/GPU/生产MaskAP与完整执行生命周期仍待；Seg禁止训练/GPU。观察台v7菜单已加预算与真实图数、修正未登记分类，详见唯一日志SEG-FULL512-BACKEND-001 / MONITOR-BUDGET-001。

最新直接批准有限BBox N25=525低标注配对：独立R与1:1混合各1500步，先实际CPU/恢复/资源Gate，当前尚未低标注派发；子集manifest bdac16a1...实际有序525 SHA52967e8f...，仅原train完整时间块，val/test不选样。EXPERIMENTS新SHAc3450093007a43c16ae747ebc8a294ba7e1ffda202cd71d8329a7a57cfafc09f，PAPER原SHA保持。旧run/已完成两预算绑定不改；Seg原图补齐/CPU后端准备继续，禁止Seg实际训练/GPU。

BBox两预算有限块已全部完成并验收，无活动BBox训练/GPU锁；独立R1500末AP55.1634%、mixed1500末54.0447%，同1500混合−R为−1.1188pp，旧6563为+2.2807pp。R短长差+3.5123pp、mixed短长仅+0.1128pp，不能宣称混合提升充分开发R的最高性能；单seed已开发单场val。RS1500的3cp/2val/50小control794039B SEALED与owned全退已核，控制SHA5a552a21262131842e37582ba082196a608b20b12bff1f7494768748cf7c7cec，G completion_verification/budget_comparison小收据保留。有限bbox-r已删除，g保留，不自动追加BBox。当前用户授权Seg补齐496同源原图、来源/疑难GT核验与CPU训练恢复后端；Seg不实际训练、不GPU预检，原GT只读。方案SHA9d8d1afbaffdf591804725e87c33eca45b155239b7c3c98a7aade7c26f50a8cf/PAPER原SHA保持。

现行实验设计见docs/plan/EXPERIMENTS.md（SHA256 8f23e187304d067a36cbe288f28438a431bb2de0ed393ecdb2d03b75c121284d），结果与方案变更只写docs/exp/EXPERIMENT_LOG.md；正文已按研究问题与流程重整，不再追加对话/审批条款。PAPER保持d48cb4160a513773cde0e910fe1c45c7eb5a63bf058871eafd38e9ad27b71c17；全部旧run计划绑定合法，既有mixed及正在运行的短SR冻结源码/配置/各自保存规则不改；RR追加最多60秒独立机械预检已获准，待SR终验后执行。有限执行链为mixed完整闭合→新短SR→新短RR；新两组1500+5063/每1000cp/每阶段0末val，先实现与实际准入，不复用弃用S3000权重。bbox-sr-mix每30分钟/原Nov7界限；低标注仅设计，Seg/Pose仅CPU准备不训练。


2026-11-03短前缀A已完整闭合：北京时间14:26:04完成，新增R6563/105008R、继承S3000/48000S，完整9563；末步EMA主AP49.7264816863%、AP50 92.7471871498%、AP75 46.9127448839%。A较强R低1.92467206348pp、较长SR高1.18577180696pp，仍未超过直接R；15新完整cp/3新750val+父PT0引用、末步曝光、50文件749382B小control SEALED与owned退出/锁退休全核。按预登记条件B未触发/未执行，无B结果；有限跟进bbox-a-b已删除，g不动，不追加搜索。Seg/Pose仍只CPU准备、GT质量待审，不训练。见唯一日志SHORT-PREFIX-RESULT-001。

2026-11-03量化评估已按直接批准写入EXPERIMENTS（新SHA77284d717d0bd1a6895be9f4ba2eda216709f611c857d0d1a80df14c95155c1c，PAPER原SHA保持）。Seg新增独立CPU数据/增强/micro1组批/Mask AP适配，14项CPU及真实R/S单图接线通过；两个足球源全部14011原人物标注已逐图光栅核验，发现ihelon209.jpg7对重复、Quantigo B3一对重复和ann4482自交，原GT未修，完整比赛独立性仍未证实，训练准入待审。Pose实际3DSP原17非COCO17、缺逐点v、官方train/test有同场重叠，14/3/3场候选2820/660/520帧，9项CPU通过；尺寸/路径/时间缺口已记。Seg/Pose均不训练/不GPU预检；现BBox A/条件B冻结包不改，未来B只用A已冻40模块。详见唯一日志QUANT-PLAN-APPLY-001、SEG-CPU-AUDIT-003、POSE-3DSP-AUDIT-003。

2026-11-03短前缀开发A已于北京时间13:12:47唯一启动：S3000完整前缀复用→新R6563，完整9563/48000S+105008R；root机械预检33.36秒PASS、14项CPU通过、40源码冻结、完整新增39.31938921GiB另100GiB与单项3h预算均核。D/G main_bbox_yolo11s_soccernet_S2R_seed17_20261103_a01，controller28928/CT1793682767.4315，wrapper30780/CT1793682767.3923464；初始local0为真实750val，不冒称新增训练步数。仅A末步AP>R .5165115374980012且完整闭合才执行B；新跟进bbox-a-b每60分钟/原Nov7截止，旧三核心/缓存/失败原bytes保留。量化标准和Seg/Pose细化仅待审规划，详见唯一日志SHORT-PREFIX-ADMIT-001、SYNTH-QUANT-001、SEG-POSE-PLAN-002。

2026-11-03有限三核心已全部闭合（S2R-RESULT-001）：S→R于北京时间12:37:51 COMPLETE/36750更新，105008R+482992S曝光；末步EMA标准COCO AP50:95=48.5407098794%、AP50=91.5425023353%、AP75=45.3166833427%。同seed同750真实SoccerNet val，R=51.6511537498%、R→R=46.65311479396%，SR-R=-3.11044387044pp，SR-RR=+1.88759508540pp；当前长预算S预训练未超直接R，不据较弱RR对照宣称普适贡献。78完整cp（77计划+1暂停）、4端点val、末步与曝光、旧暂停控制49文件747946B和新恢复控制59文件781422B小SHA、owned退出/GPU锁退休均核通过；旧control原SHA保留，无unknown writer。停止并删除bbox-r-r-r跟进，不操作每日g，不自动扩seed/规格/新实验。所有模型/预测/日志/失败和缓存原位保留，未搬整组或全哈pt/NPZ。

更新：2026-11-02。唯一工作清单，完成一项就在本文件勾选并写一句产物/阻塞。只更新本文件，不生成周报、阶段清单副本或新的方案归档。勾选代表本轮实际做过，不把“可以做”当成已完成。

## 本次明确授权的Seg/Pose CPU准备（不训练）

- [x] 量化框架入获准方案并登记新基线。
- [x] Seg原COCO/类别映射、全人物Mask光栅质量与16对847帧重复抽样核验。
- [x] Seg数据→同步几何→独立mask/micro1 CPU组批→标准Mask AP适配，14项CPU/真实R和S样本接线。
- [x] Pose 3DSP实际17点、全部比赛info/JSON/图头核验与分组工具，9项CPU检查。
- [x] Seg四源原GT/候选/ledger完整元数据血缘复核、8精确重复及11原像素抽样一致；原GT保持。新增五级SHA的CPU_review读取器与8项检查通过，实例保原/派生payload及aliases。
- [x] Seg原生criterion随机n/96CPU正空/重叠/孔洞/断开5项检查通过，0优化器步/CUDA；32小实例960分辨率有界探针留证，不冒称真实训练准入。
- [x] 来源复核：Quantigo B2/B3同场候选，三video不当三独立场；ihelon16/512已取得，跨源独立尚未证实。详见SEG-READINESS-005。
- [x] 剩余496张原媒体已补齐，ihelon512/512；原GT保留。
- [ ] 补原始比赛/上游视频来源链，冻结比赛级train/dev UID；现1359 UID仅两保守组、全部split=null，不按视频随机拆同场。
- [ ] GT候选最终裁定：原COCO像素RLE规范化非人工修复；4482不完整人物疑难需审，类别/未标观众、原area/bbox评价与小目标proto降采样处理事前固定，不静默改GT。
- [x] Seg有效16累积/EMA/完整RNG-sampler-loader磁盘恢复CPU后端已准备（旧11项）；新原生预测→记录几何逆映射→RLE/MaskAP与证据接口10项CPU通过，未运行真实公开s/960。
- [ ] 接通生产500步持久提交/停止/恢复/worker退出/输出与小control、正式逆几何与辅助BBox评价；CPU候选不等于生产训练可运行。
- [ ] 最小公平对照与绝对预算/完整资源需冻结；用户当前禁止Seg训练与GPU预检，CPU准备不替代GPU准入。
