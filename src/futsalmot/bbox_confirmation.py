"""24冻结末EMA的比赛外确认适配；默认不读test、不加载模型、不初始化CUDA。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType
import ast,hashlib,inspect,json,math,os,textwrap,time
from . import adapters as _adapters,dataset as _dataset,train_data as _train,runner_data as _data,batches as _batches
from .records import CheckpointReceipt,load_checkpoint

SCHEMA='bbox_confirmation_spec_v1'
COMPLETION_SCHEMA='bbox_curve24_completion_v1'
MATCHES=('match:soccernet-tracking:game:8','match:soccernet-tracking:game:9')
EXECUTION={'numeric_mode':'fp32','input_size':960,'microbatch_size':8,'num_workers':2,'max_seconds_per_model_match':900,'checkpoint_selection':'terminal_EMA','source_split':'test','preprocess_view':'val','optimizer_updates':0,'resume_allowed':False}
EXPECTED_KEYS={(s,f,t) for s in (17,29,43) for f in (1.,.5,.25,.1) for t in ('R','R+S')}
_AUTHORITY=object()
SOURCE_GROUP=Path('D:/projects/FutsalMOT_SyntheticDatasetResearch/runs/main_bbox_yolo11s_soccernet_R_seed17_20261103_a01')

def _require(value,message):
    if not value:raise ValueError(message)

def _digest(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def _sha(raw):return hashlib.sha256(raw).hexdigest()

def _canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False)

def _pool_sha(uids):return _sha(json.dumps(tuple(uids),ensure_ascii=True).encode())

def _small_reference(reference,*,maximum=64*1024**2):
    _require(type(reference) is dict and set(reference)>={'path','sha256'} and _digest(reference['sha256']),'缺精确path/SHA引用')
    path=Path(reference['path']);_require(path.is_absolute() and '..' not in path.parts and not str(path).startswith(('\\\\','//')),'引用须本机绝对普通路径')
    before=_adapters._source_stat(path);_require(0<before[2]<=maximum,'小引用为空或超界')
    with path.open('rb') as stream:
        _require(_adapters._stat_tuple(os.fstat(stream.fileno()))==before,'打开时小引用变化');raw=stream.read(maximum+1)
        _require(_adapters._stat_tuple(os.fstat(stream.fileno()))==before,'读取时小引用变化')
    _require(_adapters._source_stat(path)==before and _sha(raw)==reference['sha256'],'小引用身份/摘要变化')
    if 'bytes' in reference:_require(reference['bytes']==len(raw),'小引用字节数变化')
    return raw,(str(path),before)


def validate_spec(spec):
    """纯字典校验，不读取真实数据；固定24模型/2比赛/48次，不选best。"""
    _require(type(spec) is dict and spec.get('schema_version')==SCHEMA and spec.get('state')=='FROZEN','确认方案须先冻结')
    _require(spec.get('inference_only') is True and spec.get('execution')==EXECUTION,'确认只准原数值FP32/960/micro8/workers2/900秒推理')
    _require(spec.get('evaluation_count')==48 and spec.get('deadline_utc')=='2026-11-07T23:55:00Z','只准48次及原全局截止')
    models=spec.get('models');_require(type(models) is list and len(models)==24,'必须固定24个模型')
    keys=[];ids=[];paths=[]
    for model in models:
        key=(model.get('seed'),model.get('real_fraction'),model.get('strategy'));_require(type(model.get('seed')) is int and type(model.get('real_fraction')) in (int,float) and key in EXPECTED_KEYS,'模型seed/标注档/策略不符')
        _require(type(model.get('model_id')) is str and model['model_id'],'模型须唯一ID');ids.append(model['model_id']);keys.append(key)
        for name in ('status_reference','control_manifest_reference','job_table_reference','checkpoint_index_reference'):_require(type(model.get(name)) is dict and _digest(model[name].get('sha256')),'缺模型闭合引用')
        receipt=model.get('checkpoint_receipt');_require(type(receipt) is dict and receipt.get('schema_version')=='futsalmot_checkpoint_v1' and receipt.get('producer')=='futsalmot.records/v1' and _digest(receipt.get('sha256')) and _digest(receipt.get('source_bindings_sha256')) and type(receipt.get('bytes')) is int and receipt['bytes']>0,'缺完整可信checkpoint receipt')
        _require(type(model.get('expected_source_bindings')) is dict and model['expected_source_bindings'],'缺checkpoint原完整source bindings');paths.append(receipt.get('path'))
    _require(set(keys)==EXPECTED_KEYS and len(set(ids))==24 and len(set(paths))==24,'模型键/ID/checkpoint路径重复或缺失')
    matches=spec.get('matches');_require(type(matches) is list and [x.get('match_uid') for x in matches]==list(MATCHES),'只准顺序固定game8/game9')
    all_uids=[]
    for match in matches:
        uids=match.get('ordered_uids');_require(type(uids) is list and len(uids)==len(set(uids))==750 and all(type(u) is str and u.startswith('detect:soccernet_tracking:') for u in uids),'每比赛必须精确750有序UID')
        _require(match.get('ordered_uid_sha256')==_pool_sha(uids),'有序确认UID SHA不符');_require(type(match.get('files')) is list and len(match['files'])==15,'每比赛须5clip各GT/gameinfo/seqinfo三小文件绑定');all_uids+=uids
    _require(len(set(all_uids))==1500,'两比赛UID重叠')
    return deepcopy(spec)


@dataclass(frozen=True)
class ConfirmationAuthorization:
    """只由authorize_confirmation创建；保存原冻结JSON bytes，属性返回副本。"""
    _spec_bytes:bytes
    _proofs:tuple
    references:tuple
    _authority:object=None
    def __post_init__(self):_require(self._authority is _AUTHORITY,'确认授权只能由完整完工factory创建')
    def __reduce__(self):return (_restore_authorization,self.references)
    def spec(self):return json.loads(self._spec_bytes)
    def check_live(self):
        for path,identity in self._proofs:_require(_adapters._source_stat(path)==tuple(identity),'确认来源或训练闭合小证据已变化')
    def match(self,match_uid):
        _require(match_uid in MATCHES,'只准game8/game9')
        return next(row for row in self.spec()['matches'] if row['match_uid']==match_uid)
    def model(self,model_id):
        rows=[x for x in self.spec()['models'] if x['model_id']==model_id];_require(len(rows)==1,'模型不在冻结24清单');return rows[0]


def authorize_confirmation(spec_reference,*,completion_reference,plan_reference):
    """先核全部24训练闭合，随后才允许第一次test GT小hash；不加载checkpoint。"""
    raw,proof=_small_reference(spec_reference);spec=validate_spec(json.loads(raw));proofs=[proof]
    _require(spec.get('evaluation_code_sha256')==execution_code(),'确认执行42源码绑定不同')
    _require(_canonical(spec.get('plan_reference'))==_canonical(plan_reference),'确认方案与现计划引用不同')
    _,p=_small_reference(plan_reference);proofs.append(p)
    completed,p=_small_reference(completion_reference);proofs.append(p);completed=json.loads(completed)
    _require(completed.get('schema_version')==COMPLETION_SCHEMA and completed.get('state')=='COMPLETE','20新增训练/24模型尚未整体闭合')
    done=completed.get('models');_require(type(done) is list and len(done)==24 and len({x.get('model_id') for x in done})==24,'完工清单非唯一24模型')
    done={x['model_id']:x for x in done}
    from .records import _binding_digest
    for model in spec['models']:
        row=done.get(model['model_id']);_require(row is not None,'完工清单缺模型')
        for name in ('status_reference','control_manifest_reference','job_table_reference','checkpoint_index_reference','checkpoint_receipt'):_require(_canonical(row.get(name))==_canonical(model[name]),'完工清单与冻结模型引用不同:'+name)
        status,p=_small_reference(model['status_reference']);proofs.append(p);status=json.loads(status)
        _require(status.get('state')=='COMPLETE' and status.get('update')==1500 and status.get('phase_update')==1500,'模型非1500末步COMPLETE')
        expected={'R':24000,'S':0} if model['strategy']=='R' else {'R':12000,'S':12000}
        _require(status.get('exposures')==expected and status.get('terminal',{}).get('tag')=='step001500_final_final','模型曝光/末步EMA标记不同')
        control,p=_small_reference(model['control_manifest_reference']);proofs.append(p);control=json.loads(control)
        _require(control.get('state')=='SEALED' and control.get('scope')=='control_snapshot_only','模型小控制未封闭')
        _require(row.get('owned_exit_verified') is True and row.get('GPU_lock_retired') is True,'模型owned/锁未闭合')
        table_raw,p=_small_reference(model['job_table_reference']);proofs.append(p);table=json.loads(table_raw)
        _require(type(table.get('jobs')) is list and len(table['jobs'])==1,'确认只接原独立单job训练表');cfg=table['jobs'][0]
        _require((cfg.get('seed'),cfg.get('strategy'),cfg.get('N_R'),cfg.get('updates'))==(model['seed'],model['strategy'],int(model['real_fraction']*2100),1500),'训练表seed/档位/策略/步数与模型键不符')
        _require((cfg.get('model'),cfg.get('input_size'),cfg.get('numeric_mode'),cfg.get('microbatch_size'),cfg.get('num_workers'))==('yolo11s',960,'fp32',8,2),'训练原数值规格不符')
        leaf=Path(model['status_reference']['path']).parent
        _require(status.get('directory')==str(leaf) and cfg.get('job_id')==leaf.name,'status目录/原job ID不同')
        _require(Path(model['checkpoint_index_reference']['path'])==leaf/'checkpoints.jsonl','cp索引不属于该实际训练leaf')
        index_raw,p=_small_reference(model['checkpoint_index_reference']);proofs.append(p)
        terminal=[json.loads(line) for line in index_raw.splitlines() if line.strip()];terminal=[x for x in terminal if (x.get('update'),x.get('phase'),x.get('phase_update'),x.get('tag'))==(1500,'final',1500,'final')]
        _require(len(terminal)==1 and _canonical(terminal[0].get('receipt'))==_canonical(model['checkpoint_receipt']),'末receipt不来自该实际leaf唯一1500 final记录')
        receipt=model['checkpoint_receipt'];_require(Path(receipt['path'])==leaf/'checkpoints/001500_final_001500_final.pt','末checkpoint不属于该训练leaf')
        expected_bindings={**table['source_bindings'],'protocol':cfg}
        _require(_canonical(expected_bindings)==_canonical(model['expected_source_bindings']),'原完整sourcebindings/训练protocol与模型键不同')
        _require(_binding_digest(model['expected_source_bindings'])==receipt['source_bindings_sha256'],'原模型source bindings与receipt不符')
        cp=Path(receipt['path']);identity=_adapters._source_stat(cp);_require(identity[2]==receipt['bytes'],'末checkpoint字节数变化');proofs.append((str(cp),identity))
    # 此行之前不打开任何test GT/媒体。索引只含metadata，原split不改写。
    index_raw,p=_small_reference(spec['index_reference']);proofs.append(p);index=json.loads(index_raw);root=Path(spec['index_reference']['path']).parent
    _require(index.get('schema_version')=='canonical_real_dataset_v1' and index.get('source',{}).get('source_id')=='soccernet_tracking','只准原canonical SoccerNet')
    source_matches={x['match_uid']:x['split'] for x in index['matches']}
    _require(all(source_matches.get(x)=='test' for x in MATCHES),'game8/game9原split必须test')
    for match in spec['matches']:
        seqs=[x for x in index['sequences'] if x['match_uid']==match['match_uid']];_require(len(seqs)==5 and all(x['split']=='test' for x in seqs),'每比赛须原test五clip')
        uids=[x['detection_sample_uid_pattern'].format(frame_uid=x['frame_uid_pattern'].format(frame_number=f)) for x in seqs for f in x['detection_frames']]
        _require(uids==match['ordered_uids'],'确认UID非原index顺序750')
        expected={str((root/x['sequence_path']/x['files'][name]).absolute()) for x in seqs for name in ('gt','gameinfo','seqinfo')}
        _require({str(Path(x['path']).absolute()) for x in match['files']}==expected,'GT/metadata文件集合不符')
        for reference in match['files']:
            _,p=_small_reference(reference);proofs.append(p)
        for x in seqs:
            ref=next(v for v in match['files'] if Path(v['path'])==root/x['sequence_path']/x['files']['gt'])
            _require(ref['sha256']==x.get('gt_sha256'),'确认GT SHA须沿原index，不另造GT')
    return ConfirmationAuthorization(raw,tuple(proofs),(deepcopy(spec_reference),deepcopy(completion_reference),deepcopy(plan_reference)),_AUTHORITY)


def _restore_authorization(spec_reference,completion_reference,plan_reference):
    # 每个spawn worker从冻结小引用实际重新核验，不能pickle一个允许布尔值越过准入。
    return authorize_confirmation(spec_reference,completion_reference=completion_reference,plan_reference=plan_reference)


def execution_code(code_source=SOURCE_GROUP/'code/futsalmot'):
    source=Path(code_source);known=json.loads((SOURCE_GROUP/'code_bindings.json').read_bytes())
    _require(len(known)==41 and {x.name:_sha(x.read_bytes()) for x in source.glob('*.py')}==known,'确认须精确复用原R1500冻41源码')
    from . import runner,runtime_backend,records,detectors,training,postprocess,coco_metrics,metrics
    for module in (_adapters,_dataset,_train,_data,runner,runtime_backend,records,detectors,training,postprocess,coco_metrics,metrics):_require(_sha(Path(module.__file__).read_bytes())==known[Path(module.__file__).name],'当前加载的原数值模块不同')
    return {**known,'bbox_confirmation.py':_sha(Path(__file__).read_bytes())}


def _derive(function,replacements,namespace=None):
    source=textwrap.dedent(inspect.getsource(function));patched=source
    for before,after in replacements:
        _require(patched.count(before)==1,'原函数机械接点变化');patched=patched.replace(before,after)
    reverse=patched
    for before,after in reversed(replacements):reverse=reverse.replace(after,before)
    _require(ast.dump(ast.parse(reverse))==ast.dump(ast.parse(source)),'机械适配发生范围外AST变化')
    scope=dict(function.__globals__);scope.update(namespace or {});exec(compile(patched,__file__+':isolated_derived','exec'),scope)
    return scope[function.__name__],{'original_source_sha256':_sha(source.encode()),'adapted_source_sha256':_sha(patched.encode()),'substitutions':replacements,'inverse_AST_exact':True}


_READER_INIT,READER_INIT_BINDING=_derive(_adapters.RealBBoxReader.__init__,[('if seq["split"] not in ("train", "val"):', 'if seq["split"] != "test" or seq["match_uid"] != self.match_uid:')])
_READER_READ,READER_READ_BINDING=_derive(_adapters.RealBBoxReader.read,[('if split not in ("train", "val"):', 'if split != "test":')])
_PREPARE,PREPROCESS_BINDING=_derive(_train.prepare_training_sample,[('if split!=(view if domain=="R" else ("train" if view=="train" else "dev")):', 'if not (domain=="R" and split=="test" and view=="val"):')])


class ConfirmationReader(_adapters.RealBBoxReader):
    """只准已授权比赛原test UID；source字段仍test，原train reader完全不改。"""
    def __init__(self,index_path,*,authorization,match_uid,max_cached_clips=20,max_cached_rows=600000):
        _require(type(authorization) is ConfirmationAuthorization,'缺全24模型闭合授权');authorization.check_live()
        _require(Path(index_path)==Path(authorization.spec()['index_reference']['path']),'确认reader索引不同')
        self.authorization=authorization;self.match_uid=match_uid;self._allowed=frozenset(authorization.match(match_uid)['ordered_uids'])
        _READER_INIT(self,index_path,max_cached_clips=max_cached_clips,max_cached_rows=max_cached_rows)
        _require(set(uid for split,uid in self._lookup)==self._allowed,'reader实际UID不等冻结750')
    def read(self,sample_uid,*,split='test'):
        _require(split=='test' and sample_uid in self._allowed,'非授权test比赛/UID');return _READER_READ(self,sample_uid,split=split)
    def __getstate__(self):
        return {**super().__getstate__(),'authorization':self.authorization,'match_uid':self.match_uid}
    def __setstate__(self,state):
        self.__init__(state['index_path'],authorization=state['authorization'],match_uid=state['match_uid'],max_cached_clips=state['max_cached_clips'],max_cached_rows=state['max_cached_rows'])
        _require(self._index_stat==tuple(state['index_stat']) and self._index_sha==state['index_sha'],'确认worker重建index变化');self._bindings=deepcopy(state['bindings'])


class _ConfirmationDataset(_dataset.RequestBBoxDataset):
    def __init__(self,catalogue,*,authorization,match_uid,view='val',real_reader_options=None,**options):
        _require(type(authorization) is ConfirmationAuthorization and view=='val','确认只准独立授权推理view=val');authorization.check_live()
        _require(not options,'确认不准S reader/解码预算改动');match=authorization.match(match_uid)
        _require(type(catalogue) is dict and set(catalogue)==set(match['ordered_uids']),'catalogue不等所选750')
        index=str(Path(authorization.spec()['index_reference']['path']).absolute());expected={'domain':'R','index_path':index,'split':'test'}
        _require(all(ref==expected for ref in catalogue.values()),'确认引用必须保留原test split/原index')
        self.authorization=authorization;self.match_uid=match_uid;self.view='val';self._catalogue=deepcopy(catalogue)
        self.real_reader_options=deepcopy(real_reader_options or {'max_cached_clips':20,'max_cached_rows':600000});self.synthetic_reader_options={}
        _require(self.real_reader_options=={'max_cached_clips':20,'max_cached_rows':600000},'确认reader预算变化')
        self.max_encoded_bytes=64*1024**2;self.max_pixels=16*1024**2;self._readers={};self._image_bindings={}
    def _request(self,request):
        _require(type(request) is dict and request.get('domain')=='R' and request.get('stream','R')=='R' and request.get('augmentation_seed')==0 and request.get('epoch',0)==0,'确认只准无增强的R推理请求')
        return super()._request(request)
    def _read(self,uid,ref):
        key=('R',ref['index_path'])
        if key not in self._readers:self._readers[key]=ConfirmationReader(ref['index_path'],authorization=self.authorization,match_uid=self.match_uid,**self.real_reader_options)
        return super()._read(uid,ref)
    def __getitem__(self,request):return _PREPARE(super().__getitem__(request),view='val')


class ConfirmationRequestPlanDataset(_data.RequestPlanDataset):
    def __init__(self,catalogue,requests,*,authorization,match_uid,view='val',**readeroptions):
        _require(type(requests) in (list,tuple) and [x.get('uid') for x in requests]==authorization.match(match_uid)['ordered_uids'],'确认请求须冻结750原序且只一次')
        self.dataset=_ConfirmationDataset(catalogue,authorization=authorization,match_uid=match_uid,view=view,**readeroptions);self.requests=deepcopy(list(requests))
        for request in self.requests:self.dataset._request(request)


# 仅在独立授权collator内部使用；原训练collate及合法split完全不改。
_CONFIRMATION_LINEAGE,LINEAGE_BINDING=_derive(_batches._collate_lineage,[
    ('allowed=("train","val") if domain == "R" else ("train","dev")','allowed=("test",) if domain == "R" else ()'),
    ('expected=view if domain == "R" else ("train" if view == "train" else "dev")',"expected='test' if domain == 'R' and view == 'val' else None")])
_CONFIRMATION_COLLATE,COLLATE_BINDING=_derive(_batches.collate_bbox_samples,[('_collate_lineage(s)','_confirmation_lineage(s)')],{'_confirmation_lineage':_CONFIRMATION_LINEAGE})

@dataclass(frozen=True)
class ConfirmationCollator:
    authorization:ConfirmationAuthorization
    match_uid:str
    def __call__(self,samples):
        _require(type(self.authorization) is ConfirmationAuthorization,'collate缺完整确认授权');self.authorization.check_live()
        match=self.authorization.match(self.match_uid);allowed=frozenset(match['ordered_uids']);index=str(Path(self.authorization.spec()['index_reference']['path']).absolute())
        for sample in samples:
            _require(sample.get('uid') in allowed and sample.get('domain')=='R' and sample.get('data_domain')=='R' and sample.get('sampling_stream')=='R','collate非授权比赛R样本')
            _require(sample.get('source_reference')=={'domain':'R','index_path':index,'split':'test'},'collate必须原test/index身份')
            request=sample.get('request',{});_require(request.get('augmentation_seed')==0 and request.get('epoch')==0 and request.get('view','val')=='val','确认禁止训练增强或epoch')
        return _CONFIRMATION_COLLATE(samples)

def make_confirmation_loader(dataset,**options):
    """原DataLoader参数/预取/线程不变，仅换显式授权、可spawn的collator。"""
    _require(isinstance(dataset,ConfirmationRequestPlanDataset),'确认loader仅接确认请求Dataset')
    bound=dataset.dataset;collator=ConfirmationCollator(bound.authorization,bound.match_uid)
    loader=_data.make_loader(dataset,**options);loader.collate_fn=collator
    return loader


def confirmation_panel(authorization,match_uid):
    """显式授权后才读原test GT，原角色/ignore/source_xyxy完全复用。"""
    authorization.check_live();spec=authorization.spec();match=authorization.match(match_uid);reader=ConfirmationReader(spec['index_reference']['path'],authorization=authorization,match_uid=match_uid)
    gt={};catalogue={};uids=match['ordered_uids']
    for uid in uids:
        sample=reader.read(uid,split='test');positive=[];ignore=[];aux=[];roles=[]
        for kind,records in (('positive',sample['targets']),('ignore',sample['ignore_targets'])):
            for record in records:
                if kind=='positive':positive.append(deepcopy(record['source_xyxy']))
                elif record.get('ignore_reason')=='non_target_role':aux.append(deepcopy(record['source_xyxy']))
                elif record.get('ignore_reason') in ('invalid_gt_flag','unknown_role'):ignore.append(deepcopy(record['source_xyxy']))
                else:raise ValueError('未知确认ignore语义')
                roles.append({'track_id':record['track_id'],'role':record.get('role'),'gt_flag':record['gt_flag'],'source_xyxy':deepcopy(record['source_xyxy']),'bbox_supervision_valid':record['bbox_supervision_valid'],'ignore_reason':record.get('ignore_reason')})
        gt[uid]={'positive_gt':positive,'ignore':ignore,'known_non_target_auxiliary_ignore':aux,'width':sample['width'],'height':sample['height'],'role_audit':roles,'role_policy':_train.ROLE_POLICY}
        catalogue[uid]={'domain':'R','index_path':str(reader.index_path),'split':'test'}
    reader._check_index()
    for clip in reader._bindings:reader._check_clip(clip)
    return {'val_uids':list(uids),'val_catalogue':catalogue,'val_ground_truth':gt,'confirmation_match_uid':match_uid,'source_split':'test','preprocess_view':'val','binding':{'index_reference':deepcopy(spec['index_reference']),'match':deepcopy(match),'authorization_references':deepcopy(authorization.references),'role_policy':_train.ROLE_POLICY}}


def load_terminal_ema(authorization,model_id):
    """唯一模型读取接口：原可信load_checkpoint校SHA/绑定；只导出末步EMA。"""
    authorization.check_live();model=authorization.model(model_id);state=load_checkpoint(CheckpointReceipt(**model['checkpoint_receipt']),expected_bindings=model['expected_source_bindings'])
    _require(state.get('numeric_mode')=='fp32' and state.get('committed_updates')==1500 and state.get('phase_updates')==1500 and state.get('phase')=='final','只准1500最终FP32状态')
    expected={'R':24000,'S':0} if model['strategy']=='R' else {'R':12000,'S':12000}
    _require(state.get('cumulative_exposures')==expected and set(state.get('sampler',{}))=={'R','S','R_extra'},'原checkpoint实际累计曝光/采样流不符')
    for stream in ('R','S','R_extra'):
        live=state['sampler'][stream].get('sampler',{})
        _require((live.get('seed'),live.get('domain'),live.get('draws'))==(model['seed'],stream,0 if stream=='R_extra' else expected[stream]),'原checkpoint sampler seed/domain/实际曝光不同')
    ema=state.get('ema');_require(type(ema) is dict and ema.get('updates')==1500 and type(ema.get('state_dict')) is dict and ema['state_dict'],'缺完整末步EMA')
    _require(set(ema['state_dict'])==set(state['raw_model']),'raw/EMA键集合不同')
    return {'model_id':model_id,'state_dict':deepcopy(ema['state_dict']),'ema_updates':1500,'checkpoint_receipt':deepcopy(model['checkpoint_receipt']),'expected_source_bindings':deepcopy(model['expected_source_bindings']),'selection':'terminal_EMA'}


def make_confirmation_evaluator(authorization,match_uid):
    """候选原evaluate接线：只替换Dataset构造，原推理/解码/NMS/AP/native writer原代码。"""
    from . import runner
    authorization.check_live();authorization.match(match_uid)
    before="dataset=RequestPlanDataset(panel['val_catalogue'],requests,view='val',real_reader_options={'max_cached_clips':20,'max_cached_rows':600000})"
    after="dataset=ConfirmationRequestPlanDataset(panel['val_catalogue'],requests,view='val',real_reader_options={'max_cached_clips':20,'max_cached_rows':600000},authorization=_confirmation_authorization,match_uid=_confirmation_match_uid)"
    function,binding=_derive(runner.evaluate,[(before,after),('loader=make_loader(dataset,','loader=_confirmation_make_loader(dataset,')],{'ConfirmationRequestPlanDataset':ConfirmationRequestPlanDataset,'_confirmation_authorization':authorization,'_confirmation_match_uid':match_uid,'_confirmation_make_loader':lambda *args,**kwargs:make_confirmation_loader(*args,**kwargs)})
    # 仅返回机械候选函数；真实调用仍须由待实现的持锁/预算/退出执行器封装。
    return {'candidate_function':function,'mechanical_binding':binding,'real_GPU_verified':False,'lifecycle_launcher_implemented':False,'requires_explicit_outer_900s_GPU_lock_storage_owned_exit_gate':True}
