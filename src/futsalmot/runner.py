"""首个已批准的SoccerNet BBox预算校准执行器；每个有限run独立准入。

真实公开起点/增强/梯度/完整状态/COCO评价。只支持独立R50/R100，
不因校准完成自动启动s15/39或test。生产与本GPU作业互斥。
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime,timezone
from copy import deepcopy
import hashlib,json,os,random,shutil,sys,time,traceback,uuid
from pathlib import Path
import numpy as np
import psutil
import torch

REAL=Path('G:/FutsalMOT_Dataset/real/soccernet/dataset.json')
WEIGHTS=Path('D:/projects/FutsalMOT_SyntheticDatasetResearch/weights/yolo11')
RESEARCH=Path('D:/projects/FutsalMOT_SyntheticDatasetResearch')
DEADLINE='2026-11-07T23:55:00+00:00'
RESERVE=100*1024**3
PLAN_SHA='46aa2fe83daf99bc0a2e1880f593eb36b2c11297b95f8eaaf479de19c4401446'
CHECKPOINT_EVERY=500
ARCHIVE_ROOT=Path('G:/FutsalMOT_Training')
PREFLIGHT_SHA='41bce6a72dff142c11dda6c6a7f6ce1bc345fcf70f193c64dafcd46e9283ee09'
PAPER_SHA='d48cb4160a513773cde0e910fe1c45c7eb5a63bf058871eafd38e9ad27b71c17'


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value

def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(plain(value),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

def append(path,value):
    with Path(path).open('ab') as f:
        f.write((json.dumps(plain(value),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode());f.flush();os.fsync(f.fileno())

def status(directory,value):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    temp=directory/('.status_'+uuid.uuid4().hex+'.tmp');write_once(temp,{**value,'updated_at_utc':now()})
    os.replace(temp,directory/'status.json')

def production_closed():
    active=[]
    for p in psutil.process_iter(['pid','name','cmdline','create_time']):
        try:
            name=(p.info['name'] or '').lower();cmd=' '.join(p.info['cmdline'] or [])
            if name.startswith('unrealeditor') or (name.startswith(('python','uv')) and any(s in cmd for s in ('single_seed_task.py','single_seed_release.py','run_single_seed_partitions.py','camera_render_export'))):
                active.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    if active:raise RuntimeError('BLOCKED: production/UE active '+repr(active))
    return {'observed_at_utc':now(),'active_production_processes':[]}

@contextmanager
def gpu_lock(group):
    production_closed();lock=RESEARCH/'.research_gpu.lock'
    value={'pid':os.getpid(),'create_time':psutil.Process().create_time(),'group':str(Path(group).absolute()),'created_at_utc':now()}
    write_once(lock,value)
    identity=(lock.stat().st_dev,lock.stat().st_ino)
    try:yield value
    finally:
        if lock.exists() and (lock.stat().st_dev,lock.stat().st_ino)==identity and json.loads(lock.read_bytes())==value:lock.unlink()

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_initialized():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2)

def setup_cuda(admission=None):
    if os.environ.get('YOLO_OFFLINE','').lower()!='true':raise RuntimeError('YOLO_OFFLINE=true required before imports')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('CUBLAS_WORKSPACE_CONFIG must be frozen before CUDA')
    production_closed()
    torch.cuda.init()
    if admission is not None:
        fraction=admission.get("cuda_memory_fraction")
        if not isinstance(fraction,(int,float)) or not 0<fraction<=.8:raise RuntimeError("Missing actually tested allocator limit")
        torch.cuda.set_per_process_memory_fraction(fraction,device=0)
    seed_all(17)

def check_disk(required=0):
    free=shutil.disk_usage(RESEARCH).free
    if free<RESERVE+required:raise RuntimeError(f'BLOCKED: D free={free}, reserve={RESERVE}, additional_bound={required}')
    return free

def output_group(group):
    group=Path(group).absolute()
    if not (group/'RUN_LOCATION.json').exists():return group
    from .output_storage import resolve_output_group
    return resolve_output_group(group,research_root=RESEARCH,output_base=ARCHIVE_ROOT/'runs')


def check_write_space(config,required=0):
    if config.get('storage_mode')=='direct_g':
        check_disk(32*1024**2);return check_archive_space(required)
    return check_disk(required)


def stop_requested(group,directory):
    return (Path(group)/'STOP.json').exists() or (Path(directory).parent/'STOP.json').exists()


def group_status(group,value):
    status(group,value)
    target=output_group(group)
    if target!=Path(group).absolute():status(target,value)


def group_record(group,name,value):
    write_once(Path(group)/name,value)
    target=output_group(group)
    if target!=Path(group).absolute():write_once(target/name,value)


def run_bindings(panel,config,code_hashes):
    return {'real_panel':panel['binding'],'synthetic_index':{'path':'G:/FutsalMOT_Dataset/synthetic/dataset.json','sha256':'8aff1d86371211e16fd493f5dbc0ddb5a09820a12190dd58d6f6d86d7416787a','used_for_this_R_run':False},
            'public_model':{'path':str(WEIGHTS/'yolo11s.pt'),'sha256':'85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5'},
            'EXPERIMENTS_sha256':PLAN_SHA,'PAPER_sha256':PAPER_SHA,'requirements_sha256':sha(RESEARCH/'requirements.lock'),
            'source_code':code_hashes,'protocol':config}

def check_archive_space(required):
    if not ARCHIVE_ROOT.is_dir() or ARCHIVE_ROOT.is_symlink() or ARCHIVE_ROOT.is_junction():
        raise RuntimeError('BLOCKED: training archive root missing or redirected')
    free=shutil.disk_usage(ARCHIVE_ROOT).free
    if free<RESERVE+required:raise RuntimeError(f'BLOCKED: G archive free={free}, reserve={RESERVE}, additional_bound={required}')
    return free


def checkpoint_due(update,final_update,interval):
    if interval<=0 or update<0 or update>final_update:raise ValueError('Invalid checkpoint boundary')
    return update==0 or update==final_update or update%interval==0


def validate_storage_revision(group,admission,current):
    original=admission.get('executed_code_sha256',{})
    if current==original:return None
    path=Path(group)/'storage_revision.json'
    revision=json.loads(path.read_bytes())
    if sha(Path(group)/'admission.json')!=PREFLIGHT_SHA:raise RuntimeError('Storage revision requires exact measured original admission')
    if revision.get('schema_version')!='bbox_storage_only_revision_v1' or revision.get('new_plan_sha256')!=PLAN_SHA:
        raise RuntimeError('Missing directly authorized storage revision binding')
    if revision.get('source_admission_sha256')!=PREFLIGHT_SHA or revision.get('current_source_sha256')!=current:
        raise RuntimeError('Storage revision source or preflight hash changed')
    removed=set(original)-set(current)
    added=set(current)-set(original)
    changed={name for name in original.keys()&current.keys() if original[name]!=current[name]}
    if removed or changed-{'runner.py'} or added-{'run_archive.py','output_storage.py'}:
        raise RuntimeError('Numerical/data component changed; storage-only admission reuse forbidden')
    if revision.get('checkpoint_every')!=CHECKPOINT_EVERY or revision.get('archive_root')!=str(ARCHIVE_ROOT):
        raise RuntimeError('Authorized storage rule mismatch')
    from .output_storage import resolve_output_group
    target=resolve_output_group(Path(group).absolute(),research_root=RESEARCH,output_base=ARCHIVE_ROOT/'runs')
    if revision.get('storage_mode')!='direct_g' or revision.get('output_group')!=str(target):
        raise RuntimeError('Direct G output binding mismatch')
    return sha(path)


def validate_R50_reference(group,admission):
    path=Path(group)/'R50_reference.json';ref=json.loads(path.read_bytes())
    if ref.get('schema_version')!='archived_R50_terminal_reference_v1':raise RuntimeError('Unknown R50 reference schema')
    directory=Path(ref['directory']).absolute()
    if not directory.is_relative_to((ARCHIVE_ROOT/'runs').absolute()) or '..' in directory.parts:raise RuntimeError('R50 must be in authorized archive')
    for parent in (directory,*directory.parents):
        if parent.exists() and (parent.is_symlink() or parent.is_junction()):raise RuntimeError('Redirected R50 archive path')
    for item in ref['files']:
        rel=Path(item['relative_path'])
        if rel.is_absolute() or '..' in rel.parts:raise RuntimeError('Unsafe R50 reference path')
        actual=directory/rel
        if actual.is_symlink() or actual.is_junction() or sha(actual)!=item['sha256']:
            raise RuntimeError('Archived R50 reference changed')
    required={'run.json','status.json','evaluation/step006563/summary.json'}
    if {i['relative_path'] for i in ref['files']}!=required:raise RuntimeError('Incomplete R50 terminal bindings')
    archive=directory.parent/'ARCHIVE_MANIFEST.json'
    if sha(archive)!=ref['archive_manifest_sha256']:raise RuntimeError('Archive receipt changed')
    run=json.loads((directory/'run.json').read_bytes());state=json.loads((directory/'status.json').read_bytes())
    summary=json.loads((directory/'evaluation/step006563/summary.json').read_bytes())
    if state.get('state')!='COMPLETE' or state.get('update')!=6563 or summary.get('update')!=6563 or summary.get('images')!=750:
        raise RuntimeError('R50 terminal not complete')
    cfg=run['protocol']
    for key,value in {'epochs':50,'updates':6563,'model':'yolo11s','seed':17,'N_R':2100,'N_val':750,'input_size':960,'effective_batch':16}.items():
        if cfg.get(key)!=value:raise RuntimeError('R50 scientific protocol differs: '+key)
    if cfg.get('main_metric')!='COCO_API_2.0.10_AP50_95_max100_player_goalkeeper_subset':raise RuntimeError('R50 primary metric differs')
    for key in ('numeric_mode','microbatch_size','num_workers'):
        if cfg[key]!=admission[key]:raise RuntimeError('R50 numerical mode differs: '+key)
    if canonical(run['source_bindings']['real_panel'])!=canonical(admission['source_bindings']['real_panel']):
        raise RuntimeError('R50 source panel differs')
    if run['source_bindings']['public_model']['sha256']!='85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5':
        raise RuntimeError('R50 public initialization differs')
    if state['terminal_evaluation']!=summary:raise RuntimeError('R50 status and terminal evaluation differ')
    metric=summary['main_metrics']['AP50_95']
    if not isinstance(metric,(int,float)) or not np.isfinite(metric) or not 0<=metric<=1:raise RuntimeError('Invalid R50 main AP')
    return {'state':'COMPLETE','directory':str(directory),'terminal':summary,'reference_sha256':sha(path)}


def prepare_R100_after_archive(archived_group,group):
    """旧R50自然闭合、仅旧D容量拒绝R100后，准备独立新R100；不启动GPU。"""
    archived_group=Path(archived_group).absolute();group=Path(group).absolute()
    if archived_group.parent!=(ARCHIVE_ROOT/'runs').absolute():raise ValueError('Expected exact archive group root')
    if group.parent!=(RESEARCH/'runs').absolute() or group.exists():raise ValueError('New group must be a nonexistent direct runs child')
    if (RESEARCH/'.research_gpu.lock').exists():raise RuntimeError('Active research lock; do not duplicate dispatch')
    production_closed()
    if (archived_group/'R100_seed17').exists() or (archived_group/'budget_choice.json').exists():
        raise RuntimeError('Previous R100 already exists; never duplicate independent calibration')
    manifest_path=archived_group/'ARCHIVE_MANIFEST.json';manifest=json.loads(manifest_path.read_bytes())
    if manifest.get('schema')!='futsalmot_run_archive_v1' or manifest.get('status')!='PASS' or manifest.get('target')!=str(archived_group):
        raise RuntimeError('Previous group is not a verified published archive')
    original_status=json.loads((archived_group/'status.json').read_bytes())
    if original_status.get('state')!='FAIL_STOP' or not str(original_status.get('error','')).startswith('BLOCKED: D free='):
        raise RuntimeError('Expected old R100 D-capacity handoff, not cancellation or unrelated failure')
    source_admission=archived_group/'admission.json'
    if sha(source_admission)!=PREFLIGHT_SHA:raise RuntimeError('Archived measured admission changed')
    admission=json.loads(source_admission.read_bytes());protocol(admission,100)
    directory=archived_group/'R50_seed17'
    needed=['run.json','status.json','evaluation/step006563/summary.json']
    files=[];recorded={item['path']:item for item in manifest['files']}
    for rel in needed:
        actual=directory/rel;digest=sha(actual);row=recorded.get('R50_seed17/'+rel)
        if row is None or row['sha256']!=digest or row.get('bytes',row.get('size'))!=actual.stat().st_size:
            raise RuntimeError('R50 small evidence not bound by archive receipt')
        files.append({'relative_path':rel,'sha256':digest})
    ref={'schema_version':'archived_R50_terminal_reference_v1','directory':str(directory),'files':files,'archive_manifest_sha256':sha(manifest_path)}
    current={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')}
    revision={'schema_version':'bbox_storage_only_revision_v1','source_admission_sha256':PREFLIGHT_SHA,'new_plan_sha256':PLAN_SHA,
        'current_source_sha256':current,'checkpoint_every':CHECKPOINT_EVERY,'archive_root':str(ARCHIVE_ROOT),'storage_mode':'direct_g','output_group':str(ARCHIVE_ROOT/'runs'/group.name),
        'authorization':'2026-11-01 direct user: checkpoint500 and all training evidence directly on G to avoid cross-volume move',
        'numerical_preflight_not_repeated':'all model/data/loss/optimizer/augmentation/evaluator/numeric components identical; only runner storage/dispatch plus archive/output_storage modules changed',
        'verification_record':'docs/exp/EXPERIMENT_LOG.md STORAGE-003'}
    # 全部源数据/容量先校验，之后才按需建立未来工作集；不加载网络。
    group.mkdir(parents=False,exist_ok=False)
    from .output_storage import create_output_binding,capture_control
    create_output_binding(group,research_root=RESEARCH,output_base=ARCHIVE_ROOT/'runs')
    (group/'admission.json').write_bytes(source_admission.read_bytes())
    write_once(group/'storage_revision.json',revision);write_once(group/'R50_reference.json',ref)
    freeze(group,epochs=100)
    launch=("from pathlib import Path\nimport os,sys,runpy\n"
        "def main():\n    group=Path(__file__).parent\n"
        "    os.environ.update(PYTHONDONTWRITEBYTECODE='1',YOLO_OFFLINE='true',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')\n"
        "    sys.path.insert(0,str(group/'code'))\n    sys.argv=['futsalmot.runner','controller','--group',str(group)]\n"
        "    runpy.run_module('futsalmot.runner',run_name='__main__')\nif __name__=='__main__':main()\n")
    with (group/'launch.py').open('x',encoding='utf-8',newline='') as f:f.write(launch)
    capture=capture_control(group,research_root=RESEARCH,output_base=ARCHIVE_ROOT/'runs')
    return {'state':'FROZEN_NOT_DISPATCHED','group':str(group),'output_group':str(output_group(group)),'control_snapshot':capture,'epochs':100,'checkpoint_every':CHECKPOINT_EVERY,'launch_sha256':sha(group/'launch.py')}


def freeze(group,*,epochs=None):
    group=Path(group).absolute();admission=json.loads((group/'admission.json').read_bytes())
    if admission.get('status')!='PASS':raise RuntimeError('No actual PASS admission')
    current={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')}
    revision=validate_storage_revision(group,admission,current)
    if epochs==100:validate_R50_reference(group,admission)
    elif epochs is not None:raise ValueError('Single followup is only independent R100 with completed R50 reference')
    if sha(RESEARCH/'docs/plan/EXPERIMENTS.md')!=PLAN_SHA or sha(RESEARCH/'docs/plan/PAPER.md')!=PAPER_SHA:raise RuntimeError('Protected plan changed')
    code=group/'code'/'futsalmot';code.mkdir(parents=True,exist_ok=False)
    hashes={}
    for source in sorted(Path(__file__).parent.glob('*.py')):
        dest=code/source.name;dest.write_bytes(source.read_bytes());hashes[source.name]=sha(dest)
    shutil.copyfile(RESEARCH/'requirements.lock',group/'code'/'requirements.lock')
    write_once(group/'code_bindings.json',hashes)
    write_once(group/'job_table.json',{'scope':'independent_R50_R100_only','jobs':[protocol(admission,e,admit=False) for e in ((epochs,) if epochs else (50,100))],'storage_revision_sha256':revision,'R50_reference_sha256':sha(group/'R50_reference.json') if epochs==100 else None,'resource_admission':'each_job_immediately_before_dispatch','selection':'100_if_terminal_COCO_AP50_95_strictly_higher_else50','test_enabled':False,'s15_enabled':False})
    return hashes

def protocol(admission,epochs,*,admit=True):
    updates=(2100*epochs+15)//16
    checkpoints=1+updates//CHECKPOINT_EVERY+(updates%CHECKPOINT_EVERY!=0)
    evaluations=1+updates//1000+(updates%1000!=0)
    ckpt=admission['checkpoint_bytes'];raw=admission['raw_upper_per_image_bytes']
    bound=int(checkpoints*ckpt*1.10+evaluations*750*raw+(8 if epochs==50 else 16)*1024**3+2*ckpt)
    base_seconds=admission.get('R50_max_run_seconds',14400)
    max_seconds=min(24*3600,int(base_seconds*(epochs/50)))
    remaining=(datetime.fromisoformat(DEADLINE)-datetime.now(timezone.utc)).total_seconds()
    if max_seconds+600>remaining:raise RuntimeError('BLOCKED: cannot close frozen job before absolute deadline')
    if admit:
        check_disk(32*1024**2);check_archive_space(bound)
    return {'schema_version':'bbox_R_calibration_run_v1','task':'detect','model':'yolo11s','strategy':'R','seed':17,'epochs':epochs,'N_R':2100,'N_val':750,'updates':updates,
            'numeric_mode':admission['numeric_mode'],'microbatch_size':admission['microbatch_size'],'num_workers':admission['num_workers'],
            'effective_batch':16,'input_size':960,'checkpoint_every':CHECKPOINT_EVERY,'evaluation_every':1000,'checkpoint_count':checkpoints,'evaluation_count':evaluations,
            'evaluate_stage0':True,'stage_final_only_for_budget_choice':True,'best_selection':False,
            'role_policy':'role_policy_v2_known_non_target_background_invalid_unknown_ignore_positive_priority',
            'main_metric':'COCO_API_2.0.10_AP50_95_max100_player_goalkeeper_subset','auxiliary_metric':'custom_player_ignore_AP@300',
            'max_run_seconds':max_seconds,'additional_space_bound_bytes':bound,'disk_reserve_bytes':RESERVE,'raw_upper_per_image_bytes':raw,
            'deadline_utc':DEADLINE,'mainline_all_logs_checkpoints_predictions_retained':True,'infrastructure_retry_limit':1,'source_admission_sha256':None,'storage_revision_sha256':None,'archive_root':str(ARCHIVE_ROOT),'archive_after_owned_normal_exit':False,'archive_disk_reserve_bytes':RESERVE,'storage_mode':'direct_g','control_space_bound_bytes':32*1024**2}

def _requests(panel,live,updates,start):
    from .data import CycleSampler
    clone=CycleSampler(panel['train_uids'],task='detect',domain='R',seed=17)
    clone.load_state_dict(live.state_dict()['sampler'])
    return clone.take((updates-start)*16)

def _microbatches(iterator,number):
    from .training import prepare_detection_batch
    return [prepare_detection_batch(next(iterator),device='cuda:0',non_blocking=True) for _ in range(number)]

def _shutdown(iterator):
    if iterator is not None and hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()


def save_boundary(directory,backend,live,bindings,gen,recorder,tag):
    from .records import save_checkpoint
    j=backend.stage_update
    bundle=backend.snapshot(samplers={'R':live},source_bindings=bindings,committed_updates=j,cumulative_exposures={'R':j*16,'S':0},
        extra_state={'loader_generator_state':gen.get_state().clone(),'run_scope':'independent_R_budget_calibration'})
    receipt=save_checkpoint(Path(directory)/'checkpoints'/f'{j:06d}_{tag}.pt',bundle)
    payload={'kind':'checkpoint','update':j,'tag':tag,'receipt':asdict(receipt),'saved_at_utc':now()}
    recorder.event(payload);append(Path(directory)/'checkpoints.jsonl',payload)
    return receipt


def _raw_write(path,native,raw_records):
    reasons=sorted({s for row in raw_records for s in row['filter_reasons']})
    if len(reasons)>16:raise RuntimeError('Raw flags exceed declared uint16 encoding')
    flags=np.zeros(len(raw_records),dtype=np.uint16)
    for i,row in enumerate(raw_records):
        for reason in row['filter_reasons']:flags[i]|=1<<reasons.index(reason)
    with Path(path).open('xb') as f:
        np.savez_compressed(f,native_prediction=native,filter_flags=flags,reason_names=np.asarray(reasons,dtype='U64'))
        f.flush();os.fsync(f.fileno())
    return Path(path).stat().st_size


def evaluate(directory,backend,panel,config,gen,recorder,tag,*,hard_stop_unix=None):
    from .runner_data import RequestPlanDataset,make_loader
    from .training import prepare_detection_batch
    from .detectors import decode_native_eval_rows
    from .postprocess import postprocess_decoded_person
    from .coco_metrics import coco_player_ap
    from .metrics import player_ignore_ap
    from .records import capture_rng_state,restore_rng_state
    root=Path(directory)/'evaluation'/tag
    if root.exists():
        if (root/'summary.json').exists():
            result=json.loads((root/'summary.json').read_bytes())
            if result['update']!=backend.stage_update or result['images']!=750:raise RuntimeError('Completed evaluation binding differs')
            return result
        root=Path(directory)/'evaluation'/(tag+'_retry1')
    root.mkdir(parents=True,exist_ok=False);(root/'raw').mkdir()
    saved_rng=capture_rng_state(include_cuda=True);evalgen=torch.Generator(device='cpu').manual_seed(170001)
    requests=[{'uid':uid,'domain':'R','epoch':0,'exposure_index':i,'augmentation_seed':0} for i,uid in enumerate(panel['val_uids'])]
    dataset=RequestPlanDataset(panel['val_catalogue'],requests,view='val',real_reader_options={'max_cached_clips':20,'max_cached_rows':600000})
    loader=make_loader(dataset,batch_size=config['microbatch_size'],num_workers=config['num_workers'],generator=evalgen,pin_memory=True,allow_initialized_cuda=True)
    loader.timeout=300
    iterator=iter(loader);network=backend.ema_network_for_eval();images={};aux={};pending=[];raw_bytes=0;number=0;started=time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=1) as writer:
            for cpu_batch in iterator:
                if stop_requested(config.get('control_group',Path(directory).parent),directory):raise RuntimeError('USER_STOP during evaluation')
                if time.time()>min(hard_stop_unix or float('inf'),datetime.fromisoformat(DEADLINE).timestamp()):raise TimeoutError('LIMIT_STOP during evaluation')
                check_write_space(config,2*config['raw_upper_per_image_bytes'])
                batch=prepare_detection_batch(cpu_batch,device='cuda:0',non_blocking=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['numeric_mode']=='amp_bf16'):
                    output=network.evaluation_output(batch['img'])
                prediction=output[0].detach().float()
                rows=decode_native_eval_rows((prediction,[x.float() for x in output[1]]))
                raw_native=prediction.cpu().numpy()
                for i,uid in enumerate(batch['uids']):
                    if uid!=panel['val_uids'][number]:raise RuntimeError('Validation UID order changed')
                    processed=postprocess_decoded_person(uid,rows[i],geometry=batch['geometry'][i])
                    gt=deepcopy(panel['val_ground_truth'][uid]);gt.pop('role_audit',None)
                    known=gt.pop('known_non_target_auxiliary_ignore',[])
                    if gt['ignore']:raise RuntimeError('True ignore prevents strict standard COCO evaluation')
                    images[uid]={**gt,'predictions':processed['predictions']}
                    aux[uid]={**images[uid],'ignore':known}
                    path=root/'raw'/f'{number+1:06d}.npz'
                    native=raw_native[i].copy()
                    if native.nbytes+2*len(processed['raw_predictions'])+65536>config['raw_upper_per_image_bytes']:
                        raise RuntimeError('Native prediction shape exceeds frozen raw upper bound')
                    pending.append(writer.submit(_raw_write,path,native,processed['raw_predictions']))
                    if len(pending)>=2:
                        size=pending.pop(0).result()
                        if size>config['raw_upper_per_image_bytes']:raise RuntimeError('Raw serialized prediction exceeds bound')
                        raw_bytes+=size
                    counts=Counter(s for row in processed['raw_predictions'] for s in row['filter_reasons'])
                    append(root/'predictions.jsonl',{'uid':uid,'geometry':batch['geometry'][i],'raw_prediction_file':str(path.relative_to(root)),
                        'native_output_dtype':str(output[0].dtype),'raw_prediction_count':processed['input_prediction_count'],
                        'filter_reason_counts':dict(counts),'predictions':processed['predictions'],'postprocess_definition':processed['definition'],
                        'postprocess_binding':processed['binding'],'inference_numeric_mode':config['numeric_mode']})
                    number+=1
                if number%100==0:print(json.dumps({'event':'eval_progress','tag':tag,'images':number,'total':750}),flush=True)
            for future in pending:
                size=future.result()
                if size>config['raw_upper_per_image_bytes']:raise RuntimeError('Raw file exceeds upper bound')
                raw_bytes+=size
        if number!=750:raise RuntimeError('Incomplete validation images')
        result=coco_player_ap(images,image_uid_order=panel['val_uids'])
        auxiliary=player_ignore_ap(aux,image_uid_order=panel['val_uids'])
        write_once(root/'coco_results.json',result);write_once(root/'auxiliary_ignore_AP300.json',auxiliary)
        summary={'tag':tag,'update':backend.stage_update,'main_metrics':result['metrics'],'raw_prediction_bytes':raw_bytes,
                 'images':number,'wall_seconds':time.monotonic()-started,'evaluated_model':'stage_EMA_at_recorded_update','official_dataset_benchmark':False}
        write_once(root/'summary.json',summary);recorder.event({'kind':'evaluation',**summary})
        print(json.dumps({'event':'evaluation_complete',**summary}),flush=True)
        return summary
    finally:
        _shutdown(iterator);del network,iterator,loader,dataset
        torch.cuda.empty_cache();restore_rng_state(saved_rng,allow_cuda=True)


def run_one(group,config,*,resume_receipt=None):
    from .runner_data import real_panel,RequestPlanDataset,make_loader
    from .runtime_backend import NativeBBoxBackend
    from .data import CycleSampler,PrefetchSampler
    from .records import RunRecorder,CheckpointReceipt,load_checkpoint
    group=Path(group);panel=real_panel(REAL)
    if len(panel['train_uids'])!=2100 or len(panel['val_uids'])!=750:raise RuntimeError('Panel changed')
    if any(v['ignore'] for v in panel['val_ground_truth'].values()):raise RuntimeError('True val ignore not allowed for strict COCO')
    hashes=json.loads((group/'code_bindings.json').read_bytes())
    for name,digest in hashes.items():
        if sha(Path(__file__).parent/name)!=digest:raise RuntimeError('Frozen code changed')
    admitted=json.loads((group/'admission.json').read_bytes())['source_bindings']['real_panel']
    if canonical(panel['binding'])!=canonical(admitted):raise RuntimeError('Real data binding changed since preflight')
    bindings=run_bindings(panel,config,hashes)
    directory=output_group(group)/f"R{config['epochs']}_seed17"
    manifest={'schema_version':'bbox_calibration_manifest_v1','run_id':directory.name,'source_bindings':bindings,'protocol':config,'created_under_group':str(group),'output_group':str(directory.parent),'storage_mode':config.get('storage_mode','legacy_d')}
    seed_all(17)
    backend=NativeBBoxBackend.from_public(WEIGHTS,model_size='s',stage_updates=config['updates'],final_stage=True,
        microbatch_size=config['microbatch_size'],numeric_mode=config['numeric_mode'],device='cuda:0',allow_initialized_cuda=True)
    live=PrefetchSampler(CycleSampler(panel['train_uids'],task='detect',domain='R',seed=17))
    gen=torch.Generator(device='cpu').manual_seed(170017);start=0
    if resume_receipt is None:
        recorder=RunRecorder(directory,manifest);(directory/'checkpoints').mkdir();start_time=time.time()
        save_boundary(directory,backend,live,bindings,gen,recorder,'initial')
        journal_run(directory,config,'STARTED',0,{'state':'initialized_from_public_not_yet_updated','stage0_and_periodic_evaluation_retained':True})
    else:
        receipt=CheckpointReceipt(**resume_receipt);bundle=load_checkpoint(receipt,expected_bindings=bindings)
        recorder=RunRecorder.resume(directory,expected_manifest=manifest,checkpoint_committed_updates=bundle['committed_updates'])
        restored=backend.restore_bundle(bundle,samplers={'R':live},expected_bindings=bindings)
        gen.set_state(restored['extra_state']['loader_generator_state']);start=backend.stage_update;start_time=time.time()
    status(directory,{'state':'RUNNING','pid':os.getpid(),'create_time':psutil.Process().create_time(),'update':start,'updates':config['updates']})
    iterator=None;latest=start
    runtime_deadline=min(start_time+config['max_run_seconds'],datetime.fromisoformat(config['deadline_utc']).timestamp())
    try:
        if start==0:evaluate(directory,backend,panel,config,gen,recorder,'step000000_PT',hard_stop_unix=runtime_deadline)
        elif start%1000==0 or start==config['updates']:evaluate(directory,backend,panel,config,gen,recorder,f'step{start:06d}',hard_stop_unix=runtime_deadline)
        if start==config['updates']:
            end=evaluate(directory,backend,panel,config,gen,recorder,f'step{start:06d}',hard_stop_unix=runtime_deadline)
            status(directory,{'state':'COMPLETE','update':start,'updates':config['updates'],'terminal_evaluation':end})
            recorder.event({'kind':'run_complete_after_explicit_restore','update':start})
            journal_run(directory,config,'COMPLETE',start,end)
            return {'state':'COMPLETE','directory':str(directory),'terminal':end}
        requests=_requests(panel,live,config['updates'],start)
        ds=RequestPlanDataset(panel['train_catalogue'],requests,view='train',real_reader_options={'max_cached_clips':20,'max_cached_rows':600000})
        loader=make_loader(ds,batch_size=config['microbatch_size'],num_workers=config['num_workers'],generator=gen,pin_memory=True,allow_initialized_cuda=True)
        loader.timeout=300
        generator_before=gen.get_state();iterator=iter(loader)
        if start:gen.set_state(generator_before) # worker RNG不用于增强；恢复保存的独立loader Generator，不额外推进。
        for j in range(start+1,config['updates']+1):
            if stop_requested(group,directory):
                save_boundary(directory,backend,live,bindings,gen,recorder,'user_stop')
                status(directory,{'state':'USER_STOP','update':backend.stage_update});journal_run(directory,config,'USER_STOP',backend.stage_update);return {'state':'USER_STOP'}
            if time.time()-start_time>config['max_run_seconds'] or datetime.now(timezone.utc)>=datetime.fromisoformat(config['deadline_utc']):
                save_boundary(directory,backend,live,bindings,gen,recorder,'limit_stop')
                status(directory,{'state':'LIMIT_STOP','update':backend.stage_update});journal_run(directory,config,'LIMIT_STOP',backend.stage_update);return {'state':'LIMIT_STOP'}
            check_write_space(config,2*int(config.get('checkpoint_bytes',120*1024**2)))
            if torch.cuda.mem_get_info()[0]<1024**3:raise RuntimeError('LIMIT_STOP: GPU free below1GiB')
            window=live.reserve(16);began=time.monotonic();batches=_microbatches(iterator,16//config['microbatch_size'])
            actual=[row for batch in batches for row in batch['sample_lineage']]
            if [x['request'] for x in actual]!=window['samples']:raise RuntimeError('Dispatched vs committed sampler request mismatch')
            for batch in batches:
                if not all(g.get('training_input',{}).get('view')=='train' for g in batch['geometry']):raise RuntimeError('Missing actual approved augmentation')
            metrics=backend.step_update(batches,update=j);live.commit(window['token']);latest=j
            metrics.update(phase='final',domain='R',gradient_norm=metrics['gradient_norm_before_clip'],R_exposures=j*16,S_exposures=0,
                wall_seconds=time.monotonic()-began,geometry=[g for batch in batches for g in batch['geometry']],
                maximum_cuda_allocated=torch.cuda.max_memory_allocated(),maximum_cuda_reserved=torch.cuda.max_memory_reserved())
            recorder.step(metrics)
            if checkpoint_due(j,config['updates'],config['checkpoint_every']):save_boundary(directory,backend,live,bindings,gen,recorder,'final' if j==config['updates'] else 'interval')
            if j%25==0 or j==config['updates']:
                production_closed()
                status(directory,{'state':'RUNNING','pid':os.getpid(),'create_time':psutil.Process().create_time(),'update':j,'updates':config['updates'],'loss':metrics['loss']})
                print(json.dumps({'event':'update','run':directory.name,'update':j,'updates':config['updates'],'loss':metrics['loss'],'update_s':metrics['wall_seconds']}),flush=True)
            del batches
            if j%1000==0 or j==config['updates']:evaluate(directory,backend,panel,config,gen,recorder,f'step{j:06d}',hard_stop_unix=runtime_deadline)
        end=evaluate(directory,backend,panel,config,gen,recorder,f"step{config['updates']:06d}",hard_stop_unix=runtime_deadline)
        status(directory,{'state':'COMPLETE','update':latest,'updates':config['updates'],'terminal_evaluation':end})
        recorder.event({'kind':'run_complete','update':latest,'finished_at_utc':now()})
        journal_run(directory,config,'COMPLETE',latest,end)
        return {'state':'COMPLETE','directory':str(directory),'terminal':end}
    except BaseException as exc:
        error={'state':'FAIL_STOP','update':latest,'backend_stage_update':backend.stage_update,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
        try:
            if not backend.failed and backend.stage_update==latest:
                live.cancel_pending();save_boundary(directory,backend,live,bindings,gen,recorder,'controlled_failure')
        except BaseException as save_error:error['failure_checkpoint_error']=repr(save_error)
        if backend.failed or backend.stage_update!=latest:
            try:
                from .checkpoint_state import _clone_cpu
                from .records import capture_rng_state
                evidence={'schema_version':'uncommitted_failure_evidence_v1','resume_allowed':False,'failure':backend.failure,'last_committed_update':latest,
                    'raw':backend.model.state_dict(),'ema':backend.ema_state,'optimizer':backend.optimizer.state_dict(),'rng':capture_rng_state(include_cuda=True),'sampler':live.state_dict(),'source_bindings':bindings}
                with (directory/'failed_uncommitted_state.pt').open('xb') as f:
                    torch.save(_clone_cpu(evidence,allow_cuda=True,cuda_initialized=True),f);f.flush();os.fsync(f.fileno())
            except BaseException as evidence_error:error['uncommitted_evidence_error']=repr(evidence_error)
        if stop_requested(group,directory):error['state']='USER_STOP'
        elif isinstance(exc,TimeoutError):error['state']='LIMIT_STOP'
        status(directory,error)
        journal_run(directory,config,error['state'],latest,error)
        try:recorder.event({'kind':'failure',**error})
        except BaseException:pass
        if error['state'] in ('USER_STOP','LIMIT_STOP'):return {'state':error['state'],'directory':str(directory),'update':latest}
        raise
    finally:
        _shutdown(iterator);del backend;torch.cuda.empty_cache()


def journal_run(directory,config,state,updates,result=None):
    text=(f"\n## RUN：{Path(directory).name} {state}\n\n"
          f"UTC {now()}。用途：SoccerNet来源内R预算校准，不是合成贡献结果。公开yolo11s起点、seed17，"
          f"train2100图/val750图；目标{config['updates']}更新（名义{config['epochs']}轮），实际已提交{updates}更新/{updates*16}R曝光/0S。"
          f"输入960、有效batch16、micro{config['microbatch_size']}、{config['numeric_mode']}、workers{config['num_workers']}。"
          f"SGD Nesterov/momentum.937/weight decay .0005（bias/norm0），LR终阶段peak.001→.00001、warmup100，clip10，"
          f"既定HSV/仿射/翻转及默认删框、role_policy_v2；每{config['checkpoint_every']}更新完整raw/EMA/optimizer/RNG/sampler、PT/每1000/末步val全部原始预测保留。\n\n"
          f"状态与实际结果：{json.dumps(plain(result),ensure_ascii=False)}。"
          f"证据：{Path(directory).as_posix()}/run.json、steps.jsonl、events.jsonl、checkpoints、evaluation。"
          f"只用本次末步EMA做预算选择，不取best；R50/R100两独立结果未全闭合前不能选预算。\n")
    with (RESEARCH/'docs/exp/EXPERIMENT_LOG.md').open('ab') as f:f.write(text.encode());f.flush();os.fsync(f.fileno())


def controller(group):
    group=Path(group).absolute()
    admission=json.loads((group/'admission.json').read_bytes())
    if admission.get('status')!='PASS':raise RuntimeError('No actual admitted preflight')
    with gpu_lock(group):
        setup_cuda(admission);group_status(group,{'state':'RUNNING','pid':os.getpid(),'create_time':psutil.Process().create_time()})
        table=json.loads((group/'job_table.json').read_bytes())
        epochs_list=[int(job['epochs']) for job in table['jobs']]
        if epochs_list not in ([50,100],[100]):raise RuntimeError('Only original finite independent calibration jobs are allowed')
        if table.get('storage_revision_sha256'):
            validate_storage_revision(group,admission,json.loads((group/'code_bindings.json').read_bytes()))
        if epochs_list==[100] and sha(group/'R50_reference.json')!=table['R50_reference_sha256']:raise RuntimeError('Frozen R50 reference changed')
        results=[validate_R50_reference(group,admission)] if epochs_list==[100] else []
        try:
            for epochs in epochs_list:
                cfg=protocol(admission,epochs);cfg['checkpoint_bytes']=admission['checkpoint_bytes'];cfg['source_admission_sha256']=sha(group/'admission.json');cfg['storage_revision_sha256']=table.get('storage_revision_sha256');cfg['control_group']=str(group);cfg['output_group']=str(output_group(group))
                group_record(group,f'R{epochs}_protocol.json',cfg)
                append(output_group(group)/'controller_events.jsonl',{'kind':'job_admitted','epochs':epochs,'protocol':cfg,'at_utc':now()})
                result=run_one(group,cfg);results.append(result)
                if result['state']!='COMPLETE':group_status(group,{'state':result['state'],'results':results});return
            a,b=[item['terminal']['main_metrics']['AP50_95'] for item in results]
            selected=100 if b>a else 50
            choice={'state':'CALIBRATION_COMPLETE','chosen_R_epochs':selected,'R50_AP50_95':a,'R100_AP50_95':b,'selection_rule':'100_if_strictly_higher_else50','results':results,
                    'formal_s15_admitted':False,'followup':'stop_for_storage_gate_no_test_no_extra_runs'}
            group_record(group,'budget_choice.json',choice);group_status(group,choice)
        except BaseException as exc:
            group_status(group,{'state':'FAIL_STOP','results':results,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
            raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['preflight','freeze','controller','resume']);parser.add_argument('--group',required=True);parser.add_argument('--epochs',type=int,choices=[50,100]);parser.add_argument('--checkpoint-update',type=int)
    args=parser.parse_args();group=Path(args.group).absolute()
    if not group.is_relative_to(RESEARCH/'runs'):raise ValueError('Group must remain research/runs')
    if args.command=='preflight':
        from .preflight import preflight_calibration
        group.mkdir(parents=True,exist_ok=False)
        with gpu_lock(group):print(json.dumps(preflight_calibration(group,REAL,WEIGHTS,gpu_release_confirmed=True),default=plain),flush=True)
    elif args.command=='freeze':print(json.dumps(freeze(group,epochs=args.epochs)),flush=True)
    elif args.command=='controller':
        try:controller(group)
        finally:
            if (group/'RUN_LOCATION.json').exists():
                if (RESEARCH/'.research_gpu.lock').exists():raise RuntimeError('Cannot seal control while GPU lock is active')
                from .output_storage import capture_control
                print(json.dumps({'event':'control_sealed',**capture_control(group,research_root=RESEARCH,output_base=ARCHIVE_ROOT/'runs',final=True)}),flush=True)
    else:
        if args.epochs is None or args.checkpoint_update is None:raise ValueError('Explicit resume requires epochs and trusted update')
        config=json.loads((group/f'R{args.epochs}_protocol.json').read_bytes())
        index=output_group(group)/f'R{args.epochs}_seed17'/'checkpoints.jsonl'
        rows=[json.loads(line) for line in index.read_text(encoding='utf8').splitlines()]
        chosen=[r for r in rows if r['update']==args.checkpoint_update]
        if not chosen:raise ValueError('No owned checkpoint receipt at requested boundary')
        with gpu_lock(group):
            setup_cuda(json.loads((group/'admission.json').read_bytes()));print(json.dumps(run_one(group,config,resume_receipt=chosen[-1]['receipt'])),flush=True)

if __name__=='__main__':main()