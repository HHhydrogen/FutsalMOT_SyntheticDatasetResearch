"""仅接续用户明确批准的BBox剩余19项；原43执行源码不变。"""
from pathlib import Path
import json,hashlib,os,sys,shutil,time,traceback
R=Path('D:/projects/FutsalMOT_SyntheticDatasetResearch')
OLD=R/'runs/bbox_confirmation_suite_20261104_a02'
OLDG=Path('G:/FutsalMOT_Training/runs')/OLD.name
NEW=R/'runs/bbox_confirmation_suite_20261105_a01'
G=Path('G:/FutsalMOT_Training/runs')/NEW.name
sys.path.insert(0,str((NEW if NEW.exists() else OLD)/'code'))
from futsalmot import bbox_confirmation_lifecycle as life
STOP_VERIFICATION={'path':str(OLDG/'user_stop_verification.json'),'sha256':'e14a083ca18f447b8c2eb9db20461ece70f431e60f28cf5e88cc4b7065002085'}
OLD_SHA='02cd3cb88efd7b2d98f40a4137195c4d776496eaab15e2740063401adf3b7fbd'

def read(p):return json.loads(Path(p).read_bytes())
def copied(p,target):
    life.no_links(p);life.no_links(target,exists=False);assert not target.exists()
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    assert life.sha(p)==life.sha(target)
    return {'source_reference':life.reference(p),'destination':str(target),'bytes':p.stat().st_size,'sha256':life.sha(p)}
def verify_manifest(p):
    m=read(p);assert m['state']=='SEALED' and m['scope']=='control_snapshot_only'
    assert len(m['files'])==m['file_count']
    for row in m['files']:
        t=p.parent/row['path'];life.no_links(t);assert t.stat().st_size==row['bytes'] and life.sha(t)==row['sha256']
    return m

def prior():
    v=life.read_reference(STOP_VERIFICATION);assert v['state']=='VERIFIED_USER_STOP_CLOSED' and v['accepted_complete']==29 and v['user_stopped']==1 and v['not_dispatched']==18
    assert v['all_known_owned_exit_verified'] and v['GPU_lock_retired'] and not v['unknown_exact_script_writers']
    assert life.sha(OLD/'suite_request.json')==OLD_SHA
    suite=life.load_suite(OLD);assert read(OLD/'suite_result.json')['completed']==[r['job_id'] for r in suite['jobs'][:29]]
    assert read(OLD/'STOP.json')['schema']=='bbox_user_scheduled_stop_signal_v1'
    assert life.read_reference(v['stop_reference'])==read(OLD/'STOP.json')
    for row in v['rows'][:30]:
        result=life.read_reference(row['result_reference']);ll=life.read_reference(row['lifetime_reference']);verify_manifest(Path(row['control_manifest_reference']['path']))
        assert not life.wait_owned(ll['owned'],seconds=0) and not ll['owned_still_alive'] and ll['GPU_lock_retired']
        assert result['state']==row['accepted_state']
    import psutil
    for pid,ct in [(13564,1793788441.4012089),(34072,1793788441.419639),(36468,1793795879.2283459),(34756,1793795879.2487752)]:
        assert not psutil.pid_exists(pid) or abs(psutil.Process(pid).create_time()-ct)>.01
    assert not(R/'.research_gpu.lock').exists()
    return suite,v

def prepare():
    assert not NEW.exists() and not G.exists(),'接续组已存在，不可覆盖或重放'
    old,v=prior();resources19=life.check_resources(remaining=19)
    prepared=life.prepare_suite(NEW,spec_reference=old['spec_reference'],completion_reference=old['completion_reference'],plan_reference=old['plan_reference'],observer_source=OLD/'observe.py')
    suite=life.load_suite(NEW);assert suite['source_code_sha256']==old['source_code_sha256'] and suite['jobs']==old['jobs'] and suite['observer_sha256']==old['observer_sha256']
    imports=[]
    for row in old['jobs'][:29]:
        jid=row['job_id'];d=OLD/'jobs'/jid;o=OLDG/'jobs'/jid;dest=NEW/'jobs'/jid;newout=G/'jobs'/jid
        result=read(d/'result.json');ll=read(d/'lifetime_receipt.json');assert result['state']=='COMPLETE' and ll['state']=='VERIFIED_EXIT'
        assert result['summary']['images']==750 and result['summary']['update']==1500 and result['summary']['evaluated_model']=='stage_EMA_at_recorded_update'
        assert life.read_reference(result['summary_reference'])==result['summary']
        assert ll['returncode']==0 and ll['precise_GetProcessTimes'] and 0<ll['full_process_seconds']<=900
        files=[]
        for p in d.iterdir():
            assert p.is_file() and p.name in {'job_request.json','runtime_limit.json','dispatch_wrapper.json','lifetime_receipt.json','result.json','STOP.json'}
            files.append(copied(p,dest/p.name))
        files.append(copied(o/'result.json',newout/'result.json'))
        for p in (o/'control').iterdir():
            assert p.is_file();files.append(copied(p,newout/'control'/p.name))
        verify_manifest(newout/'control/CONTROL_MANIFEST.json')
        imports.append({**row,'execution_control':str(d),'execution_output':str(o),'copied_receipts_only':True,'no_new_inference':True,'source_result_reference':life.reference(d/'result.json'),'summary_reference':result['summary_reference'],'files':files})
    remaining,done=life.remaining_jobs(NEW,suite);assert len(done)==29 and remaining==suite['jobs'][29:] and len(remaining)==19
    assert not(NEW/'jobs'/remaining[0]['job_id']).exists() and read(OLD/'jobs'/remaining[0]['job_id']/'result.json')['state']=='USER_STOP'
    handoff={'schema':'bbox_confirmation_explicit_remaining_v1','state':'FROZEN','user_instruction':'2026-11-05 完成剩下部分的验证','prior_user_stop_verification_reference':STOP_VERIFICATION,'prior_suite_reference':life.reference(OLD/'suite_request.json'),'new_suite_reference':life.reference(NEW/'suite_request.json'),'imported_complete_count':29,'new_evaluation_count':19,'imported_complete':imports,'remaining_jobs':remaining,'fresh_attempt_for_user_stopped_job':remaining[0]['job_id'],'prior_partial_directory':str(OLDG/'jobs'/remaining[0]['job_id']),'partial_kept_unchanged':True,'no_implicit_retry':True,'no_numeric_or_postprocess_change':True,'source43_exact':suite['source_code_sha256'],'actual_remaining_resource_gate':resources19,'new_full48_prepare_resource_gate':prepared['resources'],'created_at_utc':life.utc()}
    life.write_new(G/'continuation_handoff.json',handoff)
    copied(Path(__file__),G/'continue_remaining.py')
    life.write_new(G/'readiness_verification.json',{'state':'PREPARED_NOT_DISPATCHED','user_explicit_continue':True,'handoff_reference':life.reference(G/'continuation_handoff.json'),'helper_reference':life.reference(G/'continue_remaining.py'),'suite_reference':life.reference(NEW/'suite_request.json'),'imported29_actual_receipts_verified':True,'remaining19_resources':resources19,'fresh_CPU_training_tests':False,'fresh_GPU_probe':False,'diagnostic_updates':0,'old_complete_not_reexecuted':True,'original_partial_retained':True,'same43_and_observer':True,'created_at_utc':life.utc()})
    print(json.dumps({'readiness_reference':life.reference(G/'readiness_verification.json'),'suite_reference':life.reference(NEW/'suite_request.json'),'handoff_reference':life.reference(G/'continuation_handoff.json'),'remaining19_resources':resources19},ensure_ascii=False),flush=True)

def verify_ready():
    ready=read(G/'readiness_verification.json');assert ready['state']=='PREPARED_NOT_DISPATCHED'
    assert life.sha(G/'continue_remaining.py')==ready['helper_reference']['sha256']
    handoff=life.read_reference(ready['handoff_reference']);suite=life.load_suite(NEW)
    assert handoff['new_suite_reference']==life.reference(NEW/'suite_request.json') and handoff['source43_exact']==suite['source_code_sha256']
    assert handoff['remaining_jobs']==suite['jobs'][29:] and len(handoff['imported_complete'])==29
    for row in handoff['imported_complete']:
        assert life.read_reference(row['source_result_reference'])['state']=='COMPLETE'
        for f in row['files']:
            p=Path(f['destination']);assert p.stat().st_size==f['bytes'] and life.sha(p)==f['sha256']
    remaining,done=life.remaining_jobs(NEW,suite);assert done==[x['job_id'] for x in suite['jobs'][:29]] and remaining==handoff['remaining_jobs']
    return suite,handoff

def run():
    import psutil
    suite,handoff=verify_ready();assert not(NEW/'suite_dispatch.json').exists() and not(G/'result.json').exists()
    life.check_resources(remaining=19);p=psutil.Process();done=[r['job_id'] for r in suite['jobs'][:29]];newdone=[]
    life.write_new(NEW/'suite_dispatch.json',{'pid':p.pid,'create_time':p.create_time(),'command':p.cmdline(),'started_at_utc':life.utc(),'suite_reference':life.reference(NEW/'suite_request.json'),'fixed_evaluations':48,'inherited_evaluations':29,'new_evaluations':19,'continuation_reference':life.reference(G/'continuation_handoff.json'),'automatic_retry':False})
    try:
        for row in handoff['remaining_jobs']:
            life.prepare_job(NEW,row['job_id'])
            if life.observe_job(NEW,row['job_id'])!=0:raise RuntimeError('接续子项失败/停止/超界/未闭合；保留证据，不自动重试')
            done.append(row['job_id']);newdone.append(row['job_id'])
        assert len(done)==48 and len(set(done))==48 and len(newdone)==19
        result={'state':'COMPLETE','completed':done,'imported_complete':done[:29],'new_complete':newdone,'distinct_model_matches':48,'new_model_matches_this_continuation':19,'continuation_reference':life.reference(G/'continuation_handoff.json'),'finished_at_utc':life.utc(),'inference_only':True,'optimizer_updates':0,'no_retry':True}
        life.write_new(NEW/'suite_result.json',result);sealed=life.seal_suite(NEW,suite)
        life.write_new(G/'result.json',{**result,'suite_control_manifest_reference':sealed})
        return 0
    except BaseException as exc:
        result={'state':'FAIL_STOP','completed':done,'imported_complete':done[:29],'new_complete':newdone,'error':str(exc),'traceback':traceback.format_exc(),'finished_at_utc':life.utc(),'no_retry':True,'continuation_reference':life.reference(G/'continuation_handoff.json')}
        if not(NEW/'suite_result.json').exists():life.write_new(NEW/'suite_result.json',result)
        else:life.write_new(G/'suite_seal_failure.json',result)
        if not(G/'result.json').exists():life.write_new(G/'result.json',result)
        return 1

if __name__=='__main__':
    args=sys.argv[1:]
    if args==['--prepare']:prepare()
    elif args==['--suite']:raise SystemExit(run())
    else:raise SystemExit('显式--prepare或--suite；不默认启动')
