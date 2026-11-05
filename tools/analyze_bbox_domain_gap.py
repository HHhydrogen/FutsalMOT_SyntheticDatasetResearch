"""固定train/dev标注域差；CPU单核/最低I/O/限速，不媒体、不模型、不CUDA。"""
from __future__ import annotations
from pathlib import Path
from array import array
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv,hashlib,json,math,os,shutil,sys,time
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import psutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from futsalmot.adapters import RealBBoxReader,_synthetic_bbox_row,_source_stat
OUT=Path('G:/FutsalMOT_Training/runs/bbox_domain_gap_train_dev_20261104_a01')
SINDEX=Path('G:/FutsalMOT_Dataset/synthetic/dataset.json')
RINDEX=Path('G:/FutsalMOT_Dataset/real/soccernet/dataset.json')
CACHE=ROOT/'cache/synthetic_bbox_960_v2'
EVAL=ROOT/'runs/bbox_confirmation_suite_20261104_a02'
PLANS={'EXPERIMENTS.md':'32c9415b2d81de7fce6a9221af72495822ffe3b70e537004b7bb64af02b065a2','PAPER.md':'d48cb4160a513773cde0e910fe1c45c7eb5a63bf058871eafd38e9ad27b71c17'}
START=time.monotonic();LIMIT=600;SOURCE_PROOFS={};TOTAL_READ=Counter();PEAK=0

def now():return datetime.now(timezone.utc).isoformat()
def guard():
 global PEAK
 if time.monotonic()-START>LIMIT:raise TimeoutError('analysis600s limit; partial retained; evaluation untouched')
 if (OUT/'STOP.json').exists():raise RuntimeError('analysis STOP requested; evaluation untouched')
 mem=psutil.Process().memory_info();PEAK=max(PEAK,getattr(mem,'peak_wset',mem.rss))
 if mem.rss>1200*1024**2:raise MemoryError('analysis RSS exceeds1.2GiB')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):p=Path(p);return {'path':str(p.absolute()),'bytes':p.stat().st_size,'sha256':sha(p)}
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def prove(p):
 p=Path(p).absolute();identity=tuple(_source_stat(p));SOURCE_PROOFS[str(p)]=identity;return identity

def lines(p):
 """单文件平均速率：G2MiB/s、D8MiB/s；单行最多1MiB。"""
 p=Path(p).absolute();before=prove(p);rate=(2 if p.drive.lower()=='g:' else 8)*1024**2;began=time.monotonic();count=0
 with p.open('rb',buffering=65536) as f:
  while True:
   raw=f.readline(1024**2+1)
   if not raw:break
   if len(raw)>1024**2:raise ValueError('annotation line exceeds1MiB')
   count+=len(raw);TOTAL_READ[p.drive]+=len(raw);guard()
   delay=count/rate-(time.monotonic()-began)
   if delay>0:time.sleep(min(delay,.5))
   yield raw
 assert tuple(_source_stat(p))==before

def small(p,maximum=64*1024**2):
 p=Path(p);before=prove(p);assert before[2]<=maximum
 # 索引/manifest是小文本；原缓存manifest33MiB不含任何数组像素。
 with p.open('rb') as f:raw=f.read(maximum+1)
 TOTAL_READ[p.drive]+=len(raw);assert tuple(_source_stat(p))==before
 return json.loads(raw)

def union_area(rects):
 if not rects:return 0.
 xs=sorted(set(x for b in rects for x in (b[0],b[2])));total=0.
 for left,right in zip(xs,xs[1:]):
  intervals=sorted((b[1],b[3]) for b in rects if b[0]<right and b[2]>left)
  if not intervals:continue
  bottom,top=intervals[0];length=0.
  for lo,hi in intervals[1:]:
   if lo>top:length+=top-bottom;bottom,top=lo,hi
   else:top=max(top,hi)
  total+=(right-left)*(length+top-bottom)
 return total

def overlaps(boxes):
 n=len(boxes);iou=[0.]*n;ioa=[0.]*n;coverage=[0.]*n
 for i,a in enumerate(boxes):
  area=(a[2]-a[0])*(a[3]-a[1]);rects=[]
  for j,b in enumerate(boxes):
   if i==j:continue
   x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
   if x2<=x1 or y2<=y1:continue
   inter=(x2-x1)*(y2-y1);other=(b[2]-b[0])*(b[3]-b[1]);rects.append((x1,y1,x2,y2));iou[i]=max(iou[i],inter/(area+other-inter));ioa[i]=max(ioa[i],inter/area)
  coverage[i]=union_area(rects)/area
  assert 0<=iou[i]<=1+1e-9 and 0<=ioa[i]<=coverage[i]+1e-9<=1+2e-9
 return iou,ioa,coverage

def describe(values):
 a=np.asarray(values,dtype=np.float64)
 if not len(a):return {'n':0}
 qs=[.01,.1,.25,.5,.75,.9,.99];v=np.quantile(a,qs)
 return {'n':len(a),'mean':float(a.mean()),'population_std':float(a.std()),'min':float(a.min()),'max':float(a.max()),**{f'q{round(q*100):02d}':float(x) for q,x in zip(qs,v)}}

def distances(a,b):
 a=np.sort(np.asarray(a,dtype=np.float64));b=np.sort(np.asarray(b,dtype=np.float64));support=np.unique(np.concatenate((a,b)));fa=np.searchsorted(a,support,side='right')/len(a);fb=np.searchsorted(b,support,side='right')/len(b)
 return {'KS_descriptive_no_pvalue':float(np.max(np.abs(fa-fb))),'Wasserstein1_in_original_units':float(np.sum(np.abs(fa[:-1]-fb[:-1])*np.diff(support)))}

def self_check():
 assert union_area([(0,0,6,10),(4,0,10,10)])==100
 a,b,c=overlaps([(0,0,10,10),(0,0,6,10),(4,0,10,10)]);assert a[0]==.6 and b[0]==.6 and c[0]==1.
 assert overlaps([(0,0,10,10),(10,0,20,10)])==([0.,0.],[0.,0.],[0.,0.])
 assert distances([0,2],[1,3])['Wasserstein1_in_original_units']==1.
 assert distances([0,2],[0,2])['KS_descriptive_no_pvalue']==0

class Pool:
 def __init__(self,name):
  self.name=name;self.frame=defaultdict(lambda:array('d'));self.target=defaultdict(lambda:array('d'));self.groups=defaultdict(lambda:defaultdict(lambda:array('d')));self.cameras=defaultdict(lambda:defaultdict(lambda:array('d')));self.shapes=Counter();self.absent=0;self.raw_targets=0;self.invalid=0;self.nontarget=0;self.raw_tail=Counter();self.uid_hash=hashlib.sha256();self.count=0
 def add(self,uid,group,clip,frame,w,h,targets,absent,ignores,writer,camera=None):
  guard();valid=[t for t in targets if t['bbox_supervision_valid']];self.raw_targets+=len(targets);self.invalid+=len(targets)-len(valid);self.absent+=len(absent);self.nontarget+=sum(t.get('ignore_reason')=='non_target_role' for t in ignores);self.shapes[f'{w}x{h}']+=1
  boxes=[t['clipped_xyxy'] for t in valid];ious,ioas,covers=overlaps(boxes);ratio=960/max(w,h);heights=[];fills=[]
  for i,(t,b) in enumerate(zip(valid,boxes)):
   bw,bh=b[2]-b[0],b[3]-b[1];area=bw*bh;heights.append(bh*ratio)
   values={'width960':bw*ratio,'height960':bh*ratio,'sqrt_area960':math.sqrt(area)*ratio,'area_image_fraction':area/(w*h),'height_image_fraction':bh/h,'max_bbox_IoU':ious[i],'max_bbox_IoA':ioas[i],'union_bbox_IoA':covers[i],'touches_image_border':float(b[0]<=0 or b[1]<=0 or b[2]>=w or b[3]>=h)}
   if 'visible_pixel_count' in t:values['visible_mask_over_tight_bbox_fill']=t['visible_pixel_count']/area;fills.append(values['visible_mask_over_tight_bbox_fill'])
   for k,v in values.items():self.target[k].append(v)
   if 'original_gt_row' in t:self.raw_tail[','.join(t['original_gt_row'][7:])]+=1
  n=len(valid);metrics={'people':n,'mean_height960':sum(heights)/n if n else 0.,'mean_union_bbox_IoA':sum(covers)/n if n else 0.,'fraction_targets_union_IoA_ge_0_1':sum(x>=.1 for x in covers)/n if n else 0.,'fraction_targets_union_IoA_ge_0_3':sum(x>=.3 for x in covers)/n if n else 0.,'fraction_targets_union_IoA_ge_0_5':sum(x>=.5 for x in covers)/n if n else 0.}
  for k,v in metrics.items():self.frame[k].append(v);self.groups[group][k].append(v)
  if camera:
   for k,v in metrics.items():self.cameras[camera][k].append(v)
  self.uid_hash.update((uid+'\n').encode());self.count+=1
  writer.writerow([self.name,uid,group,clip,frame,w,h,n,len(targets)-n,len(absent),sum(t.get('ignore_reason')=='non_target_role' for t in ignores),*[format(metrics[k],'.9g') for k in list(metrics)[1:]]])
 def result(self):
  target={k:describe(v) for k,v in self.target.items()};frame={k:describe(v) for k,v in self.frame.items()};heights=np.asarray(self.target['sqrt_area960']);crowd=np.asarray(self.target['union_bbox_IoA']);maxiou=np.asarray(self.target['max_bbox_IoU'])
  groups={g:{k:describe(v) for k,v in cols.items()} for g,cols in self.groups.items()};cameras={g:{k:describe(v) for k,v in cols.items()} for g,cols in self.cameras.items()}
  return {'frames':self.count,'targets':len(heights),'raw_positive_records':self.raw_targets,'invalid_positive_boxes':self.invalid,'synthetic_absent_person_records':self.absent,'known_non_target_records':self.nontarget,'group_count':len(groups),'group_unit':'synthetic_motion_seed' if self.name.startswith('S_') else 'real_match','native_shapes':dict(self.shapes),'ordered_UID_newline_sha256':self.uid_hash.hexdigest(),'frame_weighted':frame,'target_weighted':target,'fraction_frames_people_above10':sum(x>10 for x in self.frame['people'])/self.count,'bbox_area_size_at960_diagnostic_only':{'small_lt32_squared':float(np.mean(heights<32)),'medium_32_to96_squared':float(np.mean((heights>=32)&(heights<96))),'large_ge96_squared':float(np.mean(heights>=96))},'overlap_proxy':{'fraction_targets_any_bbox_overlap':float(np.mean(crowd>0)),'fraction_targets_union_IoA_ge_0_1':float(np.mean(crowd>=.1)),'fraction_targets_union_IoA_ge_0_3':float(np.mean(crowd>=.3)),'fraction_targets_union_IoA_ge_0_5':float(np.mean(crowd>=.5)),'fraction_targets_max_IoU_ge_0_1':float(np.mean(maxiou>=.1))},'real_GT_last3_columns_positive_records':dict(self.raw_tail),'per_group':groups,'group_equal_weighted_means':{k:describe([v[k]['mean'] for v in groups.values()]) for k in frame},'per_synthetic_camera_ID':cameras}

def check_plans():
 for name,h in PLANS.items():assert sha(ROOT/'docs/plan'/name)==h

def main():
 proc=psutil.Process();proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);proc.ionice(psutil.IOPRIO_VERYLOW);proc.cpu_affinity([proc.cpu_affinity()[-1]]);self_check();check_plans()
 assert not OUT.exists(),'one-shot analysis refuses overwrite';assert shutil.disk_usage(OUT.parent).free>100*1024**3+286156535472+32*1024**2
 evaluation_sha=sha(EVAL/'suite_request.json');assert evaluation_sha=='02cd3cb88efd7b2d98f40a4137195c4d776496eaab15e2740063401adf3b7fbd';OUT.mkdir()
 spec={'schema':'bbox_train_dev_annotation_domain_gap_v1','frozen_before_statistics':True,'at_utc':now(),'scope':{'S_train':24149,'S_dev':6030,'R_train':2100,'R_dev':750},'source_indices':{'synthetic':ref(SINDEX),'real':ref(RINDEX)},'cache_manifest_reference':ref(CACHE/'cache.json'),'allowed_real_splits':['train','val'],'forbidden_real_splits':['test'],'real_target_policy':'original player/goalkeeper+GTflag1+valid clipped box; known nontarget roles separately counted','synthetic_policy':'actual visible-mask-positive original boxes, not fixed10person records; absent separate','coordinates':'source clipped half-open XYXY;960 long-side LetterBox scale960/max(W,H),no random augmentation/no default training size filtering','size_metrics':['relative bbox height/area','width/height/sqrt-area at960'],'overlap_metrics':['max intersection/union(IoU)','max intersection/own area(IoA)','union of intersections with other target boxes/own area'],'overlap_thresholds_descriptive':[.1,.3,.5],'overlap_is_not_physical_occlusion':True,'source_specific_mask_bbox_fill_is_not_visibility_ratio':True,'pool_weighting':'frame counts weighted by frames; size/overlap by targets; per seed/match and camera separately; groups not IID frames','distances':'descriptive empirical KS and Wasserstein1; no hypothesis pvalues or domain-gap pass threshold','resource_limits':{'CPU_affinity':proc.cpu_affinity(),'priority':str(proc.nice()),'IO_priority':str(proc.ionice()),'G_annotation_read_MiBps':2,'D_cache_annotation_read_MiBps':8,'max_seconds':LIMIT,'RSS_limit_bytes':1200*1024**2},'forbidden_inputs':['RGB','mask PNG','NPY image arrays','model checkpoints','real test GT'],'evaluation_suite_reference':{'path':str(EVAL/'suite_request.json'),'sha256':evaluation_sha},'analysis_code_reference':ref(Path(__file__))}
 write(OUT/'analysis_spec.json',spec);pools={n:Pool(n) for n in spec['scope']};synthetic=small(SINDEX);realindex=small(RINDEX);cache=small(CACHE/'cache.json');assert cache['state']=='READY' and cache['count']==24149
 assert tuple(_source_stat(CACHE/'samples.jsonl'))==tuple(cache['samples_file']['stat']);seedmap={s['seed_id']:s for s in synthetic['seeds']};cache_uid_expected=[row['uid'] for row in cache['rows']];assert len(cache_uid_expected)==24149
 del cache
 framefields=['pool','uid','group','clip_or_camera','frame_index','width','height','valid_player_goalkeeper_boxes','invalid_positive_boxes','synthetic_absent_records','known_nontarget_records','mean_height960','mean_union_bbox_IoA','fraction_targets_union_IoA_ge_0_1','fraction_targets_union_IoA_ge_0_3','fraction_targets_union_IoA_ge_0_5']
 with (OUT/'frames.csv').open('x',encoding='utf-8-sig',newline='') as csvfile:
  writer=csv.writer(csvfile);writer.writerow(framefields)
  for i,raw in enumerate(lines(CACHE/'samples.jsonl')):
   row=json.loads(raw);m=row['metadata'];source=m['source_reference'];seed=Path(source['seed_dir']).name;assert row['row']==i and row['uid']==cache_uid_expected[i]==m['uid'] and m['domain']=='S' and source['split']=='train' and seedmap[seed]['split']=='train' and m['geometry']['letterbox_candidate_only'] is True
   h,w=m['geometry']['original_shape']['items'];pools['S_train'].add(row['uid'],seed,source['camera_id'],source['frame_index'],w,h,m['targets'],m['absent_targets'],m['ignore_targets'],writer,camera=source['camera_id'])
  print(json.dumps({'phase':'S_train_annotation_cache_complete','frames':pools['S_train'].count,'seconds':time.monotonic()-START}),flush=True)
  for seed in synthetic['seeds']:
   if seed['split']!='dev':continue
   seedroot=SINDEX.parent/seed['directory'];manifest=small(seedroot/'seed_manifest.json');assert sha(seedroot/'seed_manifest.json')==seed['seed_manifest_sha256'] and manifest['split']=='dev'
   for cid in seed['camera_ids']:
    camera=small(seedroot/cid/'camera.json');count=0
    for raw in lines(seedroot/cid/'annotations.jsonl'):
     row=json.loads(raw);count+=1;assert row['frame_index']==count
     sample=_synthetic_bbox_row(manifest,camera,row,camera_root=seedroot/cid,camera_id=cid,frame_index=count,split='dev');pools['S_dev'].add(sample['derived_sample_uid'],seed['seed_id'],cid,count,sample['width'],sample['height'],sample['targets'],sample['absent_targets'],sample['ignore_targets'],writer,camera=cid)
    assert count==seed['frames_per_camera']
   print(json.dumps({'phase':'S_dev_seed_complete','seed':seed['seed_id'],'frames_total':pools['S_dev'].count,'seconds':time.monotonic()-START}),flush=True)
  reader=RealBBoxReader(RINDEX,max_cached_clips=20,max_cached_rows=600000);allowed_files=[]
  for seq in realindex['sequences']:
   if seq['split'] not in ('train','val'):continue
   pool=pools['R_train' if seq['split']=='train' else 'R_dev']
   for frame in seq['detection_frames']:
    uid=seq['detection_sample_uid_pattern'].format(frame_uid=seq['frame_uid_pattern'].format(frame_number=frame));sample=reader.read(uid,split=seq['split']);pool.add(uid,seq['match_uid'],seq['clip_uid'],frame,seq['width'],seq['height'],sample['targets'],[],sample['ignore_targets'],writer)
   for key in ('gt','gameinfo','seqinfo'):
    path=RINDEX.parent/seq['sequence_path']/seq['files'][key];prove(path);allowed_files.append(str(path))
   assert sha(RINDEX.parent/seq['sequence_path']/seq['files']['gt'])==seq['gt_sha256']
  assert all(reader._clips[x]['split'] in ('train','val') for x in reader._bindings)
 for k,p in pools.items():assert p.count==spec['scope'][k],(k,p.count)
 assert len(pools['S_train'].groups)==16 and len(pools['S_dev'].groups)==4 and len(pools['R_train'].groups)==3 and len(pools['R_dev'].groups)==1
 results={k:p.result() for k,p in pools.items()};comparisons={}
 for source in ('S_train','S_dev'):
  for target in ('R_train','R_dev'):
   comparisons[source+'__'+target]={'people':distances(pools[source].frame['people'],pools[target].frame['people']),**{m:distances(pools[source].target[m],pools[target].target[m]) for m in ('height960','sqrt_area960','union_bbox_IoA')}}
 summary={'schema':'bbox_train_dev_domain_gap_results_v1','state':'COMPLETE','analysis_spec_reference':ref(OUT/'analysis_spec.json'),'pools':results,'descriptive_distribution_distances':comparisons,'limitations':['This is five-a-side source to eleven-a-side target annotation-domain difference, not isolated causal effect of sport format.','BBox overlap is potential occlusion/crowding; no comparable physical person-occlusion fraction is available.','Synthetic visible-mask tight boxes and real official boxes may differ in occluded extent; no amodal equivalence assumed.','Synthetic visibility means existence of mask pixels; mask fill/bbox is body shape fill, not unoccluded body fraction.','Actual detection pools before augmentation/default size filtering; no training or evaluation changes.','Frames and objects correlated within match/seed/trajectory/cameras; no IID pvalues or CI.','Real dev is one repeatedly observed match, not test; no game8/game9 GT/media/predictions read.']}
 write(OUT/'summary.json',summary)
 with (OUT/'groups.csv').open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['pool','group','frames','people_mean','height960_frame_mean','union_IoA_frame_mean'])
  for k,result in results.items():
   for group,v in result['per_group'].items():w.writerow([k,group,v['people']['n'],v['people']['mean'],v['mean_height960']['mean'],v['mean_union_bbox_IoA']['mean']])
 edges={'people':np.arange(-.5,40.5,1.),'height960':np.array([0,4,8,16,24,32,48,64,96,128,192,256,384,540,960]),'union_bbox_IoA':np.linspace(0,1,21)}
 with (OUT/'distribution_bins.csv').open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['pool','metric','left_inclusive','right_exclusive_except_last','count','denominator'])
  for k,p in pools.items():
   for metric,bins in edges.items():
    a=np.asarray(p.frame[metric] if metric=='people' else p.target[metric]);hist,_=np.histogram(a,bins=bins);assert sum(hist)==len(a)
    for i,n in enumerate(hist):w.writerow([k,metric,float(bins[i]),float(bins[i+1]),int(n),len(a)])
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(13.2,4.1),layout='constrained');colors={'S_train':'#da6d22','S_dev':'#b28b67','R_train':'#2465af','R_dev':'#448f8b'}
 for k,p in pools.items():
  a=np.asarray(p.frame['people']);xs=np.arange(0,max(int(max(v.frame['people'])) for v in pools.values())+1);axes[0].plot(xs,[np.mean(a==x) for x in xs],color=colors[k],label=f'{k} ({p.count:,} frames)',linestyle='--' if k.endswith('dev') else '-')
  a=np.sort(np.asarray(p.target['height960']));grid=np.linspace(0,1,501);axes[1].plot(np.quantile(a,grid),grid,color=colors[k],label=k,linestyle='--' if k.endswith('dev') else '-')
  a=np.asarray(p.target['union_bbox_IoA']);axes[2].plot([0.1,0.3,0.5],[np.mean(a>=t) for t in [0.1,0.3,0.5]],'o-',color=colors[k],label=k,linestyle='--' if k.endswith('dev') else '-')
 axes[0].set(xlabel='Valid player + goalkeeper boxes / frame',ylabel='Fraction of frames',title='Annotated target count');axes[1].set(xlabel='BBox height at 960 long side (pixels)',ylabel='Empirical cumulative fraction',title='Target size before augmentation');axes[2].set(xlabel='Union overlap / own bbox area threshold',ylabel='Fraction of targets at or above threshold',title='Potential occlusion / crowding proxy',xticks=[.1,.3,.5]);axes[0].legend(fontsize=7)
 for ax in axes:ax.grid(alpha=.2)
 fig.suptitle('Existing synthetic and SoccerNet train/dev pools; no test data or model inference',fontsize=11);fig.savefig(OUT/'domain_gap.png',dpi=180);fig.savefig(OUT/'domain_gap.svg');plt.close(fig)
 for path,identity in SOURCE_PROOFS.items():assert tuple(_source_stat(path))==identity
 assert sha(EVAL/'suite_request.json')==evaluation_sha;check_plans();assert 'torch' not in sys.modules
 write(OUT/'source_references.json',{'source_stats_before_after_equal':True,'files':[{'path':p,'stat':list(s)} for p,s in SOURCE_PROOFS.items()],'real_opened_splits':['train','val'],'real_opened_GT_files':allowed_files,'test_GT_opened':False,'RGB_mask_NPY_PT_opened':False})
 shutil.copyfile(__file__,OUT/'analysis_code.py')
 write(OUT/'execution_receipt.json',{'state':'COMPLETE','at_utc':now(),'elapsed_seconds':time.monotonic()-START,'CPU_seconds':sum(proc.cpu_times()[:2]),'CPU_affinity':proc.cpu_affinity(),'priority':str(proc.nice()),'IO_priority':str(proc.ionice()),'peak_working_set_bytes':PEAK,'bytes_read_by_drive':dict(TOTAL_READ),'torch_imported':False,'GPU_initialized':False,'real_test_GT_or_media_read':False,'evaluation_suite_request_unchanged':True,'inputs_written':False,'geometry_self_checks':5,'frames_by_pool':{k:p.count for k,p in pools.items()},'files_hashed':'small source indices/manifests/GT and analysis outputs only; no RGB/mask/NPY/PT/whole original annotation hash'})
 files=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.iterdir() if p.is_file()];assert sum(v['bytes'] for v in files)<32*1024**2
 write(OUT/'CONTROL_MANIFEST.json',{'schema':'bbox_domain_gap_analysis_manifest_v1','state':'SEALED','scope':'analysis_metadata_and_results_only_no_inputs','files':files,'file_count':len(files),'total_bytes':sum(v['bytes'] for v in files),'at_utc':now()})
 print(json.dumps({'state':'COMPLETE','output':str(OUT),'elapsed_seconds':time.monotonic()-START,'summary_reference':ref(OUT/'summary.json'),'compact':{k:{'frames':v['frames'],'targets':v['targets'],'people_mean':v['frame_weighted']['people']['mean'],'people_median':v['frame_weighted']['people']['q50'],'height960_median':v['target_weighted']['height960']['q50'],'height960_q10':v['target_weighted']['height960']['q10'],'height960_q90':v['target_weighted']['height960']['q90'],'overlap_ge0_3':v['overlap_proxy']['fraction_targets_union_IoA_ge_0_3']} for k,v in results.items()}},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
