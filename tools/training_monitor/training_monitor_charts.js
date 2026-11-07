/* 只读曲线显示运算：不改变训练日志、模型EMA或正式评价。 */
(function(root,factory){if(typeof module==='object'&&module.exports)module.exports=factory();else root.TrainingCharts=factory();})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const finite=v=>typeof v==='number'&&Number.isFinite(v);
function smooth(points,key,weight){if(!finite(weight)||weight<0||weight>=1)throw new RangeError('Smoothing must be in [0,1)');let m=0,bias=1,last=null,segment=0;return points.map(p=>{const x=p.update,y=p[key];if(!finite(x))return{update:null,raw:null,smoothed:null,segment:++segment};if(last!==null&&(x<=last||x-last>1)){m=0;bias=1;segment++;}last=x;if(!finite(y)){m=0;bias=1;return{update:x,raw:null,smoothed:null,segment:segment++};}m=weight*m+(1-weight)*y;bias*=weight;return{update:x,raw:y,smoothed:m/(1-bias),segment};});}
function clip(points,min,max){if(!finite(min)||!finite(max)||min>max)throw new RangeError('Invalid plot range');return points.filter(p=>finite(p.update)&&p.update>=min&&p.update<=max);}
function downsample(points,max=1800){if(!Number.isInteger(max)||max<7)throw new RangeError('Too few render points');if(points.length<=max)return points.slice();const buckets=Math.max(1,Math.floor((max-2)/5)),size=(points.length-2)/buckets,indices=new Set([0,points.length-1]);for(let b=0;b<buckets;b++){const start=1+Math.floor(b*size),end=1+Math.floor((b+1)*size);for(const key of ['raw','smoothed']){let low=-1,high=-1;for(let i=start;i<end;i++)if(finite(points[i][key])){if(low<0||points[i][key]<points[low][key])low=i;if(high<0||points[i][key]>points[high][key])high=i;}if(low>=0)indices.add(low);if(high>=0)indices.add(high);}for(let i=start;i<end;i++)if(!finite(points[i].raw)){indices.add(i);break;}}return[...indices].sort((a,b)=>a-b).map(i=>points[i]);}
function nearest(points,x){if(!points.length||!finite(x))return null;let a=0,b=points.length-1;while(a<b){const m=Math.floor((a+b)/2);if(points[m].update<x)a=m+1;else b=m;}return a>0&&Math.abs(points[a-1].update-x)<=Math.abs(points[a].update-x)?points[a-1]:points[a];}
function bounds(series,log=false){let min=Infinity,max=-Infinity;for(const s of series)for(const p of s)for(const k of ['raw','smoothed'])if(finite(p[k])&&(!log||p[k]>0)){min=Math.min(min,p[k]);max=Math.max(max,p[k]);}if(!finite(min))return null;if(log){min=Math.log10(min);max=Math.log10(max);}const pad=min===max?.1:(max-min)*.08;return{min:log?min-pad:Math.max(0,min-pad),max:max+pad,log};}
function cell(v){if(v===null||v===undefined||typeof v==='number'&&!finite(v))return'';let s=String(v);if(typeof v==='string'&&/^[=+\-@]/.test(s))s="'"+s;return /[",\r\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}
function csv(datasets,key,weight,min,max){const header=['run','metric','update','raw_value','display_smoothed_value','display_smoothing','loss','loss_box','loss_cls','loss_dfl','gradient_norm','learning_rate','update_seconds','R_exposures','S_exposures'];const rows=[header.join(',')];for(const d of datasets){const sm=smooth(d.points,key,weight),lookup=new Map(sm.filter(p=>finite(p.update)).map(p=>[p.update,p]));for(const p of d.points){if(!finite(p.update)||p.update<min||p.update>max)continue;const q=lookup.get(p.update);rows.push([d.name,key,p.update,p[key],q?.smoothed,weight,p.loss,p.loss_box,p.loss_cls,p.loss_dfl,p.gradient_norm,p.learning_rate,p.wall_seconds,p.R_exposures,p.S_exposures].map(cell).join(','));}}return'\uFEFF'+rows.join('\r\n')+'\r\n';}
function validationPending(run){const p=run?.progress||{},cfg=run?.protocol||{},u=p.update,interval=cfg.evaluation_every||1000;if(run?.state!=='RUNNING'||!Number.isInteger(u))return false;return(u===0||u===p.total_updates||u%interval===0)&&!(run.curves?.validation||[]).some(v=>v.update===u);}
// 筛选仅使用观察台已登记字段，不猜训练预算或改动记录。
const filterKeys=['purpose','task','model','panel','strategy','seed','labels','budget','state'];
function runFields(run){
 const label=String(run?.label||'').split('｜'),known=v=>v===undefined||v===null||v===''?'未登记':String(v),b=run?.budget||{};
 const n=b.unique_real_images,full=b.full_real_images;
 const labels=Number.isInteger(n)&&n>=0?(Number.isInteger(full)&&full>0&&n<=full?Number((n/full*100).toFixed(2))+'% · '+n+'/'+full+'图':n+'图'):'未登记';
 return{purpose:known(label[0]),task:known(label[1]),model:known(label[2]),panel:known(label[3]),strategy:known(label[4]),seed:Number.isInteger(run?.seed)?'seed'+run.seed:known(label[5]),labels,budget:Number.isInteger(b.total_updates)&&b.total_updates>=0?b.total_updates.toLocaleString('en-US')+'步':'未登记',state:known(run?.state)};
}
function cleanRunFilters(value){const result={};for(const key of [...filterKeys,'query'])result[key]=typeof value?.[key]==='string'?value[key].slice(0,200):'';return result;}
function filterRuns(runs,filters={},excludeField=null){
 const f=cleanRunFilters(filters),terms=f.query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
 return runs.filter(run=>{const fields=runFields(run);if(filterKeys.some(k=>k!==excludeField&&f[k]&&fields[k]!==f[k]))return false;const text=[run.label,run.menu_label,run.id,run.group_id,run.control_group_path,run.output_group_path,...Object.values(fields)].join(' ').toLocaleLowerCase();return terms.every(term=>text.includes(term));});
}
function runFacets(runs,filters,key){if(!filterKeys.includes(key))throw new RangeError('Unknown filter field');const counts=new Map();for(const run of filterRuns(runs,filters,key)){const value=runFields(run)[key];counts.set(value,(counts.get(value)||0)+1);}const chosen=cleanRunFilters(filters)[key];if(chosen&&!counts.has(chosen))counts.set(chosen,0);return [...counts].sort((a,b)=>a[0].localeCompare(b[0],'zh-CN',{numeric:true})).map(([value,count])=>({value,count}));}
function runMenuRows(runs,filters,selectedId='',excludeId=''){
 const visible=filterRuns(runs,filters).filter(r=>r.id!==excludeId),selected=runs.find(r=>r.id===selectedId&&r.id!==excludeId);
 return{matched:visible.length,rows:visible.map(run=>({run,outside:false})).concat(selected&&!visible.some(r=>r.id===selectedId)?[{run:selected,outside:true}]:[])};
}
return{finite,smooth,clip,downsample,nearest,bounds,csv,validationPending,filterKeys,runFields,cleanRunFilters,filterRuns,runFacets,runMenuRows};
});