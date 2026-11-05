"""仅CPU准备端的缓存路径表示兼容；不修改准入/42训练源码或数值loop。"""
import ast,hashlib,inspect
from pathlib import Path

SOURCE_SHA='2923b3fa13b27e81b0d6507125dbf31b7279895baf18705500b7f223f15268bf'
CACHE_PATH=Path('D:/projects/FutsalMOT_SyntheticDatasetResearch/cache/synthetic_bbox_960_v2')
CACHE_SHA='fce4e02611f198ba1e72fe1a5ae47ebf047f83a29b2738ef88a0d1429001b324'

def prepare_group(module,group,*,config,admission,code_source):
    from futsalmot import formal
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==SOURCE_SHA
    # 原两次读取仍加载未改admission；仅规范同一已验实际路径的两处表示比较。
    for raw in (admission['input_cache'],admission['source_bindings']['input_cache'],config['input_cache']):
        normalized=formal.cache_specification(raw)
        assert Path(normalized['path'])==CACHE_PATH and normalized['manifest_sha256']==CACHE_SHA
        assert not CACHE_PATH.is_symlink() and not CACHE_PATH.is_junction()
    source=inspect.getsource(module.prepare_group)
    substitutions=[("actual.get('input_cache')==config['input_cache']","formal.cache_specification(actual.get('input_cache'))==config['input_cache']"),("runner.canonical(actual['source_bindings'].get(key))==runner.canonical(config[key])","runner.canonical(formal.cache_specification(actual['source_bindings'].get(key)) if key=='input_cache' else actual['source_bindings'].get(key))==runner.canonical(config[key])")]
    patched=source
    for before,after in substitutions:
        assert patched.count(before)==1
        patched=patched.replace(before,after)
    reversed_source=patched
    for before,after in reversed(substitutions):reversed_source=reversed_source.replace(after,before)
    assert ast.dump(ast.parse(reversed_source))==ast.dump(ast.parse(source))
    namespace=dict(module.__dict__)
    exec(compile(patched,str(Path(__file__).absolute())+':prepare_only','exec'),namespace)
    return namespace['prepare_group'](group,config=config,admission=admission,code_source=code_source)
