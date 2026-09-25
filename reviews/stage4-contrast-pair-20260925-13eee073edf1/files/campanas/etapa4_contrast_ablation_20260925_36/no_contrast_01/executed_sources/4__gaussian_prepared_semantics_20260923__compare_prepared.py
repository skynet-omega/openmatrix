"""Exact semantic comparison of serialized prepared states, CPU and read-only.

No organism modules, pickle, GPU imports or implicit metadata exclusions.
Large NPY payloads are compared in chunks. JSON trees load one at a time.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib,re,resource,sys,time,zipfile
import numpy as np

REQUIRED=('session','prosthesis','published','effective_operator')
PROFILE_KEYS=('engine','source_identity','plan_sha256','checkpoint_manifest_sha256','preparation_ms')
ENC=json.JSONEncoder(sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
CHUNK=1024*1024

class InvalidState(ValueError):pass
def need(ok,message):
    if not ok:raise InvalidState(message)
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
    return h.hexdigest()
def reject_constant(x):raise InvalidState('Nonfinite JSON value: '+x)
def unique_pairs(pairs):
    d={}
    for k,v in pairs:
        need(k not in d,'Duplicate JSON key: '+k);d[k]=v
    return d
def load_json(p):
    with pathlib.Path(p).open() as f:return json.load(f,object_pairs_hook=unique_pairs,parse_constant=reject_constant)
def canonical(value):
    h=hashlib.sha256()
    for s in ENC.iterencode(value):h.update(s.encode())
    return h.hexdigest()

def metadata(p):
    root=load_json(p);bindings={};members=set()
    def walk(v,route):
        if isinstance(v,dict):
            if '__array__' in v:
                need(set(v)=={'__array__'},'Malformed array marker at '+str(route))
                name=v['__array__'];need(isinstance(name,str) and re.fullmatch(r'array_[0-9]+',name),'Invalid array reference at '+str(route))
                need(name not in members,'Aliased array reference '+name);members.add(name)
                key=json.dumps(route,separators=(',',':'));bindings[key]=name+'.npy'
                v['__array__']={'semantic_path':route}
            else:
                for k,ch in v.items():walk(ch,route+[k])
        elif isinstance(v,list):
            for i,ch in enumerate(v):walk(ch,route+[i])
    walk(root,[])
    return {'semantic_sha256':canonical(root),'arrays':bindings,
            'top_level_sha256':{k:canonical(v)for k,v in root.items()}if isinstance(root,dict)else {'$':canonical(root)}}

def header(f):
    v=np.lib.format.read_magic(f)
    if v==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f)
    elif v==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f)
    else:raise InvalidState('Unsupported NPY version '+str(v))
    need(not dtype.hasobject and dtype.fields is None and dtype.subdtype is None,'Object/structured/subarray dtype unsupported')
    need(dtype.itemsize>0,'Zero-size dtype unsupported')
    need(all(isinstance(x,int) and x>=0 for x in shape),'Invalid NPY dimensions')
    return {'shape':list(shape),'fortran_order':order,'dtype':dtype.str,'npy_version':list(v)},dtype

def compare_arrays(pa,pb,amap,bmap):
    need(set(amap)==set(bmap),'Semantic array paths differ')
    rows=[]
    with zipfile.ZipFile(pa)as za,zipfile.ZipFile(pb)as zb:
        for z,mapping,side in [(za,amap,'left'),(zb,bmap,'right')]:
            names=z.namelist();need(len(names)==len(set(names)),'Duplicate ZIP member '+side)
            need(set(names)==set(mapping.values()),'Orphan/missing NPZ member '+side)
        for key in sorted(amap):
            with za.open(amap[key])as fa,zb.open(bmap[key])as fb:
                ah,ad=header(fa);bh,bd=header(fb)
                row={'path':json.loads(key),'member_left':amap[key],'member_right':bmap[key],'structure_left':ah,'structure_right':bh,'structure_equal':ah==bh}
                if ah!=bh:row.update(exact=False,reason='array structure differs');rows.append(row);continue
                n=math.prod(ah['shape']);total=0;da=hashlib.sha256();db=hashlib.sha256();equal=True;nonfinite=[0,0]
                while total<n:
                    count=min(n-total,max(1,CHUNK//ad.itemsize));size=count*ad.itemsize
                    ba=fa.read(size);bb=fb.read(size);need(len(ba)==len(bb)==size,'Truncated array payload')
                    da.update(ba);db.update(bb);equal=equal and ba==bb
                    if ad.kind in 'fc':
                        nonfinite[0]+=int(np.count_nonzero(~np.isfinite(np.frombuffer(ba,dtype=ad))))
                        nonfinite[1]+=int(np.count_nonzero(~np.isfinite(np.frombuffer(bb,dtype=bd))))
                    total+=count
                need(not fa.read(1) and not fb.read(1),'Trailing NPY data')
                row.update(elements=n,payload_sha256_left=da.hexdigest(),payload_sha256_right=db.hexdigest(),bitwise_equal=equal,nonfinite_left=nonfinite[0],nonfinite_right=nonfinite[1],exact=equal and sum(nonfinite)==0)
                if sum(nonfinite):row['reason']='nonfinite scientific array'
                elif not equal:row['reason']='payload bits differ'
                rows.append(row)
    return rows

def validate_manifest(folder):
    p=folder/'MANIFEST.json';need(p.is_file()and not p.is_symlink(),'Missing/symlink MANIFEST.json')
    m=load_json(p);need(isinstance(m,dict)and isinstance(m.get('files'),dict),'Invalid manifest structure')
    names={p.name for p in folder.iterdir()if p.name!='MANIFEST.json'}
    need(names==set(m['files']),'Manifest inventory mismatch')
    for name,rec in m['files'].items():
        need(pathlib.Path(name).name==name and name not in {'.','..'},'Non-local manifest path')
        p=folder/name;need(p.is_file()and not p.is_symlink(),'Missing/nonfile/symlink manifest member '+name)
        need(isinstance(rec,dict)and set(rec)=={'bytes','sha256'},'Unexpected manifest entry fields '+name)
        need(type(rec['bytes'])is int and p.stat().st_size==rec['bytes'],'Manifest size mismatch '+name)
        need(isinstance(rec['sha256'],str)and sha(p)==rec['sha256'],'Manifest hash mismatch '+name)
    return {'manifest_sha256':sha(folder/'MANIFEST.json'),'files_verified':len(names),'metadata_semantic_sha256':canonical({k:v for k,v in m.items()if k!='files'}),'inventory':sorted(names)}

def profile(folder):
    p=folder.parent/'RUN_CONTRACT.json';need(p.is_file(),'Missing profile RUN_CONTRACT.json')
    d=load_json(p);need(isinstance(d,dict),'Invalid profile contract')
    need(all(k in d for k in PROFILE_KEYS),'Missing profile fields: '+','.join(k for k in PROFILE_KEYS if k not in d))
    need(d['engine']in ('causal_cuda','reference_cuda'),'Unknown engine profile')
    need(type(d['preparation_ms'])is int and d['preparation_ms']>=0,'Invalid preparation duration')
    for k in ('plan_sha256','checkpoint_manifest_sha256'):need(isinstance(d[k],str)and re.fullmatch('[0-9a-f]{64}',d[k]),'Invalid profile hash '+k)
    need(isinstance(d['source_identity'],dict)and 'source_lock_sha256'in d['source_identity'],'Missing source identity binding')
    return {'contract_path':str(p),'contract_sha256':sha(p),'identity':{k:d[k]for k in PROFILE_KEYS},'identity_sha256':canonical({k:d[k]for k in PROFILE_KEYS})}

def compare_prepared(left,right,expected_engine):
    left=pathlib.Path(left).resolve();right=pathlib.Path(right).resolve()
    need(expected_engine in ('causal_cuda','reference_cuda'),'Explicit supported engine required')
    result={'schema':'exact_prepared_semantics_v1','left':str(left),'right':str(right),'expected_engine':expected_engine,'required_sections':[*REQUIRED,'boundary'],'sections':{},'missing':[],'invalid':[],'scientific_state_exact':False,'complete_required_state_present':False,'full_organism_resume_validated':False}
    for side,folder in [('left',left),('right',right)]:
        expected=[s+x for s in REQUIRED for x in ('.json','.npz')]+['boundary.json','MANIFEST.json']
        result['missing'] += [{'side':side,'file':n}for n in expected if not(folder/n).is_file()]
        try:result[side+'_profile']=profile(folder)
        except (InvalidState,OSError,ValueError)as e:result['invalid'].append({'side':side,'scope':'profile','error':str(e)})
        try:result[side+'_integrity']=validate_manifest(folder)
        except (InvalidState,OSError,ValueError)as e:result['invalid'].append({'side':side,'scope':'manifest','error':str(e)})
    result['complete_required_state_present']=not result['missing']
    if result['invalid']:
        result['status']='INVALID_OR_INCOMPLETE';return result
    lp,rp=result['left_profile'],result['right_profile']
    if lp['identity_sha256']!=rp['identity_sha256']or lp['identity']['engine']!=expected_engine:
        result['status']='PROFILE_MISMATCH';return result
    li,ri=result['left_integrity'],result['right_integrity']
    result['manifest_metadata_exact']=li['metadata_semantic_sha256']==ri['metadata_semantic_sha256']
    names=set(li['inventory'])|set(ri['inventory']);equal=result['manifest_metadata_exact']
    # Every file present is consumed; no fixed whitelist silently drops extras.
    stems=sorted({pathlib.Path(n).stem for n in names if n.endswith('.json')})
    covered=set()
    for stem in stems:
        j=stem+'.json';z=stem+'.npz';covered.add(j)
        if z in names:covered.add(z)
        missing=[{'side':side,'file':n}for side,folder in [('left',left),('right',right)]for n in ([j,z]if z in names else[j])if not(folder/n).is_file()]
        if missing:result['sections'][stem]={'status':'MISSING_NOT_COMPARABLE','missing':missing};equal=False;continue
        try:
            am,bm=metadata(left/j),metadata(right/j)
            if z not in names:need(not am['arrays']and not bm['arrays'],'Array marker without NPZ')
            rows=compare_arrays(left/z,right/z,am['arrays'],bm['arrays'])if z in names else[]
            exact=am['semantic_sha256']==bm['semantic_sha256']and all(x['exact']for x in rows)
            result['sections'][stem]={'status':'EXACT'if exact else'DIFFERENT','metadata_exact':am['semantic_sha256']==bm['semantic_sha256'],'metadata_different_top_keys':sorted(k for k in set(am['top_level_sha256'])|set(bm['top_level_sha256'])if am['top_level_sha256'].get(k)!=bm['top_level_sha256'].get(k)),'left_metadata_sha256':am['semantic_sha256'],'right_metadata_sha256':bm['semantic_sha256'],'array_count':len(rows),'renumbered_members':sum(x['member_left']!=x['member_right']for x in rows),'arrays':rows}
            equal=equal and exact
        except (InvalidState,OSError,ValueError,zipfile.BadZipFile)as e:
            result['sections'][stem]={'status':'INVALID','error':str(e)};result['invalid'].append({'scope':stem,'error':str(e)});equal=False
    for n in sorted(names-covered):
        # NPZ without metadata cannot bind arrays to scientific paths.
        if n.endswith('.npz'):
            result['sections'][n]={'status':'INVALID','error':'NPZ without JSON semantic bindings'};result['invalid'].append({'scope':n,'error':'unbound NPZ'});equal=False
        else:
            exact=(left/n).is_file()and(right/n).is_file()and sha(left/n)==sha(right/n)
            result['sections'][n]={'status':'EXACT'if exact else'DIFFERENT','raw_bytes_exact':exact};equal=equal and exact
    # Sections absent on both sides must remain visible, not disappear in set union.
    for s in (*REQUIRED,'boundary'):
        if s not in result['sections']:result['sections'][s]={'status':'MISSING_NOT_COMPARABLE'}
    result['available_sections_exact']=equal and not result['invalid']
    result['scientific_state_exact']=result['available_sections_exact']and not result['missing']
    result['status']='INVALID'if result['invalid']else'INCOMPLETE_REQUIRED_STATE'if result['missing']else'EXACT'if result['scientific_state_exact']else'DIFFERENT'
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('left',type=pathlib.Path);p.add_argument('right',type=pathlib.Path);p.add_argument('--engine',required=True,choices=['causal_cuda','reference_cuda']);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU,(90,90));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    need(not a.out.exists(),'Output already exists')
    out=a.out.resolve();forbidden=[a.left.resolve(),a.right.resolve(),pathlib.Path('/home/daroch/AXIOMA_FLYWIRE')]
    need(not any(root==out or root in out.parents for root in forbidden),'Output must not modify input/historical tree')
    t=time.process_time();w=time.monotonic();r=compare_prepared(a.left,a.right,a.engine)
    r['runtime']={'cpu_s':time.process_time()-t,'wall_s':time.monotonic()-w,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'gpu_calls':0,'organism_loads':0};r['comparator_sha256']=sha(pathlib.Path(__file__))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x')as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':r['status'],'scientific_state_exact':r['scientific_state_exact'],'missing':r['missing'],'invalid':r['invalid'],'output':str(a.out),'runtime':r['runtime']},indent=2))
    return 0 if r['scientific_state_exact']else 2
if __name__=='__main__':raise SystemExit(main())
