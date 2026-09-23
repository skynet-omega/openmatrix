"""Explicit, local-only review snapshot. Never invokes publish or an organism."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, resource, shutil, stat, time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
C = ROOT / 'campanas'
NAMES = {15: 'etapa3_pn629_intervention_20260923_15',
         16: 'etapa3_funcional_20260923_16', 17: 'etapa3_funcional_repair_20260923_17',
         19: 'etapa3_continuation_repair_20260923_19', 20: 'etapa3_postclose_20260923_20'}
DIR = {k: C / v for k, v in NAMES.items()}
ARMS = ('sham', 'odor_left', 'odor_right', 'uniform')

def need(ok, message):
    if not ok: raise ValueError(message)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def js(path): return json.loads(Path(path).read_text())
def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def no_link(path):
    path=Path(path)
    need(path.is_absolute(), 'Absolute source required')
    need(not any(p.is_symlink() for p in (path,*path.parents)), 'Symlink rejected: '+str(path))
    need(path.is_file() and stat.S_ISREG(path.stat().st_mode), 'Nonregular source')
    return path

def inventory():
    files={}; omitted=[]; expected={}
    for lock in (DIR[19]/'RUN_LOCK.json', DIR[20]/'SOURCE_LOCK.json'):
        expected.update(js(lock))
    post=js(DIR[20]/'POSTCLOSE_VERIFIED_01.json')
    for path, row in post['read_input_hashes'].items():
        if path in expected: need(expected[path]==row['sha256'], 'Contradictory frozen hashes')
        expected[path]=row['sha256']
    def add(p, role, destination=None):
        p=no_link(p)
        if p.stat().st_size==0:
            omitted.append({'source':str(p),'bytes':0,'reason':'Empty historical log, not substituted with a placeholder'}); return
        if p.stat().st_size>100000000:
            omitted.append({'source':str(p),'bytes':p.stat().st_size,'reason':'Full checkpoint exceeds compact per-file ceiling',
                            'previously_verified_sha256':expected.get(str(p))}); return
        if p.name=='effective_operator.npz' and destination is None:
            omitted.append({'source':str(p),'bytes':p.stat().st_size,'reason':'Deduplicated common operator; identity linked by postclose audit',
                            'previously_verified_sha256':expected.get(str(p))}); return
        destination=destination or str(p.relative_to(ROOT.parent))
        if destination in files:
            need(files[destination]['source']==str(p), 'Destination collision'); return
        files[destination]={'source':str(p),'destination':destination,'role':role,
                            'bytes':p.stat().st_size,'expected_sha256':expected.get(str(p))}
    def shallow(folder,role):
        if not folder.is_dir():return
        need(not folder.is_symlink(),'Linked directory rejected')
        for p in sorted(folder.iterdir()):
            if p.is_symlink():
                omitted.append({'source':str(p),'reason':'Convenience symlink excluded; physical origin selected separately'});continue
            if p.is_file() and p.suffix in {'.py','.cu','.cuh','.cpp','.hpp','.md','.json','.jsonl','.npz','.log','.txt','.csv'}:
                # Publication messages/indices are not scientific inputs.
                if any(x in p.name for x in ('PUBLICATION','REMOTE_VERIFY','CHATGPT_DELIVERY','CHATGPT_FOLLOWUP_DELIVERY','_SPEC.')):continue
                add(p,role)
    for k in (16,17,19,20):shallow(DIR[k], 'Campaign source, contract or receipt')
    for name in ('PLAN.json','CLOSE.json','RESULTS.md','README.md','REVIEW_PLAN.json','PREPARATION_COMPARE.json',
                 'CAPTURE_NEUTRALITY.json','CHATGPT_CODE_REPRO.json','chatgpt_verificador_original.py',
                 'ORIENTATION_SCOPE_ANTECEDENT.md'):
        add(DIR[15]/name,'Native comparator lineage')
    # Source closure explicitly named in campaign contracts, not whole repositories.
    for k in (16,17,19):
        for p in sorted(DIR[k].glob('*SOURCES.json')):
            for path,digest in js(p).items():
                if not path.startswith('/') or not isinstance(digest,str) or len(digest)!=64:continue
                source=Path(path)
                need(source.is_relative_to(ROOT) or source.is_relative_to(ROOT.parent/'AXIOMA_FLYWIRE'),'Unrelated frozen dependency')
                expected[str(source)]=digest;add(source,'Declared frozen source dependency')
    runs={}
    for arm in ARMS:
        runs['native_'+arm]=DIR[15]/('full_'+arm+'_01')
        runs['reference_'+arm]=(DIR[17] if arm=='uniform' else DIR[16])/('reference_'+arm+'_01')
    for arm in ('odor_left','odor_right'):runs['withdrawal_'+arm]=DIR[17]/('withdrawal_'+arm+'_01')
    runs['continuation']=DIR[19]/'continuation_right_01'
    runs['incomplete_uniform']=DIR[16]/'reference_uniform_01'
    runs['incomplete_continuation']=DIR[17]/'continuation_right_01'
    for name,folder in runs.items():
        shallow(folder,name)
        for sub in ('flow','executed_sources','experiment_sources','preparation_inputs','prepared_state','state_100ms','final_state'):
            shallow(folder/sub,name+'/'+sub)
        # The failed loader left a partial checkpoint; keep its manifest/boundary,
        # enumerate its omitted state rather than pretending it was a valid snapshot.
        for sub in sorted(folder.glob('*.partial-*')):shallow(sub,name+'/incomplete snapshot')
    shallow(DIR[16]/'body_support_01','Mechanical replay receipt')
    for arm in ARMS:shallow(DIR[16]/'body_support_01'/arm,'Mechanical replay '+arm)
    for folder in (DIR[16]/'comparison_3_01', DIR[17]/'comparison_4_01'):
        shallow(folder,'Comparison receipt')
    for sub,names in {
        'etapa3_dnb05_native_20260923_12':('PLAN.json','GATE.json','README.md','REFERENCE_20_COMPARE.json','COMPACT_NUMERIC.npz','verify_compact.py'),
        'etapa3_kc_boundary_20260923_13':('PLAN.json','BOUNDARY.json','README.md','KC_LOCAL_MAPPING.json','KC_MOTOR_REACHABILITY.json')
    }.items():
        for name in names:add(C/sub/name,'Preserved KC limitation and predecessor evidence')
    op=DIR[16]/'reference_sham_01/prepared_state'
    add(op/'effective_operator.npz','Common immutable operator','shared_operator/effective_operator.npz')
    add(op/'effective_operator.json','Common immutable operator descriptor','shared_operator/effective_operator.json')
    common_digest=expected[str(op/'effective_operator.npz')]
    operator_links={p:r for p,r in post['read_input_hashes'].items() if p.endswith('/effective_operator.npz')}
    need(len(operator_links)==8 and all(v['sha256']==common_digest for v in operator_links.values()),'Shared operator assumption failed')
    for name in ('publish.py','verify_remote.py','test_publish.py','test_verify_remote.py','README.md'):
        add(ROOT/'instrumentos/openmatrix'/name,'OpenMatrix workflow')
    for name in ('PLAN.json','prepare.py','verify_compact.py','README.md','test_package.py','REVIEW_REQUEST.md','TEST_NORMAL.json','TEST_OPTIMIZED.json','PACKAGING_HISTORY.json'):
        add(HERE/name,'Compact package tooling',name)
    return files,omitted,{k:str(v.relative_to(ROOT.parent)) for k,v in runs.items()},operator_links

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build', action='store_true');parser.add_argument('--label',required=True)
    args=parser.parse_args();startcpu=time.process_time();start=time.monotonic()
    limits=js(HERE/'PLAN.json')['budget']
    resource.setrlimit(resource.RLIMIT_CPU,(55,55));resource.setrlimit(resource.RLIMIT_AS,(900*1024**2,900*1024**2))
    out=HERE/args.label;out.mkdir(exist_ok=False)
    scanner=load_module('compact_scanner', ROOT/'motor_nuevo/repair17_compact_evidence_20260923/pack_evidence.py')
    scanner.PLAN['budget'].update(cpu_seconds_max=55,wall_seconds_max=90,rss_bytes_max=900*1024**2,
        single_file_bytes_max=100000000,selected_payload_bytes_max=300000000,npz_uncompressed_bytes_max=900*1024**2)
    budget=scanner.Budget();files,omitted,runs,oplinks=inventory()
    need(sum(x['bytes'] for x in files.values())<=limits['payload_bytes_max'],'Payload budget exceeded')
    for row in files.values():
        info=scanner.inspect_file(Path(row['source']),budget)
        need(row['expected_sha256'] in (None,info['sha256']),'Frozen input changed: '+row['source'])
        row.update(sha256=info['sha256'])
    report={'schema':'stage3_compact_inventory_v1','files':list(files.values()),'omitted':omitted,
            'runs':runs,'common_operator_source_aliases':oplinks,'full_organism_reproduction':False,
            'full_checkpoint_equivalence_recomputed_here':False,'budget':limits,'scan':budget.report()}
    dump(out/'INVENTORY.json',report)
    spec={'label':'stage3-functional-confirmation-20260923',
              'description':'Confirmación funcional local de orientación PN629-off; siete corridas completas y dos intentos incompletos, con poscierre. Paquete compacto de revisión: reconstruye observables, no reproduce organismo ni igualdad integral de checkpoints. El gate KC histórico sigue FAIL; no demuestra equivalencia biológica, navegación ni etapa4.',
              'files':[{'source':r['source'],'destination':r['destination']} for r in files.values()]+[{'source':str(out/'INVENTORY.json'),'destination':'INVENTORY.json'}]}
    dump(out/'PUBLICATION_SPEC.json',spec)
    if args.build:
        workflow=load_module('openmatrix_publish',ROOT/'instrumentos/openmatrix/publish.py')
        folder,manifest=workflow.prepare(spec,out/'prepared')
        dump(out/'PREPARED.json',{'prepared':str(folder),'files':manifest['files'],'published':False})
        report={'prepared':str(folder),'published':False,'files':len(manifest['files']),
                'payload_bytes':sum(r['bytes'] for r in manifest['files']),
                'archive':js(folder/'ARCHIVE.json'),'scan':budget.report()}
        dump(out/'PREFLIGHT.json',report)
    budget.check()
    print(json.dumps({'out':str(out),'files':len(files),'payload_bytes':sum(r['bytes'] for r in files.values()),
                      'cpu_s':time.process_time()-startcpu,'wall_s':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'published':False}))

if __name__=='__main__':main()
