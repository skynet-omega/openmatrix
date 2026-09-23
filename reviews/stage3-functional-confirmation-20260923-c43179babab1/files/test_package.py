"""Small CPU falsifiers for publication boundaries; no scientific simulation."""
from pathlib import Path
import importlib.util, json, tempfile, zipfile, io
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def need(ok,msg):
    if not ok:raise ValueError(msg)
def rejected(fn,label):
    try:fn()
    except ValueError:return label
    raise ValueError('Corruption accepted: '+label)
def main():
    p=load('workflow',ROOT/'instrumentos/openmatrix/publish.py')
    scan=load('scanner',ROOT/'motor_nuevo/repair17_compact_evidence_20260923/pack_evidence.py')
    out=[]
    with tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic_') as td:
        d=Path(td);f=d/'sample.json';f.write_text('{"synthetic":true}\n')
        def spec(path,destination='test.json'):
            return {'label':'synthetic-preflight','description':'Synthetic boundary test','files':[{'source':str(path),'destination':destination}]}
        need(len(p.inspect_entries(spec(f)))==1,'Clean fixture rejected')
        link=d/'linked.json';link.symlink_to(f)
        out.append(rejected(lambda:p.inspect_entries(spec(link)),'symlink rejected'))
        out.append(rejected(lambda:p.inspect_entries(spec(f,'../bad.json')),'traversal rejected'))
        secret=d/'secret.json';secret.write_text('ghp_'+'x'*36)
        out.append(rejected(lambda:p.inspect_entries(spec(secret)),'credential text rejected'))
        secret_npz=d/'secret.npz';np.savez(secret_npz,synthetic=np.asarray(['ghp_'+'x'*36]))
        # NumPy unicode has interleaved NULs; store byte strings to check raw secret scanner.
        np.savez(secret_npz,synthetic=np.asarray([('ghp_'+'x'*36).encode()]))
        out.append(rejected(lambda:p.inspect_entries(spec(secret_npz)),'credential NPZ rejected'))
        out.append(rejected(lambda:scan.inspect_file(link,scan.Budget()),'stream preflight symlink rejected'))
        out.append(rejected(lambda:scan.inspect_file(secret,scan.Budget()),'extended credential scan rejected'))
        for name in ('etapa3_continuation_repair_20260923_19','etapa3_postclose_20260923_20'):
            need(len(p.inspect_entries(spec(ROOT/'campanas'/name/'PLAN.json')))==1,'Explicit authorized root rejected')
            out.append('explicit root admitted: '+name)
    print(json.dumps({'synthetic':True,'checks':out,'passed':len(out)==8,'optimized':not __debug__},indent=2))
if __name__=='__main__':main()
