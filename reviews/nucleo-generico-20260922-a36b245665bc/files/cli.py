"""Run a declarative model, inspect committed states, and apply explicit scheduled edits."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime import Engine
from model import require

def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--model',type=Path);g.add_argument('--resume',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--until',type=float,required=True)
    p.add_argument('--mode',choices=['fast','precise'],default='precise');p.add_argument('--route',choices=['A','C'],default='A')
    p.add_argument('--backend',choices=['cpu','gpu'],default='gpu');p.add_argument('--edits',type=Path)
    p.add_argument('--scan',action='append',default=[],help='Population:state, e.g. cells:v')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    e=Engine.restore(a.resume) if a.resume else Engine(json.loads(a.model.read_text()),a.backend,a.mode,a.route)
    edits=json.loads(a.edits.read_text()) if a.edits else []
    require(isinstance(edits,list),'edit schedule must be a list')
    last=e.t
    for action in edits:
        require(set(action)=={'at','model'},'schedule entries need at and model')
        require(last<=action['at']<=a.until,'ordered edit times within simulation required');last=action['at']
    receipts=[]
    for i,action in enumerate(edits):
        # An external intervention has an explicit before checkpoint. Removed material is preserved as evidence.
        e.advance(action['at']);e.checkpoint(a.out/f'before_edit_{i:03d}')
        path=(a.edits.parent/action['model']).resolve()
        receipts.append(e.replace(json.loads(path.read_text())))
    e.advance(a.until);e.checkpoint(a.out/'final_checkpoint')
    scans=[]
    for name in a.scan:
        pop,field=name.split(':',1);scans.append(e.scan(pop,field))
    result={'model_sha256':e.model.identity,'time':e.t,'mode':e.profile,'algorithm':e.algorithm,'backend':e.backend,'scans':scans,'edits':receipts,'coverage':'deterministic smooth ODE; no automatic spikes/delays/noise; C failed HH validation'}
    (a.out/'RUN.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'time':e.t,'out':str(a.out)},indent=2))
if __name__=='__main__':main()
