"""Eventual readback of this finite queue, without starting or retrying experiments."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
HERE=Path(__file__).resolve().parent
def js(p):return json.loads(p.read_text())
def main():
    out=HERE/'FINAL_VERIFIED_01.json'
    if out.exists():raise FileExistsError(out)
    frozen=js(HERE/'VERIFICATION_SOURCES.json')
    deadline=time.monotonic()+11000
    while time.monotonic()<deadline:
        for p in ('QUEUE_ERROR.json','REMAINING_QUEUE_ERROR.json','BODY_SUPPORT_FAILURE.json'):
            if (HERE/p).exists():
                print(json.dumps({'state':'STOPPED_WITH_RETAINED_FAILURE','receipt':p,'details':js(HERE/p)}),flush=True);return 2
        q=js(HERE/'REMAINING_QUEUE.json')
        if q['state']=='EXPERIMENTS_COMPLETE_REQUIRES_RAW_DATA_VERIFICATION':
            for name,digest in frozen.items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Verifier changed while experiments ran')
            child=subprocess.run([sys.executable,'-B',str(HERE/'verify_all.py'),'--out',str(out)],capture_output=True,text=True,timeout=180)
            (HERE/'final_verifier.log').write_text(child.stdout+child.stderr)
            result=js(out);print(json.dumps({'state':'RAW_DATA_VERIFICATION_FINISHED','exit_code':child.returncode,
                'classification':result['classification'],'functional_stage3_pass':result['functional_stage3_pass'],
                'error':result.get('error'),'receipt':str(out)}),flush=True);return child.returncode
        if q['state']=='WAITING_FOR_FOUR_REFERENCES':q=js(HERE/'QUEUE.json')
        progress=None
        if q.get('current'):
            path=HERE/q['current']/'PROGRESS.jsonl'
            if path.exists():
                lines=path.read_text().splitlines()
                if lines:progress=json.loads(lines[-1])
        print(json.dumps({'state':q['state'],'current':q.get('current'),'progress':progress,'stage3_admission':False}),flush=True)
        time.sleep(60)
    raise TimeoutError('Finite monitor ceiling; inspect existing receipts, never rerun blindly')
if __name__=='__main__':raise SystemExit(main())
