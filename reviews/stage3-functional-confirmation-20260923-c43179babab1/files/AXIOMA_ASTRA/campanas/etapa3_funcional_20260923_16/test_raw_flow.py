"""Corrupt real recorded native flow; each changed datum must be rejected."""
from pathlib import Path
import json,shutil,tempfile
from verify_all import raw_flow
HERE=Path(__file__).resolve().parent
def main():
    source=HERE/'reference_odor_right_01';operator=source/'prepared_state'
    raw_flow(source,operator);rejected=[]
    with tempfile.TemporaryDirectory(prefix='native_flow_guard_') as tmp:
        root=Path(tmp);(root/'flow').mkdir()
        for name in ('FLOW.npz','METADATA.json'):shutil.copyfile(source/'flow'/name,root/'flow'/name)
        original=(source/'flow/FLOW.jsonl').read_text().splitlines()
        for kind in ('timestamp','theta','rate','raw_input'):
            lines=list(original);row=json.loads(lines[0]);identity=max(row['rows'],key=lambda k:row['rows'][k]['target'])
            if kind=='timestamp':row['time_ns']+=1
            else:row['rows'][identity][{'theta':'theta','rate':'rate','raw_input':'raw_signed'}[kind]]+=.001
            lines[0]=json.dumps(row);(root/'flow/FLOW.jsonl').write_text('\n'.join(lines)+'\n')
            try:raw_flow(root,operator)
            except ValueError:rejected.append(kind)
            else:raise RuntimeError('Corruption escaped '+kind)
    print(json.dumps({'real_flow_control_passed':True,'corruptions_rejected':rejected,'GPU_executed':False}))
if __name__=='__main__':main()
