"""Decision includes both motor commands, as clarified before candidate exposure."""
from pathlib import Path
import argparse,hashlib,json
from compare_runs import compare,need

def evaluate(control,candidate,ms,paired=False):
    result=compare(control,candidate,ms,paired_timing=paired)
    result['gates']['same_forward_commands']=result['fields']['command_forward_mm_s']['exact']
    result['status']='PASS' if all(result['gates'].values()) else 'FAIL'
    result['evaluator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['forward_command_note']='Explicitly covers forward as well as yaw; added before candidate execution after static code review. No engine or tolerance changes.'
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--ms',type=int,required=True)
    p.add_argument('--paired-timing',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=evaluate(a.control.resolve(),a.candidate.resolve(),a.ms,a.paired_timing)
    need(not a.out.exists(),'New output required')
    a.out.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:r[k] for k in ('status','ms','gates','timing','events','counts')},indent=2))

if __name__=='__main__':main()
