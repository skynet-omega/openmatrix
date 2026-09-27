"""Final verifier adds the independently exposed law/clock gaps to the base."""
import argparse,json,time
from pathlib import Path
from analyze50 import HERE
from verify50 import verify
from verify_dng_law50 import verify_all

def complete(folder=HERE,manifest=True):
    start=time.process_time()
    extra=verify_all(folder);base=verify(folder,manifest)
    return dict(base=base,additional_DNg_checks=extra,CPU_s=time.process_time()-start,
                new_neural_ms=0,acquisition_and_primary_decision_unchanged=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=HERE);p.add_argument('--without-manifest',action='store_true');a=p.parse_args()
    print(json.dumps(complete(a.folder,not a.without_manifest)))
