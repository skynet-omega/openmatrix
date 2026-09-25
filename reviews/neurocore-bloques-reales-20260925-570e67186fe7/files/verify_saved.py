"""Verify packaged bytes and independently recompute the recorded real pair."""
from pathlib import Path
import hashlib
import json
from compare_real import compare


def need(ok,message):
    if not ok:raise ValueError(message)


def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for entry in manifest['files']:
        rel=Path(entry['path'])
        need(not rel.is_absolute() and '..' not in rel.parts,'unsafe manifest path')
        p=root/rel;need(p.is_file() and p.stat().st_size==entry['bytes'],'missing/changed file '+str(rel))
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        need(h.hexdigest()==entry['sha256'],'hash mismatch '+str(rel))
    observed=compare(root/'reference100_01',root/'candidate100_01')
    saved=json.loads((root/'PAIR100.json').read_text())
    for result in (observed,saved):
        for key in ('reference','candidate'):result[key]=Path(result[key]).name
    need(observed==saved,'saved comparison differs from recomputed evidence')
    print(json.dumps({'status':'EVIDENCE_RECOMPUTED','files':len(manifest['files']),
        'comparison_status':observed['status'],'python_optimized':not __debug__,
        'scope':'Saved real trajectories and PN/events recomputed; no new organism simulation'}))


if __name__=='__main__':main()
