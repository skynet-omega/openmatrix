"""Explicit evidence selection; no credential/config directories included."""
from pathlib import Path
import argparse
import importlib.util
import json
import tarfile
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');a=p.parse_args()
    archive=HERE/'model.tar.gz'
    if not archive.exists():
        with tarfile.open(archive,'w:gz') as tar:tar.add(HERE/'inputs/body.mjb',arcname='inputs/body.mjb')
    diff=HERE/'PROTOTYPE_DIFF.txt'
    if not diff.exists():shutil.copyfile(HERE/'PROTOTYPE_DIFF.patch',diff)
    names=['PLAN.json','SOURCE_LOCK.json','INPUT_PROVENANCE.json','ENVIRONMENT.json',
      'REPAIR_EXECUTION.json','VERIFIER_REPAIR.json','FIRST_WIND_FORCE.json','FIRST_WIND_FORCE.npz',
      'FORCE_PROBE_PORTABLE.json','run_replay.py','prepare_inputs.py','state_io.py','verify.py','verify_clock.py',
      'probe_force.py','report.py','run_external_audit.py','chatgpt_audit_original.py','finalize.py',
      'README.md','REPORT.md','REPORT.png','VERIFIED.json','VERIFIED_OPTIMIZED.json',
      'CHATGPT_CODE_REAL_RESULT.json','CHATGPT_REQUEST_01.md','CHATGPT_RESPONSE_01.md','CHATGPT_RECEIPT_01.json',
      'EXTERNAL_STATUS.json','JEV_SPEC.json','PREPARE.log','RUN_01.log','RUN_02.log','VERIFY_OPTIMIZED.log',
      'model.tar.gz','PROTOTYPE_DIFF.txt']
    paths=[HERE/n for n in names]
    for directory in ('vendor','originals','run_01','run_02','jev_01'):
        paths.extend(p for p in (HERE/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend(p for p in (HERE/'inputs').iterdir() if p.is_file() and p.name!='body.mjb')
    # A later closure may add receipts without altering scientific sources.
    for n in ('CLEAN_VERIFIED.json','CONTINUITY_RECEIPT.json','CHATGPT_CODE_REAL_RESULT_O.json'):
        if (HERE/n).exists():paths.append(HERE/n)
    spec={'label':'stage45-postwind-diagnosis-20260925-41',
      'description':'Physical replay of campaign40: exact historical identity, postwind yaw/forward ablations, preserved failed continuous replay and cold-force-cache finding. No neural simulation or Stage4/5 admission. Executable physical capsule, not whole-organism checkpoint.',
      'files':[{'source':str(p),'destination':str(p.relative_to(HERE))} for p in sorted(paths)]}
    (HERE/'PUBLICATION_MANIFEST.json').write_text(json.dumps(spec,indent=2)+'\n')
    path=ROOT/'instrumentos/openmatrix/publish.py'
    ms=importlib.util.spec_from_file_location('publish41',path);mod=importlib.util.module_from_spec(ms);ms.loader.exec_module(mod)
    mod.ALLOWED.append(HERE)
    if a.publish:
        mod.REPO=HERE/'publication_checkout'
        mod.publish(spec,HERE/'PUBLICATION.json')
    else:
        folder,public=mod.prepare(spec,HERE/'prepared')
        (HERE/'PREPARED_PACKAGE.json').write_text(json.dumps({'prepared':str(folder),'files':public['files']},indent=2)+'\n')
        print(folder)


if __name__=='__main__':main()
