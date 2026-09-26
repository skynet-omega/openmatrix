"""Export the physical state and unchanged controller for independent replay.

This one-time exporter reads the historical tree. The exported replay never
imports that tree, loads neural state, or changes neural/muscle parameters.
"""
from pathlib import Path
import ast
import difflib
import hashlib
import importlib.util
import json
import shutil
import sys
import platform

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
DONOR = ROOT/'campanas/etapa45_navigation_wind_20260925_40'
PREP = DONOR/'navigation_minus_filtered_wind_03/prepared_state'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def main():
    need(__debug__, 'Historical loaders require normal Python')
    out = HERE/'inputs'
    out.mkdir(exist_ok=False)
    manifest = json.loads((PREP/'MANIFEST.json').read_text())
    for name in ('session.json','session.npz','prosthesis.json','prosthesis.npz'):
        need(sha(PREP/name) == manifest['files'][name]['sha256'], 'Prepared source hash: '+name)
    sys.path.insert(0, str(OLD/'src'))
    from rh_tarsal_body import RHTarsalBody
    from session_io import read_state, write_state
    # Decode only the body subtree; never materialize the CNS arrays.
    desc = json.loads((PREP/'session.json').read_text())['body']
    with np.load(PREP/'session.npz', allow_pickle=False) as arrays:
        def decode(v):
            if isinstance(v, dict):
                if set(v) == {'__array__'}:
                    return arrays[v['__array__']].copy()
                return {k: decode(x) for k, x in v.items()}
            if isinstance(v, list):
                return [decode(x) for x in v]
            return v
        body_state = decode(desc)
    controller_state = read_state(PREP/'prosthesis')['controller']
    body = RHTarsalBody.from_state(body_state)
    try:
        need(body.dt == 25e-6, 'Wrong physical clock')
        need(not np.any(body.data.ctrl) and not np.any(body.data.qfrc_applied)
             and not np.any(body.data.xfrc_applied), 'Native forces not withdrawn')
        # Export model before the restored contact controller applies its patch.
        mj.mj_saveModel(body.model, str(out/'body.mjb'))
        write_state(out/'physical', {'integration':body.integration_state(),
                    'spec':body.spec, 'steps':body.steps, 'dt':body.dt,
                    'controller':controller_state})
        contact = OLD/'work/stage2_contact_prosthesis_20260915/controller.py'
        synthetic = OLD/'work/stage2_synthetic_prosthesis_20260915/controller.py'
        prototype = synthetic.with_name('prototype.py')
        spec = importlib.util.spec_from_file_location('export_contact41', contact)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ctrl = mod.Controller.from_state(body.model, body.data, controller_state)
        np.savez_compressed(out/'reference.npz', reference=ctrl.recovery.reference,
                            dt=np.array(ctrl.recovery.reference_dt), roots=ctrl.recovery.reference_roots)
        sources = {str(p):sha(p) for p in (contact,synthetic,prototype,PREP/'MANIFEST.json',
                    PREP/'session.json',PREP/'session.npz',PREP/'prosthesis.json',PREP/'prosthesis.npz',
                    DONOR/'motor_wind.py',DONOR/'run_navigation.py',Path(__file__))}
        # Preserve controller classes literally. Only the unused prototype
        # driver and HDF5 loading are replaced with the exact exported arrays.
        for src in (contact,synthetic):
            dest = HERE/'vendor'/src.relative_to(OLD)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dest)
        source = prototype.read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='LegServo')
        class_source = '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'
        portable = ('"""Literal LegServo donor; exact saved reference replaces HDF5 I/O."""\n'
                    'from pathlib import Path\nimport numpy as np\n'
                    'LEGS=[f"T{i}_{side}" for i in (1,2,3) for side in ("left","right")]\n\n'
                    +class_source+'\ndef load_reference(model,servo):\n'
                    '    with np.load(Path(__file__).resolve().parents[3]/"inputs/reference.npz",allow_pickle=False) as z:\n'
                    '        return z["reference"].copy(),float(z["dt"]),z["roots"].copy()\n')
        (HERE/'vendor/work/stage2_synthetic_prosthesis_20260915/prototype.py').write_text(portable)
        originals = HERE/'originals'
        originals.mkdir()
        shutil.copy2(prototype,originals/'prototype.py')
        shutil.copy2(DONOR/'motor_wind.py',originals/'motor_wind.py')
        shutil.copy2(OLD/'src/session_io.py',HERE/'state_io.py')
        (HERE/'PROTOTYPE_DIFF.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),portable.splitlines(True),fromfile='originals/prototype.py',tofile='vendor/work/stage2_synthetic_prosthesis_20260915/prototype.py')))
        for name in ('traces.npz','GAUSSIAN_SPEC.json','RESULT.json','MOTOR_WIND_AUDIT.json','MERGE_RECEIPT.json'):
            shutil.copy2(DONOR/'merged_full_01'/name,out/('donor_'+name))
        environment = {'python':platform.python_version(),'numpy':np.__version__,'mujoco':mj.__version__,
                       'platform':platform.platform(),'brain_loaded':False,
                       'model_nq':body.model.nq,'model_nv':body.model.nv,
                       'MJB_portability':'Pinned MuJoCo version; model recompilation from source not claimed',
                       'omitted_shadow_state':'Muscle activation shadows do not enter controller.torque; all native forces withdrawn. Full2s physical identity is a mandatory falsifier.'}
        (HERE/'ENVIRONMENT.json').write_text(json.dumps(environment,indent=2)+'\n')
        (HERE/'INPUT_PROVENANCE.json').write_text(json.dumps({'sources':sources,'exported':{str(p.relative_to(HERE)):sha(p) for p in out.iterdir()},'scope':'Complete executable physical replay capsule, not CNS continuation'},indent=2)+'\n')
        print(json.dumps({'status':'EXPORTED','bytes':sum(p.stat().st_size for p in out.iterdir()),'environment':environment}))
    finally:
        body.close()


if __name__ == '__main__':
    main()
