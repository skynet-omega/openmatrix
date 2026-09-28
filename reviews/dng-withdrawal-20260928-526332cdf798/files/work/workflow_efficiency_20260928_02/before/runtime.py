"""Bounded recipe execution over registered scientific components.

This is an operational sandbox (new outputs, subprocess, deadline), not OS
isolation. It never promotes a checkpoint or imports a historical script.
Content identity includes declared data, all workbench code and installed
distributions. COMPLETE means execution, not a positive scientific result.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid

from session_io import sha256


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def scalar_metrics(value):
    """Tracking/comparison metadata; full observations stay in result.json."""
    return {k:v for k,v in value.items() if type(v) in (int,float)} if isinstance(value,dict) else {}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    os.replace(tmp, path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def environment():
    packages = sorted((d.metadata['Name'].lower(), d.version)
                      for d in metadata.distributions() if d.metadata['Name'])
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                platform=platform.platform(), machine=platform.machine(),
                packages=packages, threads={'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1'})


class Workbench:
    def __init__(self, root, store=None, tracking=True):
        self.root = Path(root).resolve()
        self.store = Path(store).resolve() if store else self.root/'runs/workbench'
        self.tracking = tracking
        self._client = None

    def client(self):
        if self._client is None:
            from mlflow.tracking import MlflowClient
            self.store.mkdir(parents=True, exist_ok=True)
            self._client = MlflowClient(tracking_uri='sqlite:///'+str(self.store/'tracking.sqlite'))
        return self._client

    def recipe(self, name):
        path = Path(name)
        if not path.is_file():
            path = self.root/'config/workflows'/f'{name}.json'
        spec = json.loads(path.read_text())
        from .tasks import COMPONENTS
        if spec.get('schema') != 'matrix_workflow_v1' or not spec.get('scope'):
            raise ValueError('Recipe requires schema matrix_workflow_v1 and explicit scope')
        seen = set()
        for step in spec.get('steps', []):
            sid = step.get('id', '')
            if 'scope' in step and (not isinstance(step['scope'],str) or not step['scope'].strip()):
                raise ValueError('Explicit step scope must be a nonempty scientific contract')
            if not re.fullmatch(r'[A-Za-z0-9_-]+', sid) or sid in seen:
                raise ValueError('Step IDs must be unique simple names')
            if step.get('component') not in COMPONENTS:
                raise ValueError('Unregistered component: '+str(step.get('component')))
            for value in step.get('inputs', {}).values():
                if isinstance(value, dict):
                    if set(value) != {'step','artifact'} or value['step'] not in seen:
                        raise ValueError('Dependencies must refer to a preceding step and artifact')
                    artifact = Path(value['artifact'])
                    if artifact.is_absolute() or '..' in artifact.parts:
                        raise ValueError('Artifact reference must remain inside a step output')
                elif not isinstance(value, str):
                    raise ValueError('Input must be a file path or explicit step/artifact reference')
            seen.add(sid)
        if not seen:
            raise ValueError('Empty workflow')
        return spec

    def plan(self, name):
        spec = self.recipe(name)
        rows = []
        for step in spec['steps']:
            inputs = {}
            for key, value in step.get('inputs', {}).items():
                if isinstance(value, dict):
                    inputs[key] = value
                else:
                    p = (self.root/value).resolve()
                    inputs[key] = dict(path=str(p), available=p.is_file(),
                                       bytes=p.stat().st_size if p.is_file() else None)
            sources = [self.root / v for v in step.get('inputs', {}).values()
                       if isinstance(v, str) and (self.root / v).is_file()]
            dependencies = self.source_dependencies(sources)
            rows.append(dict(id=step['id'], component=step['component'],
                             scope=step.get('scope',spec['scope']),
                             inputs=inputs, parameters=step.get('parameters', {}),
                             source_dependencies=dependencies))
        return dict(id=spec['id'], scope=spec['scope'], steps=rows,
                    execution='registered subprocesses, new outputs, deadline; no OS isolation or automatic CNS promotion')

    def code_identity(self):
        paths = sorted((self.root/'src/matrix_workbench').glob('*.py'))
        paths += [self.root/'src/session_io.py']
        dependency_source = self.root/'src/source_dependencies.py'
        if dependency_source.is_file():
            paths.append(dependency_source)
        return {str(p.relative_to(self.root)):sha256(p) for p in paths}

    def source_dependencies(self, sources):
        from source_dependencies import source_dependencies
        return source_dependencies(self.root, sources,
            covered=[*map(lambda p: str(p.relative_to(self.root)),
                          (self.root/'src/matrix_workbench').glob('*.py')),
                     'src/session_io.py', 'src/source_dependencies.py'])

    def receipt(self, path):
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        result = dict(path=str(p), bytes=p.stat().st_size, sha256=sha256(p))
        # Standalone scientific helpers are explicit inputs, outside the package
        # code identity. Preserve their exact versions in the existing store too.
        # Dataset/checkpoint payloads remain referenced, never copied here.
        if p.suffix == '.py' and p.is_relative_to(self.root):
            self.preserve_code({str(p.relative_to(self.root)):result['sha256']})
        return result

    def preserve_code(self, code):
        """Save small source blobs once by content, never datasets/checkpoints."""
        for rel, checksum in code.items():
            blob = self.store/'source_blobs'/checksum
            with locked(self.store/'locks'/f'source-{checksum}.lock'):
                if blob.exists():
                    if sha256(blob) != checksum:
                        raise ValueError('Preserved source was altered: '+str(blob))
                    continue
                blob.parent.mkdir(parents=True, exist_ok=True)
                temporary = blob.with_name(checksum+'.'+uuid.uuid4().hex+'.tmp')
                shutil.copyfile(self.root/rel, temporary)
                if sha256(temporary) != checksum:
                    temporary.unlink()
                    raise RuntimeError('Implementation changed before preservation')
                os.replace(temporary,blob)

    @staticmethod
    def verify_outputs(record):
        for rel, info in record['outputs'].items():
            path = (Path(record['output_dir'])/rel).resolve()
            if not path.is_relative_to(Path(record['output_dir']).resolve()):
                raise ValueError('Output escaped its run')
            if not path.is_file() or sha256(path) != info['sha256']:
                raise ValueError('Cached output missing/changed: '+str(path)+'; use --force to create a new attempt')

    def _track_start(self, component, key, parameters, attempt):
        if not self.tracking:
            return None
        c = self.client()
        with locked(self.store/'locks/tracking.lock'):
            experiment = c.get_experiment_by_name('MATRIX workbench')
            eid = experiment.experiment_id if experiment else c.create_experiment(
                'MATRIX workbench', artifact_location=(self.store/'tracking_artifacts').as_uri())
        run = c.create_run(eid, tags={'mlflow.runName':component, 'matrix.key':key,
                                    'matrix.output':str(attempt), 'matrix.qualification':'not_implied_by_execution'})
        rid = run.info.run_id
        for k, v in parameters.items():
            c.log_param(rid, k, canonical(v)[:5900])
        return rid

    def _track_end(self, run_id, state, output):
        if run_id is None:
            return
        c = self.client()
        if state['status'] == 'COMPLETE':
            metrics = scalar_metrics(state.get('metrics', {}))
            # Structured observations remain in result.json. Only scalar maps
            # have an MLflow metric representation; a list is not a failed run.
            for key, value in metrics.items():
                if type(value) in (int,float):
                    c.log_metric(run_id, key, value)
            c.log_artifact(run_id, str(output/'result.json'))
        c.log_artifact(run_id, str(output/'provenance.json'))
        c.set_terminated(run_id, status='FINISHED' if state['status']=='COMPLETE' else 'FAILED')

    def execute(self, component, inputs, parameters, env, code, scope, force=False):
        from .tasks import COMPONENTS
        contract = COMPONENTS[component]
        receipts = {k:self.receipt(v) for k,v in inputs.items()}
        dependencies = self.source_dependencies(inputs.values())
        identity = dict(schema='matrix_unit_v1', component=component, contract=contract,
                        inputs=receipts, parameters=parameters, environment=env, code=code,
                        source_dependencies=dependencies, scope=scope)
        # Absolute locations are provenance, not computational identity. A moved
        # byte-identical input must not trigger another calculation.
        key = digest({**identity, 'inputs':{k:{f:v[f] for f in ('bytes','sha256')} for k,v in receipts.items()}})
        unit = self.store/'units'/key
        with locked(self.store/'locks'/f'{key}.lock'):
            complete = unit/'complete.json'
            if complete.exists() and not force:
                record = json.loads(complete.read_text())
                self.verify_outputs(record)
                return {**record, 'reused':True}
            self.preserve_code(code)
            self.preserve_code(dependencies)
            output = unit/'attempts'/uuid.uuid4().hex
            output.mkdir(parents=True, exist_ok=False)
            state = dict(schema='matrix_execution_v1', key=key, component=component,
                         status='RUNNING', started=stamp(), output_dir=str(output),
                         scope=scope, qualification=contract['qualification'], inputs=receipts)
            write_json(output/'provenance.json', identity)
            write_json(output/'request.json', dict(component=component, inputs=inputs, parameters=parameters))
            write_json(output/'state.json', state)
            run_id = None
            start = time.monotonic()
            try:
                run_id = self._track_start(component,key,parameters,output)
                worker_env = os.environ.copy()
                worker_env.update(PYTHONPATH=str(self.root/'src'), PYTHONDONTWRITEBYTECODE='1',
                                  OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                                  MPLCONFIGDIR=str(output/'.mplconfig'))
                with (output/'stdout.log').open('w') as log:
                    subprocess.run([sys.executable,'-m','matrix_workbench.worker',str(output/'request.json')],
                                   cwd=output, env=worker_env, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=contract['maximum_seconds'], check=True)
                if self.code_identity() != code:
                    raise RuntimeError('Implementation changed during execution; result not cached')
                if self.source_dependencies(inputs.values()) != dependencies:
                    raise RuntimeError('Scientific dependency changed during execution; result not cached')
                for info in receipts.values():
                    if sha256(info['path']) != info['sha256']:
                        raise RuntimeError('Input changed during execution; result not cached')
                result = json.loads((output/'result.json').read_text())
                outputs = {}
                for p in sorted(output.rglob('*')):
                    if p.is_symlink():
                        raise ValueError('Output symlinks are not accepted')
                    if p.is_file() and p.name not in {'state.json','stdout.log'}:
                        outputs[str(p.relative_to(output))] = dict(bytes=p.stat().st_size, sha256=sha256(p))
                if not outputs or 'result.json' not in outputs:
                    raise ValueError('Worker did not publish result.json')
                state.update(status='COMPLETE', metrics=scalar_metrics(result.get('metrics',{})), outputs=outputs,
                             finished=stamp(), seconds=time.monotonic()-start, tracking_run_id=run_id)
                try:
                    self._track_end(run_id,state,output)
                except Exception as tracking_error:
                    # Completed, verified outputs must not be recomputed merely
                    # because the auxiliary tracking sink failed at delivery.
                    state['tracking_status'] = 'FAILED'
                    state['tracking_error'] = f'{type(tracking_error).__name__}: {tracking_error}'
                    write_json(output/'tracking_error.json', {'error':state['tracking_error']})
                write_json(output/'state.json',state)
                write_json(complete,state)
                return {**state, 'reused':False}
            except BaseException as exc:
                state.update(status='FAILED', error=f'{type(exc).__name__}: {exc}',
                             finished=stamp(), seconds=time.monotonic()-start, tracking_run_id=run_id)
                write_json(output/'state.json',state)
                try:
                    self._track_end(run_id,state,output)
                except Exception as tracking_error:
                    write_json(output/'tracking_error.json', {'error':str(tracking_error)})
                raise

    def run(self, name, force=False):
        spec = self.recipe(name)
        preflight = None
        if spec.get('operational_preflight') is True:
            doctor = self.root/'scripts/project_doctor.py'
            check = subprocess.run([sys.executable,str(doctor),'--details'],cwd=self.root,
                                   capture_output=True,text=True,timeout=60,check=False)
            try:
                preflight = json.loads(check.stdout)
            except (ValueError, TypeError) as error:
                raise ValueError('Operational preflight did not return a valid report') from error
            if check.returncode != 0 or preflight.get('ready') is not True:
                raise ValueError('Operational preflight blocks workflow: '+str(preflight.get('blockers', [])))
            preflight['source_sha256'] = sha256(doctor)
        env, code = environment(), self.code_identity()
        results = {}
        for step in spec['steps']:
            inputs = {}
            for key,value in step.get('inputs',{}).items():
                inputs[key] = str((Path(results[value['step']]['output_dir'])/value['artifact']).resolve()) if isinstance(value,dict) else str((self.root/value).resolve())
            results[step['id']] = self.execute(step['component'],inputs,step.get('parameters',{}),
                                               env,code,step.get('scope',spec['scope']),force=force)
        summary = dict(schema='matrix_workflow_execution_v1', workflow=spec['id'], scope=spec['scope'],
                       recipe=spec, completed=stamp(), steps=results, qualified_CNS=False)
        if spec.get('evidence_record'):
            from lab_evidence import register_analysis
            try:summary['laboratory_index']=register_analysis(self.root,spec['evidence_record'],summary)
            except (OSError,ValueError,KeyError,sqlite3.Error) as exc:
                # Analysis output survives a catalogue failure and can be
                # indexed later, without re-executing the science.
                summary['laboratory_index']={'status':'failed','error':str(exc)}
        if preflight is not None:
            summary['operational_preflight'] = preflight
        path = self.store/'workflows'/f'{digest(spec)}.json'
        write_json(path,summary)
        return summary

    def history(self):
        return [json.loads(p.read_text()) for p in sorted((self.store/'units').glob('*/attempts/*/state.json'))]

    def show(self, key):
        pieces=key.split(':')
        if len(pieces)>2 or any(not re.fullmatch('[0-9a-f]+',p) for p in pieces):
            raise ValueError('Use a hexadecimal key prefix, optionally key:attempt')
        pattern=pieces[0]+'*/complete.json' if len(pieces)==1 else pieces[0]+'*/attempts/'+pieces[1]+'*/state.json'
        paths=list((self.store/'units').glob(pattern))
        if len(paths)!=1:
            raise ValueError('Choose a unique completed unit key; matches='+str(len(paths)))
        record=json.loads(paths[0].read_text())
        if record['status']!='COMPLETE':
            raise ValueError('Only complete attempts have comparable outputs')
        self.verify_outputs(record)
        record['result']=json.loads((Path(record['output_dir'])/'result.json').read_text())
        return record

    def compare(self, left, right):
        a,b=self.show(left),self.show(right)
        pa=json.loads((Path(a['output_dir'])/'provenance.json').read_text())
        pb=json.loads((Path(b['output_dir'])/'provenance.json').read_text())
        checks=dict(component=a['component']==b['component'], scope=a['scope']==b['scope'],
                    contract=pa['contract']==pb['contract'],
                    parameters=pa['parameters']==pb['parameters'],
                    input_contents={k:v['sha256'] for k,v in pa['inputs'].items()}=={k:v['sha256'] for k,v in pb['inputs'].items()},
                    code=pa['code']==pb['code'],environment=pa['environment']==pb['environment'],
                    source_dependencies=('source_dependencies' in pa and 'source_dependencies' in pb
                                         and pa['source_dependencies']==pb['source_dependencies']))
        compatible=all(checks.values())
        am,bm=scalar_metrics(a.get('metrics',{})),scalar_metrics(b.get('metrics',{}))
        deltas={k:bm[k]-v for k,v in am.items() if k in bm} if compatible else {}
        return dict(compatible_numerical_repeat=compatible,checks=checks,deltas=deltas,
                    reason='Cross-condition scientific comparison requires an explicit registered comparator; differing units/cohorts/protocols must not be silently mixed.')

    def import_legacy(self, manifest):
        """Reference an existing seal; do not rerun, copy data or invent verification."""
        path=(self.root/manifest).resolve()
        obj=json.loads(path.read_text())
        field=next((k for k in ('files','artifacts','artifact_sha256')
                    if isinstance(obj.get(k),(dict,list)) and obj[k]),None)
        if field is None:
            raise ValueError('Expected nonempty preservation manifest')
        key=sha256(path)
        record=dict(schema='matrix_legacy_reference_v1', key=key, manifest=str(path),
                    kind='imported_reference_not_reexecuted', indexed_files=len(obj[field]),
                    manifest_entries_field=field,
                    payload_verification='not_rehashed_on_import', qualification='consult_original_evidence',
                    imported=stamp())
        dest=self.store/'legacy'/f'{key}.json'
        if not dest.exists():
            write_json(dest,record)
        return json.loads(dest.read_text())
