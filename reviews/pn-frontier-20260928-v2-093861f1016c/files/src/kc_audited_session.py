"""Version the source-archive repair without changing inherited CNS dynamics.

The visual session used CanonicalPathways at runtime without archiving it.
Anatomical preparation code was also outside the static import closure receipt.
Old checkpoints remain byte-identical; this boundary pins both dependencies.
"""
from pathlib import Path
import ast,copy
from kc_visual_session import KcVisualSession,SOURCES as PARENT_SOURCES
from kc_visual_brain import GpuKcVisualBrain
from kc_session_storage import save_session,load_session
from kc_spatial_session import ROOT,_fingerprints,sha256

SCHEMA='matrix_kc_audited_session_v1'
SOURCES=tuple(PARENT_SOURCES)+('canonical_pathway_probe.py','anatomical_preparation.py','kc_audited_session.py')


def source_closure(root,entry):
    """All statically named project-local imports, including inside functions."""
    root=Path(root);seen=set();pending=[entry]
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((root/name).read_text())):
            names=([node.module] if isinstance(node,ast.ImportFrom) else
                   [n.name for n in node.names] if isinstance(node,ast.Import) else [])
            pending.extend(n+'.py' for n in names if n and (root/(n+'.py')).is_file())
    return sorted(seen)


def require_covered_dependencies(root,entry,sources):
    closure=source_closure(root,entry);missing=set(closure)-set(sources)
    if missing:raise ValueError('Unarchived runtime dependencies: '+', '.join(sorted(missing)))
    return closure


class KcAuditedSession(KcVisualSession):
    @classmethod
    def from_checkpoint(cls,path):
        path=Path(path).resolve();parent=KcVisualSession.load(path)
        obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention'}
            before=_fingerprints(parent,excluded)
            closure=require_covered_dependencies(ROOT/'src','kc_audited_session.py',SOURCES)
            obj.config=copy.deepcopy(parent.config)
            obj.config['runtime_source_contract']=dict(entrypoint='kc_audited_session.py',
                static_local_imports=closure,additional_dependencies=['canonical_pathway_probe.py','anatomical_preparation.py'],
                scope='Static local imports and existing archived runtime files; third-party binaries remain subject to inherited environment checks. No claim that missing files were pinned retroactively.')
            obj.source_identity={name:sha256(ROOT/'src'/name) for name in SOURCES}
            after=_fingerprints(obj,excluded)
            checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('Source repair changed organism state')
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),
                operation='Archive all statically imported local runtime sources',time_ns=obj.time_ns,
                parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),
                preserved_state_checks=checks,neural_equations_changed=False,biological_validation=False)
            obj._validate();obj._validate_pending();obj._validate_afferent_pending();return obj
        except BaseException:obj.close();raise

    def _validate(self):
        super()._validate()
        if not self.config.get('runtime_source_contract'):raise ValueError('Missing source dependency contract')

    def state_dict(self):
        out=super().state_dict();out['schema']=SCHEMA;return out

    def _manifest_fields(self):
        out=super()._manifest_fields();out.update(schema=SCHEMA,
            runtime_source_files=len(SOURCES),runtime_static_import_closure=self.config['runtime_source_contract']['static_local_imports'],
            source_archive_repair=True,neural_equations_changed=False);return out

    def save(self,path):
        closure=require_covered_dependencies(ROOT/'src','kc_audited_session.py',SOURCES)
        if closure!=self.config['runtime_source_contract']['static_local_imports']:raise ValueError('Source import contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src','kc_audited_session.py',SOURCES)
        return load_session(path,cls,SCHEMA,SOURCES,GpuKcVisualBrain)
