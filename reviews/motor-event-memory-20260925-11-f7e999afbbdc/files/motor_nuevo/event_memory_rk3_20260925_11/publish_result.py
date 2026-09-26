"""Publish the reviewed evidence through the existing explicit-manifest workflow."""
from pathlib import Path
import importlib.util, json, zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def main():
    verification=json.loads((HERE/'VERIFY1000.json').read_text())
    if verification['status']!='PASS_SAVED_EVIDENCE':raise ValueError('Evidence not verified')
    module=ROOT/'instrumentos/openmatrix/publish.py'
    spec=importlib.util.spec_from_file_location('openmatrix_publish',module)
    publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)
    # Another campaign has unpublished files in the shared exchange checkout.
    # This sparse clone was created for this delivery and preserves its work.
    publisher.REPO=ROOT/'intercambio/event_memory_20260925_11_publish'
    with zipfile.ZipFile(HERE/'MOTOR_EVENT_MEMORY_RK3_RESULTADOS.zip') as archive:
        names=json.loads(archive.read('PACKAGE_MANIFEST.json'))
    files=[]
    for name in sorted(names):
        source=ROOT/name
        if source.suffix=='.so':continue  # Source-only public core; no GPU binary needed for comparison.
        if source.suffix=='.diff':
            copy=HERE/'public_sources'/source.with_suffix('.txt').name
            copy.parent.mkdir(exist_ok=True);copy.write_bytes(source.read_bytes());source=copy
        files.append({'source':str(source),'destination':name})
    evidence={'label':'motor-event-memory-20260925-11',
              'description':'RK3 residente: una mejora de propuesta tras eventos; pareja real de100ms y confirmación de1s. Código del núcleo y evidencia para recalcular comparaciones en CPU. No contiene checkpoint/modelo para reiniciar el organismo. Ver RESULTADOS.md para coste, alcance funcional y limitaciones.',
              'files':files}
    path=HERE/'PUBLIC_FILES.json'
    if path.exists():raise ValueError('New publication manifest required')
    path.write_text(json.dumps(evidence,indent=2)+'\n')
    publisher.publish(evidence,HERE/'PUBLICATION.json')

if __name__=='__main__':main()
