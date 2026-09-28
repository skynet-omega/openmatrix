"""Remove only archive-identical transport copies after retaining their receipts."""
from pathlib import Path
import argparse,hashlib,json,shutil,time
H=Path(__file__).resolve().parent
EX=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO')
STEM='ETAPA45_REPARACION_20260927_52'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['local','remote']);a=p.parse_args()
    cpu=time.process_time();local=json.loads((H/'LOCAL_DELIVERY.json').read_text())
    archive=Path(local['zip'])
    if sha(archive)!=local['sha256']:raise ValueError('Primary compact ZIP identity')
    receipt=H/('CLEANUP_'+a.mode.upper()+'.json')
    if receipt.exists():raise ValueError('Immutable cleanup receipt')
    tag='LOCAL' if a.mode=='local' else 'REMOTO'
    root=EX/'recibos'/(STEM+'_'+tag)/STEM
    manifest=json.loads((H/'MANIFEST.json').read_text())
    files=dict(manifest['files'])
    files['MANIFEST.json']=dict(bytes=(H/'MANIFEST.json').stat().st_size,sha256=sha(H/'MANIFEST.json'))
    extra='aporte_motor52/RECOMPUTE_DELIVERY.json'
    actual={str(f.relative_to(root)) for f in root.rglob('*') if f.is_file()}
    if actual!=set(files)|{extra}:raise ValueError('Unexpected files in extraction; do not remove')
    total=0
    for name,v in files.items():
        f=root/name
        if f.is_symlink() or f.stat().st_size!=v['bytes'] or sha(f)!=v['sha256']:raise ValueError('Extraction differs: '+name)
        total+=f.stat().st_size
    saved=H/'cierre'/('RECOMPUTE_'+tag+'.json');saved.parent.mkdir(exist_ok=True)
    if saved.exists():raise ValueError('Preserve recomputation receipt')
    shutil.copy2(root/extra,saved);total+=(root/extra).stat().st_size
    paths=[str(root)]
    # The original compact ZIP and scientific files remain. This directory is
    # solely an already verified extraction, never a managed worktree.
    shutil.rmtree(root)
    root.parent.rmdir()
    if a.mode=='remote':
        remote=json.loads((H/'REMOTE_DELIVERY.json').read_text())
        if remote['archive_sha256']!=local['sha256']:raise ValueError('Remote identity receipt')
        downloaded=EX/'recibos'/(STEM+'_DESCARGADO.zip')
        if sha(downloaded)!=local['sha256']:raise ValueError('Downloaded duplicate identity')
        total+=downloaded.stat().st_size;downloaded.unlink();paths.append(str(downloaded))
        scope=json.loads((H/'PUBLICATION_SCOPE.json').read_text())
        for name in remote['parts']:
            f=EX/'salida'/name;v=scope['files'][name]
            if sha(f)!=v['sha256'] or f.stat().st_size!=v['bytes']:raise ValueError('Transport part changed')
            total+=f.stat().st_size;f.unlink();paths.append(str(f))
    receipt.write_text(json.dumps(dict(paths_removed=paths,bytes_released=total,
        preserved_primary_zip=str(archive),archive_sha256=local['sha256'],
        preserved_verification_receipt=str(saved),CPU_s=time.process_time()-cpu,
        no_scientific_input_removed=True),indent=2)+'\n')
    print(receipt.read_text())
if __name__=='__main__':main()
