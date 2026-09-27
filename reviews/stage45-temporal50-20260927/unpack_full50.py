"""Materialize original files by streaming their lossless archive recipes."""
import argparse,hashlib,json
from pathlib import Path
import zipfile

def unpack(archive,destination,prefix=''):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    count=0
    with zipfile.ZipFile(archive) as z:
        m=json.loads(z.read('MANIFEST.json'))
        for item in m['files']:
            path=Path(item['path'])
            if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe archive path')
            if prefix and not path.as_posix().startswith(prefix):continue
            out=destination/path;out.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
            parts=item.get('segments',[dict(storage=item.get('storage'))])
            with out.open('xb') as f:
                for part in parts:
                    with z.open(part['storage']) as stream:
                        for b in iter(lambda:stream.read(4*1024**2),b''):
                            h.update(b);size+=len(b);f.write(b)
            if h.hexdigest()!=item['sha256'] or size!=item['bytes']:raise ValueError('Hash mismatch '+str(path))
            count+=1
    return dict(files_materialized=count,scope='Original bytes materialized; no GPU simulation executed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('destination',type=Path);p.add_argument('--prefix',default='');a=p.parse_args()
    print(json.dumps(unpack(a.archive,a.destination,a.prefix)))
