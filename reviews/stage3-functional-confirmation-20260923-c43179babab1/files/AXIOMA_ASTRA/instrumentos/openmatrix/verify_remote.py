"""Verify any explicit OpenMatrix publication and optionally extract it safely."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,re,shutil,tempfile,urllib.request,zipfile

PREFIX='https://raw.githubusercontent.com/skynet-omega/openmatrix/'
def require(ok,message):
    if not ok:raise ValueError(message)
def safe_name(name):
    p=PurePosixPath(name)
    require(bool(name) and str(p)==name and not p.is_absolute() and
            '..' not in p.parts and '\\' not in name and name!='.','Unsafe archive path')
    return name
def digest(stream):
    h=hashlib.sha256();size=0
    for b in iter(lambda:stream.read(1024*1024),b''):h.update(b);size+=len(b)
    return size,h.hexdigest()
def validate_archive(path,expected):
    names=[safe_name(f['path']) for f in expected]
    require(len(set(names))==len(names) and 'MANIFEST.json' not in names,'Duplicate/reserved inventory path')
    with zipfile.ZipFile(path) as z:
        members=z.infolist()
        require(len(members)==len(names)+1 and set(z.namelist())==set(names)|{'MANIFEST.json'},'Archive membership differs from explicit inventory')
        for info in members:
            safe_name(info.filename)
            require(not info.is_dir() and (info.external_attr>>16)&0o170000!=0o120000,'Archive links/directories rejected')
        require(z.getinfo('MANIFEST.json').file_size<=2*1024*1024,'Manifest size')
        manifest=json.loads(z.read('MANIFEST.json'))
        require(manifest['files']==expected,'Embedded inventory differs')
        for entry in expected:
            require(z.getinfo(entry['path']).file_size==entry['bytes'],'Member size differs')
            with z.open(entry['path']) as stream:size,sha=digest(stream)
            require(size==entry['bytes'] and sha==entry['sha256'],'Member hash differs: '+entry['path'])
    return len(names)

def verify(publication,receipt,*,extract=None,archive_out=None):
    publication=Path(publication);receipt=Path(receipt);r=json.loads(publication.read_text())
    require(re.fullmatch('[a-f0-9]{40}',r['commit']) is not None,'Invalid commit')
    base=PREFIX+r['commit']+'/'+safe_name(r['snapshot'])+'/'
    require(r['raw_index']==base+'README.md','Unexpected remote publication URL')
    if extract is not None:require(not Path(extract).exists(),'Extraction must be a new directory')
    if archive_out is not None:require(not Path(archive_out).exists(),'Archive output already exists')
    def metadata(name):
        with urllib.request.urlopen(base+safe_name(name),timeout=60) as stream:
            value=stream.read(2*1024*1024+1)
        require(len(value)<=2*1024*1024,'Remote metadata too large')
        return json.loads(value)
    public=metadata('MANIFEST.json');require(public['files']==r['files'],'Public inventory differs from local publication')
    a=metadata('ARCHIVE.json');parts=a['parts']
    require(0<a['archive_bytes']<=1024**3 and 0<len(parts)<=100,'Archive budget')
    require(sum(p['bytes'] for p in parts)==a['archive_bytes'],'Part sizes do not add up')
    require([p['path'] for p in parts]==[f'evidence.zip.part{i:03d}' for i in range(len(parts))],'Part order')
    receipt.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='readback_',dir=receipt.parent) as temporary:
        archive=Path(temporary)/'evidence.zip';whole=hashlib.sha256();total=0
        with archive.open('xb') as out:
            for part in parts:
                sha=hashlib.sha256();size=0
                with urllib.request.urlopen(base+part['path'],timeout=60) as stream:
                    for b in iter(lambda:stream.read(1024*1024),b''):
                        size+=len(b);require(size<=part['bytes'],'Part exceeded declared size')
                        sha.update(b);whole.update(b);out.write(b)
                require(size==part['bytes'] and sha.hexdigest()==part['sha256'],'Part hash differs')
                total+=size
        require(total==a['archive_bytes'] and whole.hexdigest()==a['archive_sha256'],'Combined archive differs')
        members=validate_archive(archive,r['files'])
        if extract is not None:
            dest=Path(extract);dest.mkdir(parents=True,exist_ok=False)
            with zipfile.ZipFile(archive) as z:
                for info in z.infolist():
                    target=dest/info.filename;target.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(info) as source,target.open('xb') as out:shutil.copyfileobj(source,out)
        if archive_out is not None:
            with archive.open('rb') as source,Path(archive_out).open('xb') as out:shutil.copyfileobj(source,out)
    result={'commit':r['commit'],'archive_sha256':a['archive_sha256'],'verified_bytes':total,
            'verified_members':members,'all_binary_downloads_verified':True,
            'extraction':None if extract is None else str(Path(extract).resolve()),
            'scope':'Published bytes and explicit inventory verified; scientific reproduction requires separate execution'}
    receipt.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('publication',type=Path);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--extract',type=Path);p.add_argument('--archive-out',type=Path)
    a=p.parse_args();print(json.dumps(verify(a.publication,a.receipt,extract=a.extract,archive_out=a.archive_out)))
