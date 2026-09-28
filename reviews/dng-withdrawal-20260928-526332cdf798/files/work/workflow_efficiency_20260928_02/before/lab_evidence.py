"""Incremental discovery in the existing scientific SQLite catalogue.

Reads declared text only, never executes campaign code or infers admission.
Raw array integrity belongs to typed workbench adapters, not this text index.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sqlite3
import time
import unicodedata
import fcntl
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT = 2_000_000
JSON_NAMES = {'PLAN.json','RESULTADOS.json','RESULT.json','CIERRE.json','DECISION.json',
              'result.json','results.json','assessment.json','protocol.json','decision.json',
              'NEXT_WORK.json','REVIEW.json','STATUS.json','QUEUE_RESULT.json','SOURCES.json',
              'SOURCE_INDEX.json','FREEZE.json'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):h.update(block)
    return h.hexdigest()

def inside(root, rel):
    p=(Path(root)/rel).resolve()
    p.relative_to(Path(root).resolve())
    if any(x.startswith('.') for x in p.relative_to(root).parts):
        raise ValueError('Hidden paths are outside evidence coverage')
    return p

def config(root):
    return json.loads((Path(root)/'config/lab_evidence_v1.json').read_text())

def lease(root):
    p=Path(root)/'catalogo';p.mkdir(exist_ok=True)
    stream=(p/'lab_write.lock').open('a');fcntl.flock(stream,fcntl.LOCK_EX)
    return stream

class WriteConnection(sqlite3.Connection):
    def close(self):
        try:super().close()
        finally:
            if getattr(self,'lease',None) is not None:
                self.lease.close();self.lease=None

@contextmanager
def preserve_catalog(root):
    """Serialize full rebuild with lab writers; retain versions and analyses."""
    guard=lease(root);saved=[];p=Path(root)/'catalogo/catalogo.sqlite'
    try:
        if p.exists():
            old=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True)
            try:
                for name in ('lab_records','lab_files','lab_search','lab_meta','lab_versions','lab_analyses'):
                    schema=old.execute('SELECT sql FROM sqlite_master WHERE name=?',(name,)).fetchone()
                    if schema:saved.append((name,schema[0],old.execute('SELECT * FROM '+name).fetchall()))
            finally:old.close()
        try:
            yield
        finally:
            db=sqlite3.connect(p)
            try:
                for name,schema,rows in saved:
                    if not db.execute('SELECT 1 FROM sqlite_master WHERE name=?',(name,)).fetchone():
                        db.execute(schema)
                        if rows:db.executemany('INSERT INTO '+name+' VALUES('+','.join('?' for _ in rows[0])+')',rows)
                db.commit()
            finally:db.close()
    finally:guard.close()

def connect(root, write=False):
    p=Path(root)/'catalogo/catalogo.sqlite'
    if write:
        guard=lease(root)
        try:db=sqlite3.connect(p,timeout=30,factory=WriteConnection);db.lease=guard
        except Exception:guard.close();raise
        db.executescript('''CREATE TABLE IF NOT EXISTS lab_records(
            id TEXT PRIMARY KEY, category TEXT, entry TEXT, fingerprint TEXT, metadata TEXT);
            CREATE TABLE IF NOT EXISTS lab_files(
            path TEXT PRIMARY KEY, record_id TEXT, sha256 TEXT, bytes INTEGER);
            CREATE VIRTUAL TABLE IF NOT EXISTS lab_search USING fts5(
            record_id UNINDEXED,title,text,tokenize="unicode61 remove_diacritics 2");
            CREATE TABLE IF NOT EXISTS lab_meta(key TEXT PRIMARY KEY,value TEXT);''')
        db.execute('CREATE TABLE IF NOT EXISTS lab_versions(record_id TEXT,fingerprint TEXT,metadata TEXT,'
                   'PRIMARY KEY(record_id,fingerprint))')
        db.execute('CREATE TABLE IF NOT EXISTS lab_analyses(record_id TEXT,key TEXT,metadata TEXT,'
                   'PRIMARY KEY(record_id,key))')
        db.execute('INSERT OR IGNORE INTO lab_versions SELECT id,fingerprint,metadata FROM lab_records')
    else:
        db=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);db.execute('PRAGMA query_only=ON')
    db.row_factory=sqlite3.Row
    return db

def reported(value, prefix=''):
    """Attributed literal fields, never a classifier over free prose."""
    out=[]
    if not isinstance(value,dict):return out
    for k,v in value.items():
        path=prefix+k
        if k in {'status','classification','verdict','veredicto','stage4_admitted','stage5_admitted'} and isinstance(v,(str,bool)):
            out.append({'field':path,'value':v})
        elif isinstance(v,dict) and prefix.count('.')<2:
            out.extend(reported(v,path+'.'))
    return out


def record_content(root, directory, category):
    """One discovery contract shared by refresh and the read-only quality check."""
    docs=[];claims=[];errors=[]
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():continue
        if path.suffix!='.md' and path.name not in JSON_NAMES:continue
        if path.stat().st_size>MAX_TEXT:continue
        raw=path.read_bytes();rel=str(path.relative_to(root));text=raw.decode('utf-8',errors='replace')
        docs.append({'path':rel,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'text':text})
        if path.suffix=='.json':
            try:claims.extend({'source':rel,**x} for x in reported(json.loads(text)))
            except (ValueError,TypeError):errors.append({'path':rel,'error':'invalid JSON retained as text'})
    fingerprint=hashlib.sha256(json.dumps({'category':category,
        'documents':[(d['path'],d['sha256']) for d in docs]},sort_keys=True).encode()).hexdigest()
    return docs,claims,fingerprint,errors


def replace_search(db,root,rid,docs):
    """Reports join the parent record; a refresh never blesses an altered report."""
    texts=[d['text'] for d in docs]
    for row in db.execute('SELECT metadata FROM lab_analyses WHERE record_id=?',(rid,)):
        for doc in json.loads(row[0])['documents']:
            try:
                path=inside(root,doc['path'])
                if doc['bytes']>MAX_TEXT or path.stat().st_size!=doc['bytes']:continue
                raw=path.read_bytes()
                if hashlib.sha256(raw).hexdigest()==doc['sha256']:
                    texts.append(raw.decode('utf-8',errors='replace'))
            except OSError:continue
    text='\n'.join(texts)
    old=db.execute('SELECT text FROM lab_search WHERE record_id=?',(rid,)).fetchall()
    if len(old)==1 and old[0][0]==text:return False
    db.execute('DELETE FROM lab_search WHERE record_id=?',(rid,))
    db.execute('INSERT INTO lab_search VALUES(?,?,?)',(rid,Path(rid).name,text))
    return True


def audit(root=ROOT):
    """Check actual text coverage and bound reports, without writing or loading arrays."""
    root=Path(root).resolve();cfg=config(root);db=connect(root);seen=set();stale=[];empty=[];errors=[]
    try:
        stored={r['id']:dict(r) for r in db.execute('SELECT id,category,fingerprint FROM lab_records')}
        for area in cfg['roots']:
            base=inside(root,area['path'])
            if not base.is_dir():errors.append({'path':area['path'],'error':'missing root'});continue
            for p in sorted(base.iterdir()):
                if p.is_symlink() or not p.is_dir() or p.name.startswith('.'):continue
                rid=str(p.relative_to(root));seen.add(rid)
                docs,_,fingerprint,problems=record_content(root,p,area['category']);errors.extend(problems)
                if not docs:empty.append(rid)
                if rid in stored and (stored[rid]['fingerprint']!=fingerprint or stored[rid]['category']!=area['category']):
                    stale.append(rid)
        analysis_errors=[]
        for row in db.execute('SELECT record_id,metadata FROM lab_analyses'):
            for d in json.loads(row['metadata'])['documents']:
                try:
                    path=inside(root,d['path'])
                    if path.stat().st_size!=d['bytes'] or sha(path)!=d['sha256']:
                        raise ValueError('report hash/size changed')
                except (OSError,ValueError) as exc:
                    analysis_errors.append({'record':row['record_id'],'path':d['path'],'error':str(exc)})
        missing=sorted(set(stored)-seen);new=sorted(seen-set(stored))
        return {'ready':not (stale or missing or new or errors or analysis_errors),
                'records_checked':len(seen),'stale_records':stale,'unindexed_records':new,
                'unavailable_records':missing,'missing_metadata':empty,'errors':errors,
                'analysis_errors':analysis_errors,
                'scope':'Declared direct text and registered report hashes; empty folders are explicit, raw arrays not checked.'}
    finally:db.close()

def refresh(root=ROOT, only=None):
    root=Path(root).resolve();cfg=config(root);start=time.process_time();db=connect(root,True)
    changed=reused=search_updated=0;seen=set();errors=[]
    selected=inside(root,only) if only else None
    try:
        for area in cfg['roots']:
            base=inside(root,area['path'])
            if not base.is_dir():errors.append({'path':area['path'],'error':'missing root'});continue
            for p in sorted(base.iterdir()):
                if p.is_symlink() or not p.is_dir() or p.name.startswith('.'):continue
                if selected and p!=selected:continue
                rid=str(p.relative_to(root));seen.add(rid)
                docs,claims,fingerprint,problems=record_content(root,p,area['category']);errors.extend(problems)
                search_updated+=replace_search(db,root,rid,docs)
                prev=db.execute('SELECT fingerprint FROM lab_records WHERE id=?',(rid,)).fetchone()
                if prev and prev['fingerprint']==fingerprint:reused+=1;continue
                entry=next((d['path'] for n in ('RESULTADOS.md','README.md','DECISION.md') for d in docs if Path(d['path']).name==n),docs[0]['path'] if docs else None)
                meta={'id':rid,'category':area['category'],'entry':entry,'fingerprint':fingerprint,
                      'documents':[{k:v for k,v in d.items() if k!='text'} for d in docs],
                      'reported_fields':claims,'scientific_admission':None,
                      'coverage':'direct Markdown and declared small JSON; no raw-array verification',
                      'missing_metadata':not bool(docs)}
                db.execute('DELETE FROM lab_files WHERE record_id=?',(rid,))
                db.execute('INSERT OR REPLACE INTO lab_records VALUES(?,?,?,?,?)',
                           (rid,area['category'],entry,fingerprint,json.dumps(meta,ensure_ascii=False)))
                db.execute('INSERT OR IGNORE INTO lab_versions VALUES(?,?,?)',
                           (rid,fingerprint,json.dumps(meta,ensure_ascii=False)))
                db.executemany('INSERT INTO lab_files VALUES(?,?,?,?)',[(d['path'],rid,d['sha256'],d['bytes']) for d in docs])
                changed+=1
        if selected and not seen:raise ValueError('Not a direct record in a declared root: '+str(selected))
        # Missing records are retained, marked unavailable by readers. No history deletion.
        result={'changed':changed,'reused':reused,'search_updated':search_updated,
                'records_seen':len(seen),'cpu_s':time.process_time()-start,
                'errors':errors,'selection':only or 'all_declared_roots',
                'scope':'metadata discovery and verified report text; stages unchanged'}
        db.execute('INSERT OR REPLACE INTO lab_meta VALUES(?,?)',('last_refresh',json.dumps(result)))
        db.commit();return result
    finally:db.close()

def get(root,rid,verify=True):
    db=connect(root)
    try:
        row=db.execute('SELECT metadata FROM lab_records WHERE id=?',(rid,)).fetchone()
        analyses=[json.loads(x[0]) for x in db.execute('SELECT metadata FROM lab_analyses WHERE record_id=?',(rid,))]
    finally:db.close()
    if row is None:raise ValueError('Unknown evidence ID: '+rid)
    data=json.loads(row[0]);data['available']=inside(root,rid).is_dir();data['analyses']=analyses
    for a in analyses:
        data['documents'].extend({**d,'analysis_key':a['key'],
            'reuse_note':a.get('reuse_note','')} for d in a['documents'])
    data['metadata_integrity']='not_checked'
    if verify:
        missing=[];changed=[]
        for d in data['documents']:
            p=inside(root,d['path'])
            if not p.is_file():missing.append(d['path'])
            elif sha(p)!=d['sha256']:changed.append(d['path'])
        integrity=('unavailable' if not data['available'] else 'stale' if missing or changed
                   else 'missing_metadata' if not data['documents'] else 'matches_index')
        data.update(metadata_integrity=integrity,
                    missing=missing,changed=changed)
    return data

def register_analysis(root,rid,summary):
    """Workbench completion hook: index verified reports, never run a simulator."""
    refresh(root,only=rid);db=connect(root,True)
    try:
        for step in summary['steps'].values():
            if step['status']!='COMPLETE':raise ValueError('Only completed analysis artifacts can be registered')
            docs=[];out=Path(step['output_dir'])
            for rel,info in step['outputs'].items():
                if Path(rel).suffix!='.md' or info['bytes']>MAX_TEXT:continue
                path=inside(root,out/rel);name=str(path.relative_to(root))
                if sha(path)!=info['sha256']:raise ValueError('Analysis report changed')
                docs.append({'path':name,'sha256':info['sha256'],'bytes':info['bytes']})
                db.execute('INSERT OR REPLACE INTO lab_files VALUES(?,?,?,?)',
                           (name,'analysis:'+step['key'],info['sha256'],info['bytes']))
            row={'key':step['key'],'output_dir':str(out.relative_to(root)),
                 'documents':docs,'reused':step.get('reused',False),'scientific_admission':None}
            db.execute('INSERT OR REPLACE INTO lab_analyses VALUES(?,?,?)',(rid,step['key'],json.dumps(row)))
        meta=json.loads(db.execute('SELECT metadata FROM lab_records WHERE id=?',(rid,)).fetchone()[0])
        docs,_,_,_=record_content(Path(root).resolve(),inside(root,rid),meta['category'])
        replace_search(db,root,rid,docs)
        db.commit()
    finally:db.close()
    return {'status':'indexed','record':rid,'scientific_admission':None}

def search(root,query,limit=12,category=None):
    words=re.findall(r'\w+',unicodedata.normalize('NFKC',query))[:24]
    if not words:return []
    db=connect(root)
    try:
        clause=' AND r.category=?' if category else ''
        parameters=[' OR '.join('"'+w+'"' for w in words)]
        if category:parameters.append(category)
        parameters.append(min(100,max(1,limit)))
        rows=db.execute('SELECT record_id,bm25(lab_search,0,6,1) AS rank FROM lab_search '
                        'JOIN lab_records r ON r.id=lab_search.record_id '
                        'WHERE lab_search MATCH ?'+clause+' ORDER BY rank LIMIT ?',parameters).fetchall()
    finally:db.close()
    return [get(root,r['record_id']) for r in rows]

def document(root,rel):
    db=connect(root)
    try:row=db.execute('SELECT sha256,bytes FROM lab_files WHERE path=?',(rel,)).fetchone()
    finally:db.close()
    if row is None:raise ValueError('Document not in declared evidence index')
    p=inside(root,rel)
    if row['bytes']>MAX_TEXT or sha(p)!=row['sha256']:raise ValueError('Stale document; refresh before reading')
    return p.read_text(errors='replace')

def overview(root=ROOT):
    db=connect(root)
    try:
        rows=db.execute('SELECT id,category,entry,fingerprint FROM lab_records ORDER BY id').fetchall()
        count=db.execute('SELECT count(*) FROM lab_files').fetchone()[0]
        meta=db.execute("SELECT value FROM lab_meta WHERE key='last_refresh'").fetchone()
    finally:db.close()
    cfg=config(root)
    return {'canonical_root':str(root),'records':[dict(r) for r in rows], 'documents':count,
            'last_refresh':json.loads(meta[0]) if meta else None,'current':cfg['current'],
            'cell_recordings':list(cfg.get('cell_recordings',{})),
            'limits':['Text discovery is not scientific validation.','Missing measurements are not zero.',
                      'Per-record access verifies indexed text. Raw arrays are verified by typed analyses.']}

def cell(root,rid,neuron):
    import numpy as np
    cfg=config(root);spec=cfg.get('cell_recordings',{}).get(rid)
    if not spec:raise ValueError('No typed cell adapter registered for this recording')
    traces=[]
    for arm,rel in spec['arms'].items():
        p=inside(root,rel);h=sha(p)
        if spec.get('source_sha256',{}).get(arm)!=h:
            raise ValueError('Cell recording changed or lacks its registered hash: '+arm)
        with np.load(p,allow_pickle=False) as z:
            t=z['time_ns'];ids=z['ids'];q=z['q']
            if neuron not in ids:ids=z['ORN_ids'];q=z['ORN_q']
            if len(np.unique(ids))!=len(ids) or q.shape!=(len(t),len(ids)):
                raise ValueError('Cell identity/shape mismatch')
            hit=np.flatnonzero(ids==neuron)
            if not len(hit):
                traces.append({'arm':arm,'available':False,'reason':'not recorded in this panel'});continue
            y=q[:,int(hit[0])]
            if t.dtype.kind not in 'iu' or np.any(np.diff(t)<=0) or not np.isfinite(y).all():
                raise ValueError('Invalid clock or nonfinite observation')
            if len(t)>4000:raise ValueError('Panel exceeds interactive sample budget')
            traces.append({'arm':arm,'available':True,'time_ns':t.tolist(),'values':y.tolist(),
                           'source':rel,'sha256':h})
    return {'id':int(neuron),'record':rid,'unit':spec['unit'],'clock':spec['clock'],
            'preparation':spec['preparation'],'traces':traces,
            'limit':'Committed dimensionless model state q; not voltage, spikes or causal attribution.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('refresh');sub.add_parser('status');sub.add_parser('check')
    s=sub.add_parser('find');s.add_argument('query',nargs='+');s.add_argument('--limit',type=int,default=12)
    s.add_argument('--category',choices=['campaign','study','engine','dataset','evidence','tool'])
    s=sub.add_parser('show');s.add_argument('id')
    s=sub.add_parser('close');s.add_argument('path')
    s=sub.add_parser('cell');s.add_argument('id');s.add_argument('neuron',type=int)
    s=sub.add_parser('serve');s.add_argument('--port',type=int,default=8765)
    a=p.parse_args()
    if a.action=='serve':
        from brain_atlas_server import serve
        serve(ROOT,a.port);return
    if a.action=='refresh':out=refresh()
    elif a.action=='close':out=refresh(only=a.path)
    elif a.action=='find':out=search(ROOT,' '.join(a.query),a.limit,a.category)
    elif a.action=='check':out=audit()
    elif a.action=='show':out=get(ROOT,a.id)
    elif a.action=='cell':out=cell(ROOT,a.id,a.neuron)
    else:
        out=overview();out['records_count']=len(out.pop('records'))
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
    if a.action=='check' and not out['ready']:raise SystemExit(2)

if __name__=='__main__':main()
