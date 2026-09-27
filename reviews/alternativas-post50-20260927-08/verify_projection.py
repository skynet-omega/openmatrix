"""Recompute the portable endpoint projection and median diagnostic, not CNS."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
H=Path(__file__).resolve().parent
PAIRS={'DNa01':(10442,10760),'DNa02':(523769,10360),'DNb05':(10118,10065),'DNb06':(10888,11067),'DNg13':(11074,512006)}
ARMS=['sham','dm1','profile','permuted']
def need(x,m):
    if not x:raise ValueError(m)
def compute(data):
    ids=data['body_ids'];types=data['types'];sides=data['sides'];rows=data['rows']
    need(len(ids)==10 and len(set(map(int,ids)))==10,'Identity count')
    need(np.all(np.diff(rows)>0),'Ordered original rows')
    for a in ARMS:need(data[a].shape==(10,) and np.isfinite(data[a]).all(),'Finite data')
    table=[]
    for t,(left,right) in PAIRS.items():
        il=np.flatnonzero(ids==left);ir=np.flatnonzero(ids==right)
        need(len(il)==len(ir)==1,'Missing fixed biological identity')
        l,r=int(il[0]),int(ir[0]);need(types[l]==types[r]==t and sides[l]=='L' and sides[r]=='R','Identity/side/type')
        q={a:[float(data[a][l]),float(data[a][r])] for a in ARMS}
        delta={a:(q[a][0]-q[a][1])-(q['sham'][0]-q['sham'][1]) for a in ARMS if a!='sham'}
        table.append(dict(type=t,ids_LR=[left,right],q_LR=q,L_minus_R_delta_from_sham=delta))
    return table
def main():
    manifest=H/'MANIFEST.json'
    if manifest.exists():
        for name,entry in json.loads(manifest.read_text())['files'].items():
            p=H/name
            need(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],'Manifest mismatch: '+name)
    with np.load(H/'steering_endpoints_projection.npz',allow_pickle=False) as z:data={k:z[k] for k in z}
    table=compute(data);reported=json.loads((H/'STEERING_ENDPOINTS.json').read_text())
    need(table==reported['table'],'Recomputed endpoint table differs')
    expected=json.loads((H/'MEDIAN_READER_ENDPOINT_CHECK.json').read_text())['values']
    medians={}
    for a in ARMS[1:]:
        v=[-r['L_minus_R_delta_from_sham'][a]*(-1 if r['type']=='DNb06' else 1) for r in table]
        medians[a]=dict(sign_aligned_R_minus_L_deltas=v,median=float(np.median(v)),historical_DNb05_delta=v[2])
    need(medians==expected,'Median diagnostic differs')
    rejected=[]
    for mutation in ['identity','value','nonfinite']:
        d={k:v.copy() for k,v in data.items()}
        if mutation=='identity':d['body_ids'][0]=0
        elif mutation=='value':d['profile'][0]+=0.01
        else:d['sham'][0]=np.nan
        try:need(compute(d)==table,'Altered values')
        except ValueError:rejected.append(mutation)
    need(len(rejected)==3,'Corruption escaped')
    print(json.dumps(dict(endpoint_table_recomputed=True,median_recomputed=True,corruptions_rejected=rejected,new_neural_ms=0,scope='Projected recorded q only; source snapshot integrity was checked during extraction. No raw CNS replay.')))
if __name__=='__main__':main()
