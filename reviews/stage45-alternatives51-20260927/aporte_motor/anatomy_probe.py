"""Read-only anatomical JO-C/E projection screen; no model/CUDA imports."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import resource
resource.setrlimit(resource.RLIMIT_CPU,(50,55))
import hashlib,json,re,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

H=Path(__file__).resolve().parent
D=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10')
def need(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 start=time.process_time()
 need(not (H/'ANATOMY.json').exists(),'Preserve prior result')
 cols=['bodyId','node_index','type','instance','rootSide','somaSide','entryNerve',
       'superclass','class','subclass','synonyms','hemibrainType','flywireType','nt_consensus_nt']
 n=pd.read_parquet(D/'nodes.parquet',columns=cols).fillna('unknown')
 ids=np.load(D/'node_ids.npy',allow_pickle=False)
 need(np.array_equal(ids,n.bodyId.to_numpy()),'Canonical row identity')
 need(np.array_equal(n.node_index.to_numpy(),np.arange(len(n))),'Canonical row numbering')
 c=sparse.load_npz(D/'counts_pre_post.npz').tocsr()
 need(c.shape==(len(n),len(n)) and c.has_canonical_format,'Counts schema')
 need(c.dtype.kind in 'iu' and np.all(c.data>=0),'Unsigned synaptic counts')
 src=n[n.type.str.match(r'^JO-[CE]')].copy()
 need(len(src)==335 and src.rootSide.isin(['L','R']).all() and (src.entryNerve=='AN').all(),'JO-C/E source identities')
 rows=src.node_index.to_numpy(np.int64)
 sub=c[rows].tocoo()
 # Verify declared orientation against separately stored node connectivity.
 con=pd.read_parquet(D/'node_connectivity.parquet').set_index('node_index')
 out_count=np.asarray(c[rows].sum(axis=1)).ravel()
 need(np.array_equal(out_count,con.loc[rows,'internal_output_synapses'].to_numpy()),'Pre/post orientation or counts mismatch')
 src['internal_output_synapses']=out_count
 src['JO_family']=src.type.str.slice(0,4)
 src.to_csv(H/'JO_CE_sources.csv',index=False)
 e=pd.DataFrame({'pre_row':rows[sub.row],'post_row':sub.col,'count':sub.data.astype(np.int64)})
 for name in ['bodyId','type','rootSide','somaSide','nt_consensus_nt']:
  e['pre_'+name]=n[name].to_numpy()[e.pre_row]
  e['post_'+name]=n[name].to_numpy()[e.post_row]
 # Keep all actual outgoing edges, not just the strongest apparent relay.
 e.to_csv(H/'JO_CE_edges.csv',index=False)
 agg=e.groupby(['post_row','post_bodyId','post_type','post_rootSide','post_somaSide'],dropna=False)['count'].sum().reset_index(name='JO_CE_synapses')
 agg['all_internal_input_synapses']=con.loc[agg.post_row,'internal_input_synapses'].to_numpy()
 agg['JO_CE_fraction']=agg.JO_CE_synapses/agg.all_internal_input_synapses
 side=e.groupby(['post_row','pre_rootSide'])['count'].sum().unstack(fill_value=0)
 agg['from_left_antenna']=side.reindex(agg.post_row).get('L',pd.Series(0,index=agg.post_row)).to_numpy()
 agg['from_right_antenna']=side.reindex(agg.post_row).get('R',pd.Series(0,index=agg.post_row)).to_numpy()
 agg.sort_values(['JO_CE_synapses','post_bodyId'],ascending=[False,True],inplace=True)
 agg.to_csv(H/'all_direct_targets.csv',index=False)
 text=n[['type','synonyms','hemibrainType','flywireType','instance']].astype(str).agg(' | '.join,axis=1)
 relevant=text.str.contains(r'WED|AMMC|\bAPN[123]\b|wind',case=False,regex=True)
 ann=n.loc[relevant].copy();ann.to_csv(H/'relay_candidate_annotations.csv',index=False)
 relay=agg[agg.post_row.isin(ann.node_index)].copy()
 relay.to_csv(H/'direct_relay_candidates.csv',index=False)
 # A two-hop wedge screen records anatomical counts only, not causal gains.
 relay_rows=relay.post_row.to_numpy(np.int64)
 dst_rows=n.index[n.type.str.startswith('WEDPN')].to_numpy(np.int64)
 two=c[relay_rows][:,dst_rows].tocoo()
 relay_edges=pd.DataFrame({'pre_row':relay_rows[two.row],'post_row':dst_rows[two.col],'count':two.data.astype(np.int64)})
 for name in ['bodyId','type','rootSide','somaSide']:
  relay_edges['pre_'+name]=n[name].to_numpy()[relay_edges.pre_row]
  relay_edges['post_'+name]=n[name].to_numpy()[relay_edges.post_row]
 relay_edges.to_csv(H/'relay_to_WEDPN_edges.csv',index=False)
 np.savez_compressed(H/'JO_anatomy_arrays.npz',source_ids=src.bodyId.to_numpy(np.int64),source_rows=rows,
  source_side=src.rootSide.to_numpy(dtype='U1'),source_type=src.type.to_numpy(dtype='U'),
  pre_rows=e.pre_row.to_numpy(np.int64),post_rows=e.post_row.to_numpy(np.int64),counts=e['count'].to_numpy(np.int64),
  relay_rows=relay_rows,WEDPN_rows=dst_rows)
 aliases=n[text.str.contains(r'\bAPN[123]\b|WPN|AMMC-B1|AMMC-A1',case=False,regex=True)]
 result=dict(scope='Unsigned structural synapse counts, not effective weights, activity or identified physiological tuning',
  neurons=len(n),graph_edges=int(c.nnz),sources=len(src),source_synapses=int(out_count.sum()),
  source_family_side_counts=src.groupby(['JO_family','rootSide']).size().to_dict(),
  direct_edges=len(e),direct_target_neurons=len(agg),direct_relay_candidates=len(relay),
  relay_candidate_synapses=int(relay.JO_CE_synapses.sum()),relay_to_WEDPN_edges=len(relay_edges),
  relay_candidate_types=relay.groupby('post_type').JO_CE_synapses.sum().sort_values(ascending=False).to_dict(),
  explicit_alias_matches=aliases[['bodyId','type','synonyms','hemibrainType','flywireType']].to_dict(orient='records'),
  orientation_verified_against_node_connectivity=True,
  input_hashes={f:sha(D/f) for f in ['nodes.parquet','node_ids.npy','counts_pre_post.npz','node_connectivity.parquet','provenance.json']},
  CPU_s=time.process_time()-start,process_CPU_total_s=time.process_time(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
  CNS_steps=0,GPU_calls=0,download_bytes=0)
 result['source_family_side_counts']={f'{a}/{b}':int(v) for (a,b),v in result['source_family_side_counts'].items()}
 (H/'ANATOMY.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('input_hashes','explicit_alias_matches')},ensure_ascii=False))
 print(json.dumps({'explicit_alias_matches':result['explicit_alias_matches'][:12]}))
if __name__=='__main__':main()
