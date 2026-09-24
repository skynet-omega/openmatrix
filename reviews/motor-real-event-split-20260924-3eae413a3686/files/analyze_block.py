"""Recompute work floors and operator facts from the captured real block."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
DATA=HERE/'capture_01'


def main():
    out=HERE/'BLOCK_ANALYSIS_01.json'
    if out.exists():raise FileExistsError(out)
    clock=np.load(DATA/'trace_clock.npz',allow_pickle=False)
    ev=np.load(DATA/'block_events.npz',allow_pickle=False)
    meta=json.loads((DATA/'block_events.json').read_text())
    def field(name):return ev[meta[name]['__array__']]
    times=field('times');duration=meta['duration_ns']*1e-9
    unique=np.unique(times[(times>0)&(times<duration)])
    queries=len(clock['query_s']);intervals=len(unique)+1
    mapping=json.loads((DATA/'EFFECTIVE_ARRAY_PATHS.json').read_text())
    with np.load(DATA/'effective_gpu_arrays.npz',allow_pickle=False) as arrays:
        cols=arrays[mapping['cuda/indices']]
        ptr=arrays[mapping['cuda/indptr']]
        mask=np.zeros(len(ptr)-1,dtype=bool);mask[field('rows')]=True
        port_edges=int(np.count_nonzero(mask[cols]))
    with np.load(DATA/'trace_rate.npz',allow_pickle=False) as arrays:
        rates=arrays['values']
        rate_stats={'min':float(rates.min()),'max':float(rates.max()),
                    'max_times_duration':float(rates.max()*duration)}
    result={'schema':'real_block_work_floor_v1',
            'queries':queries,'trial_count':queries//6,
            'internal_event_times_s':unique.tolist(),'event_free_intervals':intervals,
            'global_per_interval_speedup_ceiling':queries/intervals,
            'gate_speedup_min':10.,'global_per_interval_can_pass_work_gate':queries/intervals>=10.,
            'scope_of_ceiling':'Only algorithms requiring at least one full CSR per event-free interval. Does not rule out local event forcing or sparse corrections.',
            'port_sources':len(field('rows')),'distinct_tau_q':len(np.unique(field('tau'))),
            'tau_q_min_max_s':[float(field('tau').min()),float(field('tau').max())],
            'edges':len(cols),'edges_from_ports':port_edges,'port_edge_fraction':port_edges/len(cols),
            'full_divided_by_port_edge_work':len(cols)/port_edges,
            'recurrent_edge_fraction':1.-port_edges/len(cols),
            'relaxation_rates':rate_stats,
            'stiffness_inference':'Diagonal relaxation only; does not bound the full coupled Jacobian.',
            'classification':'A_GLOBAL_PER_CUT_DISCARDED_FOR_THIS_WORK_GATE',
            'whole_organism_speedup_measured':False,'stage_admission':False}
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
