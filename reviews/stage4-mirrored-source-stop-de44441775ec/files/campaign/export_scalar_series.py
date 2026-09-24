"""Portable observed scalar rows for external audit; source NPZ remains authoritative."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from verify_mirror import check_trace

HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    identities=(('causal_cuda','plus','native_plus_01'),
                ('causal_cuda','minus','native_minus_01'),
                ('reference_cuda','plus','reference_plus_01'))
    path=HERE/'SCALAR_SERIES_01.csv'
    fields=('profile','arm','trial_ms','field_L','field_R','sensor_used_L','sensor_used_R',
            'DN_q_usada_L','DN_q_usada_R','DN_baseline_L','DN_baseline_R',
            'command_deg_s','yaw_deg')
    manifest={'schema':'stage4_scalar_series_export_v1','source_sha256':sha(__file__),
              'scope':'Raw observed samples; reference plus ends at 391/400 ms; no full numerical or navigation admission.',
              'arms':{}}
    with path.open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(fields)
        for profile,arm,folder in identities:
            source=HERE/folder/'traces.npz'
            with np.load(source,allow_pickle=False) as z:t={k:z[k] for k in z.files}
            metric=check_trace(t,arm)
            n=metric['steps']
            for j in range(n):
                i=j+40
                writer.writerow([profile,arm,j+1,*map(float,t['concentracion_campo'][i]),
                                 *map(float,t['sensores_usados'][i,:2]),
                                 *map(float,t['DN_q_usada'][i,2:4]),
                                 *map(float,t['DN_baseline'][i,2:4]),
                                 float(np.rad2deg(t['command_yaw_rate_rad_s'][i])),
                                 float(t['yaw_delta_deg'][i])])
            manifest['arms'][profile+'/'+arm]={'folder':folder,'trace_sha256':sha(source),
                                                'trial_rows':n,'complete_400ms':metric['complete']}
    manifest['csv_sha256']=sha(path)
    with (HERE/'SCALAR_SERIES_MANIFEST_01.json').open('x') as f:
        json.dump(manifest,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(manifest))

if __name__=='__main__':main()
