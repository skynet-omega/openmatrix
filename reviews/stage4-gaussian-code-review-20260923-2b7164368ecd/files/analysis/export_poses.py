"""Small, lossless numeric subset for external analysis; no simulation."""
from pathlib import Path
import csv,hashlib,json,resource,time
import numpy as np
from pose_gaussian_sensitivity import HERE,ROOT,NATIVE,TRACE_HASHES,RAW_GEOMETRY,RAW_GEOMETRY_HASH,sha,need

def main():
 start=time.perf_counter();cpu=time.process_time()
 need(sha(RAW_GEOMETRY)==RAW_GEOMETRY_HASH,'Geometry changed')
 scale=float(json.loads(RAW_GEOMETRY.read_text())['body_native_units_per_mm'])
 need(scale>0,'Bad body scale')
 out=HERE/'portable_poses_01';out.mkdir(exist_ok=False)
 columns=['arm','phase','step','root_x_mm','root_y_mm','antenna_left_x_mm','antenna_left_y_mm',
          'antenna_right_x_mm','antenna_right_y_mm','actual_binary_field_left','actual_binary_field_right',
          'actually_used_sensor_left','actually_used_sensor_right','command_yaw_rad_s','yaw_delta_deg']
 rows=0;input_hashes={}
 with (out/'POSES_PN629.csv').open('x',newline='') as f:
  writer=csv.writer(f);writer.writerow(columns)
  for arm,digest in TRACE_HASHES.items():
   p=NATIVE/('full_'+arm+'_01')/'traces.npz'
   need(sha(p)==digest,'Input trace changed: '+arm)
   input_hashes[arm]=digest
   with np.load(p,allow_pickle=False) as z:
    a=z['antenas_mm'];q=z['qpos'];field=z['concentracion_campo'];used=z['sensores_usados']
    phase=z['fase'];step=z['paso'];command=z['command_yaw_rate_rad_s'];yaw=z['yaw_delta_deg']
    need(len(a)==440 and a.shape[1:]==(2,3),'Bad trace')
    for i in range(39,440):
     vals=[q[i,0]/scale,q[i,1]/scale,a[i,0,0],a[i,0,1],a[i,1,0],a[i,1,1],
           field[i,0],field[i,1],used[i,0],used[i,1],command[i],yaw[i]]
     need(np.isfinite(vals).all(),'Nonfinite portable row')
     writer.writerow([arm,str(phase[i]),int(step[i]),*[repr(float(x)) for x in vals]])
     rows+=1
 need(rows==1604,'Rows missing')
 wall=time.perf_counter()-start;cpu_s=time.process_time()-cpu;mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 budget=json.loads((HERE/'EXPORT_PLAN.json').read_text())['budget']
 need(wall<=budget['wall_s_max'] and cpu_s<=budget['cpu_s_max'] and mem<=budget['ram_MiB_max'],'Budget exceeded')
 receipt={'schema':'stage4_chatgpt_portable_pose_subset_v1','rows':rows,'columns':columns,'input_trace_sha256':input_hashes,
          'raw_geometry_sha256':RAW_GEOMETRY_HASH,'csv_sha256':sha(out/'POSES_PN629.csv'),
          'export_code_sha256':sha(__file__),'plan_sha256':sha(HERE/'EXPORT_PLAN.json'),
          'wall_s':wall,'cpu_s':cpu_s,'peak_ram_MiB':mem,
          'scope':'Recorded trajectories from binary-field runs only; no Gaussian CNS or body outcome.'}
 (out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'rows':rows,'csv_sha256':receipt['csv_sha256'],'wall_s':wall}))
if __name__=='__main__':main()
