"""Portable reconstruction from exact selected arrays, with no local project imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from chatgpt_bounds_original import cota_monotona
from reader_bounds import monotone_bound


def need(ok, message):
    if not ok:
        raise ValueError(message)


def ahash(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def close(a, b, label):
    need(np.isfinite(a) and np.isfinite(b) and abs(a-b) <= 1e-12, 'Numeric mismatch: '+label)


def verify(root):
    manifest = json.loads((root/'DATA.json').read_text())
    plan = json.loads((root/'PLAN.json').read_text())
    recorded = json.loads((root/'screen_01/RESULT.json').read_text())
    cal = json.loads((root/'screen_01/CALIBRATION.json').read_text())
    data = {}
    for name, row in manifest['traces'].items():
        need(row['original_sha256'] == plan['inputs'][name]['sha256'], 'Wrong original source')
        path = root/row['path']
        need(hashlib.sha256(path.read_bytes()).hexdigest() == row['subset_sha256'], 'Subset bytes changed')
        with np.load(path, allow_pickle=False) as z:
            need(set(z.files) == set(row['arrays']), 'Subset members changed')
            data[name] = {k:z[k].copy() for k in z.files}
        for k, x in data[name].items():
            need(ahash(x) == row['arrays'][k], 'Array identity changed')
            need(np.isfinite(x).all(), 'Nonfinite array')
    sham = data['sham']
    need(np.all(sham['sensores_usados'][40:] == 0), 'Calibration stimulus changed')
    d = sham['DN_q_usada'][40:240,2]-sham['DN_q_usada'][40:240,3]
    mu = float(d.mean()); scale = float(np.sqrt(np.mean((d-mu)**2)))
    close(mu, cal['offset_q'], 'rest offset'); close(scale, cal['scale_q'], 'rest scale')
    need(scale > 64*np.finfo(float).eps, 'Unresolved neutral scale')
    metrics={}; signals={}; bounds={}
    for name, z in data.items():
        q=z['DN_q_usada'][40:]; base=z['DN_baseline'][40:]
        need(q.shape[1] == 4 and base.shape == q.shape and np.all((q>=0)&(q<=1)), 'Invalid q domain')
        ticks=np.diff(z['CNS_time_ns'][39:]);need(np.all(ticks==1000000), 'Clock changed')
        x=(q[:,2]-base[:,2])-(q[:,3]-base[:,3]); signals[name]=(x,ticks)
        prep=z['DN_q_actual'][39]
        arrays=dict(parent=np.array([5*math.tanh(250*float(v)) for v in x]),
                    candidate=np.array([5*math.tanh((float(a)-float(b)-mu)/scale) for a,b in q[:,2:4]]),
                    reset_diagnostic=np.array([5*math.tanh(250*((float(a)-prep[2])-(float(b)-prep[3]))) for a,b in q[:,2:4]]))
        need(np.max(abs(np.deg2rad(arrays['parent'])-z['command_yaw_rate_rad_s'][40:])) < 1e-12, 'Archived command changed')
        sl=slice(200,400) if name=='sham' else slice(None)
        metrics[name]={}
        for key, values in arrays.items():
            v=values[sl]
            m=dict(net_command_deg=.001*math.fsum(v),absolute_command_deg=.001*math.fsum(abs(v)),
                   positive_fraction=float(np.mean(v>0)),near_ceiling_fraction=float(np.mean(abs(v)>4.5)))
            for k, value in m.items():close(value, recorded['records'][name][key][k], name+'/'+key+'/'+k)
            metrics[name][key]=m
        expected=recorded['records'][name]
        need(bool(np.any(np.ptp(q[:,:2],axis=0))) == expected['neural_forward_channels_vary'], 'Forward-channel flag changed')
        need([float(np.min(z['command_forward_mm_s'][40:])),float(np.max(z['command_forward_mm_s'][40:]))] == expected['forward_command_range_mm_s'], 'Forward command changed')
    net=lambda n:metrics[n]['candidate']['net_command_deg']
    gates=dict(heldout_rest_nonregression=abs(net('sham')) <= abs(metrics['sham']['parent']['net_command_deg'])+plan['screen']['rest_net_nonregression_margin_deg'],
               static_left_direction=net('odor_left')>0, static_right_direction=net('odor_right')<0,
               spatial_plus_direction=net('spatial_plus')>0, spatial_minus_direction=net('spatial_minus')<0,
               static_bilateral_separation=net('odor_left')-net('odor_right')>=plan['screen']['minimum_pair_directional_separation_deg'],
               spatial_bilateral_separation=net('spatial_plus')-net('spatial_minus')>=plan['screen']['minimum_pair_directional_separation_deg'])
    classification='PROMETEDOR_NO_CONFIRMADO' if all(gates.values()) else 'DESCARTADO'
    need(gates == recorded['gates'] and classification == recorded['classification'], 'Readout verdict changed')
    brec=json.loads((root/'bounds_01/RESULT.json').read_text())
    for name in ('long_identity','long_equal','shared_equal_minus_original'):
        if name.startswith('shared'):
            x=np.r_[signals['long_equal'][0],signals['long_identity'][0]]
            w=np.r_[signals['long_equal'][1],-signals['long_identity'][1]]
            target=brec[name]
        else:
            x,w=signals[name];target=brec['individual'][name]
        b,_=monotone_bound(x,w); external=cota_monotona(x,w*1e-9)
        close(b['lower_deg'], target['lower_deg'],name+'/lower');close(b['upper_deg'],target['upper_deg'],name+'/upper')
        close(b['lower_deg'],external['min_grados_mando'],name+'/donor lower')
        close(b['upper_deg'],external['max_grados_mando'],name+'/donor upper')
        bounds[name]={'lower_deg':b['lower_deg'],'upper_deg':b['upper_deg']}
    pop=json.loads((root/'population_01/RESULT.json').read_text())
    arrays={}
    for name,row in manifest['published'].items():
        path=root/row['path']
        need(hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],'Published checkpoint changed')
        with np.load(path,allow_pickle=False) as z: arrays[name]=z['array_0']
        need(arrays[name].shape==(166700,) and np.isfinite(arrays[name]).all(),'Invalid full published rates')
    need(np.array_equal(arrays['identity_01/prepared_state'],arrays['no_contrast_01/prepared_state']),'Published preparation changed')
    zero_both=0
    for row in pop['rows']:
        ix=row['row_index'];need(ix==row['node_index']==row['matrix_node_index'],'Index fields disagree')
        close(float(arrays['identity_01/prepared_state'][ix]),row['prepared_model_hz'],'prepared rate')
        a=float(arrays['identity_01/final_state'][ix]);b=float(arrays['no_contrast_01/final_state'][ix])
        close(a,row['identity_final_model_hz'],'identity rate');close(b,row['equal_final_model_hz'],'equal rate')
        close(a-b,row['identity_minus_equal_model_hz'],'rate contrast')
        zero_both += a==0 and b==0
    need(not any(recorded[k] or pop[k] for k in ('stage4_admitted','stage5_admitted')), 'Unsupported admission flag')
    return dict(readout_classification=classification,rest_candidate_deg=net('sham'),gates=gates,
                bounds=bounds,population_cells=len(pop['rows']),zero_both_final_cells=zero_both,
                scope='Recomputed command screen, algebraic bounds and published endpoints only; no organism or physiological calibration')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();need(not args.out.exists(),'Unique output required')
    result=verify(args.root)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':
    main()
