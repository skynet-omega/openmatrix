"""Offline, observation-aware sensory comparison. Never loads the simulator.

--extract reads records45 and projects only the observations used here.
Without it, the same calculation runs from the portable projection.
Study means are a representation contrast, not raw trials or a PN validation set.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from acquire_door import rows, need

HERE = Path(__file__).resolve().parent
OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
ASTRA = Path('/home/daroch/AXIOMA_ASTRA')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def extract(folder):
    import pandas as pd
    folder = Path(folder)
    campaign = ASTRA / 'campanas/iniciacion_olfativa_20260926_45'
    prepared = ASTRA / 'campanas/etapa45_navigation_wind_20260925_40/navigation_minus_filtered_wind_03/prepared_state'
    origin = {}
    def record(p):
        origin[str(p)] = dict(sha256=sha(p), bytes=p.stat().st_size)
    meta_path = prepared / 'session.json'
    meta = json.loads(meta_path.read_text())
    record(meta_path)
    h = meta['hybrid']
    with np.load(prepared / 'session.npz', allow_pickle=False) as z:
        def arr(v):
            return z[v['__array__']].copy()
        syn = h['orn_pn_synaptic_manifest']
        ids = arr(syn['source_orn_ids']).astype(np.int64)
        caps = arr(syn['source_rmax_hz']).astype(float)
        filter_ids = arr(h['olfactory_endogenous_manifest']['source_ids']).astype(np.int64)
    record(prepared / 'session.npz')
    need(len(ids) == len(caps) == 74 and len(set(ids)) == 74, 'ORN identities/capacities')
    cap_by_id = dict(zip(ids.tolist(), caps.tolist()))
    filter_caps = np.array([cap_by_id[int(v)] for v in filter_ids])
    nodes_path = OLD / 'data/male_v10/nodes.parquet'
    nodes = pd.read_parquet(nodes_path).sort_values('node_index')
    record(nodes_path)
    by_id = nodes.set_index('bodyId')
    need(set(by_id.loc[ids, 'type']) == {'ORN_DM1'}, 'Wrong receptor population')
    anatomy = {}
    database = json.loads((HERE / 'DOOR_EXTRACTION.json').read_text())
    for r in database['profile']:
        need(len(r['glomeruli']) == 1, 'Ambiguous receptor mapping')
        typename = 'ORN_' + r['glomeruli'][0]
        group = nodes[nodes.type == typename]
        need(len(group) > 0, 'Absent anatomical population')
        anatomy[r['receptor']] = dict(type=typename, n=len(group),
            ids=group.bodyId.astype(int).tolist(), sides=group.rootSide.fillna('unknown').value_counts().to_dict())
    par = h['pn_online_manifest']['orn_peripheral_terminal']['parameters']
    constants = dict(odor_drive=meta['config']['odor_drive'], peripheral=par,
        synaptic=syn['parameters'], general_outputs_enabled=h['pn_online_manifest']['general_outputs']['enabled'],
        fine_PN_id=10208, legacy_PN_ids=[10208, 10176],
        filter_ids=filter_ids.tolist(), filter_caps=filter_caps.tolist(), anatomy=anatomy,
        scope='ORN caps convert q to an internal nominal proxy, not independently measured Hz.')
    need(constants['general_outputs_enabled'] is False, 'Different PN interface')
    arrays = {}
    for arm in ('sham', 'odor'):
        arm_ids_path = campaign / arm / 'OBSERVED_IDS.npz'
        with np.load(arm_ids_path, allow_pickle=False) as z:
            for side in ('L', 'R'):
                a = z['ORN_' + side + '_ids'].astype(np.int64)
                need(set(by_id.loc[a, 'rootSide']) == {side}, 'Recorded side mismatch')
                arrays[arm + '__caps_' + side] = np.array([cap_by_id[int(i)] for i in a])
                arrays[arm + '__ids_' + side] = a.copy()
        record(arm_ids_path)
        fields = ('paso', 'sensores_usados', 'ORN_q_L', 'ORN_q_R', 'ORN_filters',
                  'PN_q_legacy', 'PN_gamma_nS', 'PN_additional_nS')
        collected = {k: [] for k in fields}
        blocks = sorted((campaign / arm / 'blocks').glob('*ms/traces.npz'))
        need(len(blocks) == 40, 'Incomplete arm')
        for file in blocks:
            with np.load(file, allow_pickle=False) as z:
                for k in fields:
                    collected[k].append(z[k].copy())
            record(file)
        for k, values in collected.items():
            arrays[arm + '__' + k] = np.concatenate(values)
    np.savez_compressed(folder / 'OBSERVATIONS.npz', **arrays)
    save(folder / 'CONTRACT.json', constants)
    save(folder / 'PROVENANCE.json', dict(sources=origin, projection_sha256=sha(folder / 'OBSERVATIONS.npz'),
        scope='Projection of recorded endpoints45, not a restart, consumer-stage witness, or new biological trial.'))

def calculate(folder):
    folder = Path(folder)
    c = json.loads((folder / 'CONTRACT.json').read_text())
    provenance = json.loads((folder / 'PROVENANCE.json').read_text())
    need(sha(folder / 'OBSERVATIONS.npz') == provenance['projection_sha256'], 'Projection changed')
    with np.load(folder / 'OBSERVATIONS.npz', allow_pickle=False) as z:
        a = {arm: {k.split('__', 1)[1]: z[k].copy() for k in z.files if k.startswith(arm+'__')}
             for arm in ('sham', 'odor')}
    source = json.loads((folder / 'DOOR_EXTRACTION.json').read_text())
    # Re-extract source values; never accept the previously summarized numbers alone.
    profile = []
    for r in source['profile']:
        path = folder / r['source']
        record = next(v for v in source['records'] if v['path'] == r['source'])
        need(sha(path) == record['sha256'], 'Biological source changed')
        tab = rows(path)
        basal = next(x for x in tab if x['Name'] == 'sfr')['Bruyne.2001.WT']
        odor = next(x for x in tab if x['Name'] == '1-hexanol')['Bruyne.2001.WT']
        need(float(basal) == r['baseline_Hz'] and float(odor) == r['increment_Hz'], 'Misread source means')
        profile.append(dict(r, anatomy=c['anatomy'][r['receptor']]))
    info = next(x for x in rows(folder / 'primary_door/door_dataset_info.csv') if x['dataset'] == 'Bruyne.2001.WT')
    need(info['SFR.substracted'] == 'yes' and info['solvents.subtracted'] == 'no', 'Different measurement convention')
    p = c['peripheral']
    f, s = c['synaptic']['fast'], c['synaptic']['slow']
    fast_area, slow_area = f['k_ns_per_spike'] * f['tau_g_s'], s['k_ns_per_spike'] * s['tau_g_s']
    for arm, d in a.items():
        need(np.array_equal(d['paso'], np.arange(1, 4001)), 'Time sequence differs')
        for k, v in d.items():
            need(np.isfinite(v).all(), 'Nonfinite ' + k)
        need(d['sensores_usados'].shape == (4000, 3), 'Sensor axes differ')
        # The first consumed input belongs to the common inherited prepared state.
        expected = np.zeros((3999, 3))
        if arm == 'odor':
            expected[999:2999, :2] = .5
        need(np.array_equal(d['sensores_usados'][1:], expected), 'Consumed protocol differs')
        u = np.clip(d['sensores_usados'][:, :2]*c['odor_drive']/p['world_drive_anchor'], 0, 1)
        d['peripheral_nominal'] = p['basal_hz'] + p['increment_hz']*u
        for side in ('L', 'R'):
            d['terminal_' + side] = np.mean(d['ORN_q_' + side]*d['caps_' + side], axis=1)
        filters = d['ORN_filters']
        need(filters.shape == (4000, 4, 74), 'Filter layout differs')
        z = (fast_area*filters[:, 2] + slow_area*filters[:, 3])/(fast_area+slow_area)
        for side in ('L', 'R'):
            select = np.isin(c['filter_ids'], d['ids_' + side])
            need(select.sum() == len(d['ids_' + side]), 'Filter identity mismatch')
            d['bridge_' + side] = np.mean(z[:, select]*np.asarray(c['filter_caps'])[select], axis=1)
        d['gamma_mean'] = d['PN_gamma_nS'].mean(axis=1)
        d['additional_mean'] = d['PN_additional_nS'].mean(axis=1)
    statistics = {}
    for name, sl in [('rest_501_1000ms', slice(500,1000)), ('ON_1001_3000ms',slice(1000,3000)),
                     ('late_ON_2501_3000ms',slice(2500,3000)), ('OFF_3001_4000ms',slice(3000,4000))]:
        statistics[name] = {}
        for key in ('terminal_L', 'terminal_R', 'bridge_L', 'bridge_R', 'gamma_mean', 'additional_mean'):
            s0, o0 = float(a['sham'][key][sl].mean()), float(a['odor'][key][sl].mean())
            statistics[name][key] = dict(sham=s0, odor=o0, difference=o0-s0,
                                         odor_to_sham=o0/s0 if s0 != 0 else None)
        statistics[name]['legacy_PN_difference_10208_10176'] = (a['odor']['PN_q_legacy'][sl] - a['sham']['PN_q_legacy'][sl]).mean(axis=0).tolist()
    result = dict(status='REPRESENTATION_GAP_IDENTIFIED_TRANSFER_NOT_CALIBRATED',
        simulated_neural_ms=0, fitted_parameters=0, biological_independent_replicates=0,
        source=dict(study=info['study'], doi=info['DOI'], concentration_preparation=info['concentration'],
                    solvent=info['solvents'], solvent_subtracted=False, scope='Published curated means; no trial-level uncertainty.'),
        receptor_channels=len(profile), mapped_neurons=sum(r['anatomy']['n'] for r in profile),
        current_direct_odor_population='ORN_DM1', current_direct_odor_cells=74,
        other_channels_with_measured_increment=[r['receptor'] for r in profile if r['receptor']!='Or42b' and r['increment_Hz']!=0],
        current_input_nominal_ON_Hz=p['basal_hz']+p['increment_hz']*.5,
        empirical_anchor_total_Hz=p['basal_hz']+p['increment_hz'],
        nominal_ON_equals_empirical_concentration=False,
        PN_legacy_is_firing_measurement=False, PN_general_available_is_consumed=False,
        statistics=statistics,
        decision='Keep45 as a selective virtualDM1 experiment. A chemical ensemble is a distinct testable input hypothesis, not a repair proven to activateDNg100. No intrinsic law or gain selected. ORN→PN biological calibration remains open.')
    return result, a, profile

def plot(folder, result, arms, profile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
    ax = axes[0,0]
    ordered = sorted(profile, key=lambda r:r['increment_Hz'])
    ax.barh([r['receptor']+' / '+r['glomeruli'][0] for r in ordered],
            [r['increment_Hz'] for r in ordered], color=['#ef8200' if r['receptor']=='Or42b' else '#6546ae' for r in ordered])
    ax.set(title='A · 1-hexanol: patrón de 15 canales con datos', xlabel='Incremento medio publicado (espigas/s)')
    ax.text(.98,.04,'Dilución 10⁻² en parafina\nVehículo sin descontar · no es el estímulo45',
            ha='right',transform=ax.transAxes, fontsize=9)
    t = arms['odor']['paso']/1000
    axes[0,1].plot(t, arms['sham']['peripheral_nominal'][:,0],color='#777777',ls='--',label='Control')
    axes[0,1].plot(t, arms['odor']['peripheral_nominal'][:,0],color='#ef8200',label='Entrada bilateral virtual0,5')
    axes[0,1].set(title='B · Entrada periférica algebraica del ensayo45',ylabel='Tasa nominal de entrada (s⁻¹)')
    axes[0,1].legend()
    for ax, prefix, title in ((axes[1,0],'terminal','C · Actividad terminal ORN: q × capacidad'),
                              (axes[1,1],'bridge','D · Puente ORN→PN con recursos y filtros')):
        for side,color in [('L','#007db7'),('R','#e56018')]:
            for arm,style in [('odor','-'),('sham','--')]:
                ax.plot(t,arms[arm][prefix+'_'+side],style,color=color,label=side+(' · estímulo' if arm=='odor' else ' · control'))
        ax.set(title=title,ylabel='Media de señal interna nominal\n(no espigas medidas)')
        ax.legend(ncol=2)
    for index,label,color in [(0,'PN10208 · estado legado','#9c379f'),(1,'PN10176 · estado legado','#087c68')]:
        for arm,style in [('odor','-'),('sham','--')]:
            axes[2,0].plot(t,arms[arm]['PN_q_legacy'][:,index],style,color=color,label=label+(' · estímulo' if arm=='odor' else ' · control'))
    axes[2,0].set(title='E · Estados PN guardados: q',ylabel='q (adimensional; no frecuencia medida)')
    axes[2,0].legend(fontsize=8)
    for key,label,color in [('gamma_mean','100 destinos KCγ','#be303c'),('additional_mean','366 destinos adicionales','#355ca8')]:
        for arm,style in [('odor','-'),('sham','--')]:
            axes[2,1].plot(t,arms[arm][key],style,color=color,label=label+(' · estímulo' if arm=='odor' else ' · control'))
    axes[2,1].set(title='F · PN10208: salidas locales modeladas',ylabel='Conductancia media entre destinos (nS del modelo)')
    axes[2,1].legend(fontsize=8)
    for ax in axes.ravel()[1:]:
        ax.axvspan(1,3,color='#d7ae37',alpha=.08)
        ax.set(xlabel='Tiempo del ensayo45 (s)',xlim=(0,4))
        ax.grid(alpha=.18)
    fig.suptitle('Correspondencia sensorial y transmisión registrada\nA: medias experimentales de otro protocolo · B–F: datos del modelo45, sin nueva simulación',fontsize=17)
    fig.savefig(folder/'COMPARACION.png',dpi=145)
    fig.savefig(folder/'COMPARACION.pdf')
    plt.close(fig)

def main():
    start, cpu = time.monotonic(), time.process_time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract',action='store_true')
    parser.add_argument('--folder',type=Path,default=HERE)
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU,(600,600))
    if args.extract:
        need(not (args.folder/'OBSERVATIONS.npz').exists(),'Preserve existing projection')
        extract(args.folder)
    result, arms, profile = calculate(args.folder)
    if args.verify:
        need(result == json.loads((args.folder/'RESULTADOS.json').read_text()),'Recalculated result differs')
    else:
        need(not (args.folder/'RESULTADOS.json').exists(),'Preserve existing result')
        save(args.folder/'RESULTADOS.json',result)
        with (args.folder/'PERFIL_1_HEXANOL.csv').open('w',newline='') as out:
            writer=csv.writer(out);writer.writerow(['receptor','glomerulus','baseline_Hz','increment_Hz','total_Hz_approx','model_cells','direct_stimulus45'])
            for r in profile:
                writer.writerow([r['receptor'],r['glomeruli'][0],r['baseline_Hz'],r['increment_Hz'],r['reconstructed_total_Hz'],r['anatomy']['n'],r['receptor']=='Or42b'])
        plot(args.folder,result,arms,profile)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    need(rss<4*1024**3 and time.monotonic()-start<1800,'Analysis budget exceeded')
    usage=dict(CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start,RAM_peak_bytes=rss,GPU_s=0,simulated_ms=0)
    if not args.verify:save(args.folder/'EXECUTION.json',usage)
    print(json.dumps(dict(status='RECALCULATED' if args.verify else result['status'],
                         channels=result['receptor_channels'],mapped_cells=result['mapped_neurons'],usage=usage,
                         late_ON=result['statistics']['late_ON_2501_3000ms']),indent=2))

if __name__ == '__main__':
    main()
