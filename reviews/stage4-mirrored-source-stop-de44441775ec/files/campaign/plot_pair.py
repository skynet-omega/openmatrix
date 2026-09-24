"""Post-verification descriptive figure; never an acceptance test or animation."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    verified = json.loads((HERE/'RAW_VERIFIED_01.json').read_text())
    if verified['errors'] or verified['missing'] or verified['decision']['classification'] not in (
            'DESCARTADO_EN_ESTE_CONTRATO', 'PROMETEDOR_NO_CONFIRMADO',
            'CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA'):
        raise ValueError('Complete, valid raw pair required before drawing')
    series = {}
    for arm in ('plus', 'minus'):
        path = HERE/f'native_{arm}_01/traces.npz'
        with np.load(path, allow_pickle=False) as z:
            if z['fase'].tolist() != ['preparacion']*40+['ensayo']*400:
                raise ValueError('Unexpected trace phase')
            q = z['DN_q_usada']-z['DN_baseline']
            series[arm] = {
                'sensory_L_minus_R':z['sensores_usados'][40:,0]-z['sensores_usados'][40:,1],
                'reader_L_minus_R':q[40:,2]-q[40:,3],
                'command_deg_s':np.rad2deg(z['command_yaw_rate_rad_s'][40:]),
                'yaw_deg':z['yaw_delta_deg'][40:],
                'trace_sha256':sha(path),
            }
    x = np.arange(1,401)
    figure, axes = plt.subplots(4,1,figsize=(10,10),sharex=True,layout='constrained')
    labels = (('sensory_L_minus_R','Olor consumido L − R (unidad normalizada)'),
              ('reader_L_minus_R','DNb05: qL − qR respecto de basal'),
              ('command_deg_s','Mando angular (°/s)'),
              ('yaw_deg','Giro corporal respecto del preparado (°)'))
    for ax,(name,label) in zip(axes,labels):
        for arm,color,title in (('plus','#1764ab','Fuente izquierda'),
                                ('minus','#d47116','Fuente derecha')):
            ax.plot(x,series[arm][name],color=color,label=title,linewidth=1.4)
        ax.axhline(0,color='#777777',linewidth=.6)
        ax.axvspan(300,400,color='#888888',alpha=.10)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend(loc='best')
    axes[-1].set_xlabel('Tiempo desde el inicio del olor (ms)')
    figure.suptitle('Dos fuentes espejo: señales registradas (descriptivo, no prueba de navegación)')
    out=HERE/'PAIR_DIAGNOSTIC.png'
    if out.exists():
        raise FileExistsError(out)
    figure.savefig(out,dpi=160)
    plt.close(figure)
    receipt={'schema':'posthoc_pair_plot_v1','raw_verifier_sha256':sha(HERE/'RAW_VERIFIED_01.json'),
             'source_sha256':sha(__file__),'traces_sha256':{arm:series[arm]['trace_sha256'] for arm in series},
             'image_sha256':sha(out),'scope':'Selected observed signals at 1-ms spacing; not a full-brain movie or navigation test'}
    (HERE/'PAIR_DIAGNOSTIC.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
