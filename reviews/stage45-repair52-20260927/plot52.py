"""Scientific figure from verified population and original metrics only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent


def main():
    p=json.loads((H/'POPULATIONS.json').read_text())
    r=json.loads((H/'PILOT_RESULTS.json').read_text())
    time=np.arange(1,91)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(11.6,7.6),constrained_layout=True)
    colors={'JO_CE':'#3577A5','AMMC_WED':'#C76C32','all_DN':'#389476'}
    c=p['contrasts']['0']
    for direction,color in [('L','#356DA3'),('R','#CA7642')]:
        ax[0,0].plot(time,c['JO_consumed_sum_'+direction+'_per_ms'],label='Campo '+direction,color=color)
    ax[0,0].set(title='Entrada JO en frontera aditiva · sin olor',ylabel='Suma JO antes del cast FP32',xlabel='Tiempo de continuación (ms)')
    ax[0,0].legend(frameon=False)
    labels={'JO_CE':'335 JO-C/E','AMMC_WED':'1.108 AMMC/WED','all_DN':'1.314 descendentes'}
    for group in p['groups']:
        y=np.array(c['populations'][group]['rms_per_ms'])
        ax[0,1].plot(time,np.where(y>0,y,np.nan),label=labels[group],color=colors[group])
    ax[0,1].set_yscale('log')
    ax[0,1].set(title='Respuesta distribuida · campo L−R, sin olor',ylabel='RMS de diferencia en salida q',xlabel='Tiempo de continuación (ms)')
    ax[0,1].legend(frameon=False)
    for odor,color in [('0','#356DA3'),('1','#8C4F93')]:
        ax[1,0].plot(time,p['contrasts'][odor]['raw_yaw_half_per_ms'],color=color,label='Con olor' if odor=='1' else 'Sin olor')
    ax[1,0].axhline(0,color='#777777',lw=.7)
    ax[1,0].axhspan(-.02,.02,color='#999999',alpha=.12,label='±0,02 °/s')
    ax[1,0].set(title='Mando calculado, sin aplicar al cuerpo',ylabel='½(L−R) del giro (°/s)',xlabel='Tiempo de continuación (ms)')
    ax[1,0].legend(frameon=False)
    names=['air0_odor0','air0_odor1','G_odor0','G_odor1','I_odor0','I_odor1']
    vals=[r['arms'][name]['mean_forward'] for name in names]
    ax[1,1].bar(np.arange(6),vals,color=['#AEB8C3','#587CA2','#94C8B3','#3D9374','#DAB795','#B27745'])
    ax[1,1].set_xticks(np.arange(6),['Base\nsin olor','Base\ncon olor','G\nsin olor','G\ncon olor','I\nsin olor','I\ncon olor'])
    ax[1,1].set(title='La actividad basal y la respuesta al olor se separan',ylabel='Mando medio de avance (mm/s), 51–90 ms')
    for a in [ax[0,0],ax[0,1],ax[1,0]]:
        a.axvline(10.5,color='#999999',ls=':',lw=.8)
        a.axvspan(50.5,90,color='#98B4CA',alpha=.08)
        a.grid(alpha=.12)
    fig.suptitle('Campaña 52 · unidades corregidas y observación ampliada\nUna preparación, diez condiciones de 90 ms; sin admisión de etapas 4/5',fontsize=14)
    fig.savefig(H/'COMPARACION.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
