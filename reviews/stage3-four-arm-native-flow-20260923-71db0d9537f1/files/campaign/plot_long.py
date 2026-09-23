"""Four-arm organism diagnostic figure from frozen traces; no stage-3 verdict."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ARMS={'sham':'full_sham_01','odor_left':'full_odor_left_01',
      'odor_right':'full_odor_right_01','uniform':'full_uniform_01'}
LABELS={'sham':'Sin olor','odor_left':'Olor izquierdo',
        'odor_right':'Olor derecho','uniform':'Uniforme'}
COLORS={'sham':'#333333','odor_left':'#246db7',
        'odor_right':'#d05c25','uniform':'#759349'}

def load(folder):
    with np.load(folder/'traces.npz',allow_pickle=False) as z:
        mask=z['fase']=='ensayo'
        ms=z['paso'][mask].astype(int)
        if not np.array_equal(ms,np.arange(1,401)):
            raise ValueError('A full, consecutive 400-ms trial is required')
        dq=z['DN_q_usada'][mask]-z['DN_baseline'][mask]
        return {'ms':ms,'yaw':z['yaw_delta_deg'][mask].copy(),
                'command':np.rad2deg(np.cumsum(z['command_yaw_rate_rad_s'][mask])*.001),
                'dnb05':(dq[:,2]-dq[:,3]).copy()}

def main():
    data={name:load(HERE/folder) for name,folder in ARMS.items()}
    sham=data['sham']
    fig,ax=plt.subplots(2,2,figsize=(11,7),sharex=True,layout='constrained')
    for name,row in data.items():
        kw={'color':COLORS[name],'label':LABELS[name],'lw':1.8}
        ax[0,0].plot(row['ms'],row['yaw'],**kw)
        ax[0,1].plot(row['ms'],row['yaw']-sham['yaw'],**kw)
        ax[1,0].plot(row['ms'],row['command'],**kw)
        ax[1,1].plot(row['ms'],row['dnb05'],**kw)
    for a in ax.flat:
        a.axvspan(250,400,color='#f5dc80',alpha=.22)
        a.axhline(0,color='#777777',lw=.7)
        a.grid(alpha=.15)
        a.set_xlim(0,400)
    ax[0,0].set(ylabel='Yaw absoluto desde ON (°)',title='Cuerpo')
    ax[0,1].set(ylabel='Yaw menos sham (°)',title='Contraste con control')
    ax[1,0].set(xlabel='ms desde ON',ylabel='Comando integrado (°)',title='Lector DNb05')
    ax[1,1].set(xlabel='ms desde ON',ylabel='(DNb05 L−R) menos basal',title='Entrada efectiva del lector')
    ax[0,0].legend(loc='upper left',fontsize=8)
    fig.suptitle('Etapa 3: cuatro condiciones, mismo inicio, organismo completo; diagnóstico no certificado')
    out=HERE/'FOUR_ARM_DIAGNOSTIC.png'
    fig.savefig(out,dpi=170)
    print(out)

if __name__=='__main__':main()
