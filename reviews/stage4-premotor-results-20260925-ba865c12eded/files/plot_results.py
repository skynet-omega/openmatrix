"""Scientific plot of measured neural responses; imposed motion is not navigation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

HERE=Path(__file__).resolve().parent


def main():
    result=json.loads((HERE/'RESULT.json').read_text())
    if result['classification']!='DESCRIPTIVE_CAUSAL_SCREEN_COMPLETE':raise ValueError('Incomplete experiment')
    panel=json.loads((HERE/'CELL_PANEL.json').read_text())['rows']
    names={x['bodyId']:x['type']+' '+str(x['somaSide']) for x in panel}
    with np.load(HERE/'sham_01/probe/PANEL.npz') as z:q0=z['q'][40:].copy();ids=z['ids'].copy()
    with np.load(HERE/'sham_01/traces.npz') as z:command0=z['neural_command_shadow'][40:,1].copy()
    fig,axes=plt.subplots(2,3,figsize=(13,9),gridspec_kw={'height_ratios':[4,1]},layout='constrained')
    conditions=['DNp09_bilateral','DNa03_izquierda','DNa03_derecha']
    for j,condition in enumerate(conditions):
        folder=HERE/(condition+'_01')
        with np.load(folder/'probe/PANEL.npz') as z:delta=z['q'][40:]-q0
        im=axes[0,j].imshow(delta.T,aspect='auto',origin='upper',extent=(1,200,len(ids)-.5,-.5),
            cmap='RdBu_r',norm=SymLogNorm(linthresh=.001,vmin=-1,vmax=1,base=10))
        axes[0,j].set_title(condition.replace('_',' '));axes[0,j].set_yticks(range(len(ids)))
        axes[0,j].set_yticklabels([names[int(i)] for i in ids] if j==0 else ['']*len(ids))
        axes[0,j].axvline(20,color='black',ls=':',lw=.8);axes[0,j].axvline(60,color='black',ls=':',lw=.8)
        axes[0,j].set_xlabel('Tiempo del ensayo (ms)')
        with np.load(folder/'traces.npz') as z:y=np.rad2deg(z['neural_command_shadow'][40:,1]-command0)
        axes[1,j].plot(np.arange(1,201),y,color='#255a80');axes[1,j].axhline(0,color='grey',lw=.5)
        axes[1,j].axvspan(20,60,color='grey',alpha=.15);axes[1,j].set_xlabel('Tiempo (ms)')
        axes[1,j].set_ylabel('Δ mando neural (°/s)')
    fig.colorbar(im,ax=axes[0,:],label='Δ q frente a sham; escala simétrica logarítmica fuera de ±0,001',shrink=.75)
    fig.suptitle('Reclutamiento premotor en el modelo completo\nMovimiento corporal y entradas impuestos iguales; no es una prueba de navegación',fontsize=13)
    out=HERE/'RESPUESTAS.png'
    if out.exists():raise FileExistsError(out)
    fig.savefig(out,dpi=160);plt.close(fig)
    print(out)


if __name__=='__main__':main()
