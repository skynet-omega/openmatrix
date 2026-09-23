"""Export behavior from saved complete traces; plotting is outside the simulator."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'etapa3_largo_diagnostico_20260923_10'
def main():
    names=('sham','odor_left','odor_right','uniform')
    labels={'sham':'Sin olor','odor_left':'Olor izquierdo','odor_right':'Olor derecho','uniform':'Uniforme'}
    colors={'sham':'#5b6572','odor_left':'#ce603d','odor_right':'#286aa5','uniform':'#916da1'}
    data={}
    for group,folder in [('parent',PARENT),('child',HERE)]:
        for name in names:
            with np.load(folder/('full_'+name+'_01')/'traces.npz',allow_pickle=False) as z:
                m=z['fase']=='ensayo';t=z['paso'][m];y=z['yaw_delta_deg'][m]
                if len(t)!=400 or not np.array_equal(t,np.arange(1,401)):raise ValueError('Plot requires all complete arms')
                data[group,name]=(t.copy(),y.copy())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for i,(group,title) in enumerate([('parent','Padre: sustitución PN629 activa'),('child','Intervención: sustitución PN629 desactivada')]):
        for name in names:
            t,y=data[group,name];axes[i].plot(t,y,color=colors[name],label=labels[name],lw=1.8)
        axes[i].axhline(0,color='#222222',lw=.8);axes[i].axvspan(250,400,color='#dddddd',alpha=.25)
        axes[i].set(xlabel='Tiempo desde el estímulo (ms)',title=title,xlim=(0,400),ylim=(-.09,.18))
        axes[i].grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Giro absoluto (°); positivo = izquierda');axes[1].legend(frameon=False)
    fig.suptitle('Mismo circuito anatómico; intervención en un adaptador de salida',fontsize=12)
    fig.savefig(HERE/'behavior.png',dpi=180)
    fig.savefig(HERE/'behavior.pdf')
    plt.close(fig)
if __name__=='__main__':main()
