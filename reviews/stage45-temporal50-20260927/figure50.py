"""Static scientific figure from verified contrasts; never rerun simulation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
start=time.process_time();v=json.loads((H/'RESULTADOS.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
t=np.arange(1,141)
for ax,name,label,limit in [(axes[0,0],'DNb05','DNb05 L−R: interacción J',1.6e-5),(axes[0,1],'raw_yaw_deg_s','Giro crudo sin aplicar: J (°/s)',.02)]:
    c=v['contrasts'][name];ax.plot(t,c['J'],color='#006c84',lw=1.7,label='J(t)')
    ax.axhline(c['mean_J'],color='#dd613d',ls='--',label='Media prefijada 11–130 ms')
    ax.axhline(limit,color='#777777',ls=':',label='Umbral de materialidad ±')
    ax.axhline(-limit,color='#777777',ls=':')
    ax.axvspan(10.5,130.5,color='#e4eef2',alpha=.5,zorder=-10)
    ax.set(title=label,xlabel='Tiempo desde reanudación (ms)');ax.legend(fontsize=8)
for name,color in [('PN_10208','#006c84'),('PN_10176','#8b4c9b')]:
    axes[1,0].plot(t,v['contrasts'][name]['J'],label=name,color=color)
axes[1,0].set(title='Secundario: q de dos PN, contraste J',xlabel='Tiempo (ms)')
axes[1,0].legend(fontsize=8)
labels=list(v['arms']);values=[v['arms'][a]['DNg100']['target_max'] for a in labels]
axes[1,1].bar(labels,values,color='#777777')
axes[1,1].set(title='Objetivo DNg100 máximo en cada condición',ylabel='q objetivo',ylim=(0,max(1e-3,max(values)*1.2)))
if max(values)==0:axes[1,1].text(.5,.5,'Cero en todos los registros',transform=axes[1,1].transAxes,ha='center')
for ax in axes.flat:
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
fig.suptitle('Campaña 50 · '+v['decision']['classification']+'\nEtapas 4/5 abiertas; ocho condiciones, una preparación expuesta',fontsize=12)
fig.savefig(H/'RESULTADOS.png',dpi=180);plt.close(fig)
(H/'FIGURE_COST.json').write_text(json.dumps({'CPU_s':time.process_time()-start,'new_neural_ms':0})+'\n')
