from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_pilot import compute
H=Path(__file__).resolve().parent;r,d=compute(H);x=np.arange(1,91)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
colors={1:'#2878b5',-1:'#d46b27'}
for odor in [0,1]:
 for side,sign in [('L',1),('R',-1)]:
  name=f'air{side}_odor{odor}';base=f'air0_odor{odor}';ax[0,0].plot(x,d[name]['direction']-d[base]['direction'],color=colors[sign],ls='-' if odor else '--',label=f'Aire {side}, '+('olor' if odor else 'sin olor'))
ax[0,0].set(title='A. Respuesta de DNb05 a campos opuestos',ylabel='Cambio de q izquierdo − derecho (DNb05)');ax[0,0].legend(fontsize=8)
for mode,color,label in [('air0','#727272','Padre'),('G','#8660a8','Conductancia'),('I','#3d9b76','Corriente emparejada')]:
 ax[0,1].plot(x,d[mode+'_odor1']['direction']-d[mode+'_odor0']['direction'],label=label,color=color)
ax[0,1].set(title='B. Respuesta al olor por ley neuronal',ylabel='Diferencia olor − sin olor de q L−R');ax[0,1].legend(fontsize=8)
for odor in [0,1]:
 y=(d[f'airL_odor{odor}']['yaw']-d[f'airR_odor{odor}']['yaw'])/2;ax[1,0].plot(x,y,label='Con olor' if odor else 'Sin olor',color='#2878b5' if odor else '#727272')
ax[1,0].axhline(.02,color='#c84f49',ls=':',label='Mínimo material ±0,02');ax[1,0].axhline(-.02,color='#c84f49',ls=':');ax[1,0].set(title='C. Contraste de mando observado, sin aplicar',ylabel='Mitad de contraste de aire L−R (°/s)');ax[1,0].legend(fontsize=8)
g=d['G_odor1']['yaw']-d['G_odor0']['yaw'];i=d['I_odor1']['yaw']-d['I_odor0']['yaw'];ax[1,1].plot(x,g-i,color='#8660a8',label='Conductancia − corriente');ax[1,1].axhline(.02,color='#c84f49',ls=':');ax[1,1].axhline(-.02,color='#c84f49',ls=':');ax[1,1].set(title='D. Efecto adicional de conductancia sobre olor',ylabel='Diferencia de respuestas de mando (°/s)');ax[1,1].legend(fontsize=8)
for a in ax.ravel():
 a.axvspan(0,10,color='#dddddd',alpha=.4);a.axvspan(50,90,color='#dddddd',alpha=.2);a.axhline(0,color='#aaaaaa',lw=.7);a.set_xlabel('Tiempo de continuación (ms)');a.grid(alpha=.15);a.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
fig.suptitle('Campaña 51 · Diez condiciones del organismo completo\nTransferencia neural; etapas 4/5 abiertas\nRegistros originales: conversión de velocidad corporal pendiente de repetir',fontsize=13)
fig.savefig(H/'COMPARACION.png');plt.close(fig)
