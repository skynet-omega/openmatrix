"""Generate the result and scientific figure from the reconstructed metrics."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify53 import compute, arrays

H=Path(__file__).resolve().parent


def main():
    result,data=compute(H)
    (H/'RESULTADOS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    queue=json.loads((H/'QUEUE_RESULT.json').read_text())
    fig,axs=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    x=np.arange(1,91)
    old_metrics={}
    for odor,color in [(0,'#2166ac'),(1,'#d95f02')]:
        label='Con olor' if odor else 'Sin olor'
        l,r=data[f'airL_odor{odor}'],data[f'airR_odor{odor}']
        a,b=arrays(H/'reference52'/f'airL_odor{odor}'/'traces.npz'),arrays(H/'reference52'/f'airR_odor{odor}'/'traces.npz')
        old_yaw=np.rad2deg(a['neural_yaw_unapplied_rad_s']-b['neural_yaw_unapplied_rad_s'])/2
        new_yaw=(l['yaw']-r['yaw'])/2
        old_q=((a['DN_q_actual'][:,2]-a['DN_q_actual'][:,3])-(b['DN_q_actual'][:,2]-b['DN_q_actual'][:,3]))/2
        old_metrics[str(odor)]=dict(q=float(old_q[50:].mean()),yaw_deg_s=float(old_yaw[50:].mean()))
        axs[0,0].plot(x[10:],l['totals'],color=color,label=label+' · campo L')
        axs[0,0].plot(x[10:],r['totals'],color=color,ls='--',label=label+' · campo R')
        axs[0,1].plot(x,new_yaw,color=color,label=label+' · emparejado53')
        axs[0,1].plot(x,old_yaw,color=color,ls=':',alpha=.7,label=label+' · original52')
        axs[1,0].plot(x,(l['direction']-r['direction'])/2,color=color,label=label)
    axs[0,0].set(title='Cantidad JO que recibe el kernel',ylabel='Suma de componentes FP32 (unidades del modelo)')
    axs[0,0].set_ylim(0,result['common_total']*1.12)
    axs[0,0].ticklabel_format(axis='y',style='plain',useOffset=False)
    axs[0,1].set(title='Diferencia del giro calculado: (L − R)/2',ylabel='°/s · giro sin aplicar al cuerpo')
    axs[1,0].set(title='Diferencia de salida DNb05: (L − R)/2',ylabel='Unidades de salida del modelo')
    labels=['Sin olor','Con olor'];old=[old_metrics[str(i)]['yaw_deg_s'] for i in [0,1]]
    new=[result['contrasts'][str(i)]['half_L_minus_R_yaw_deg_s'] for i in [0,1]]
    pos=np.arange(2)
    axs[1,1].bar(pos-.18,old,.36,label='Original52',color='#999999')
    axs[1,1].bar(pos+.18,new,.36,label='Emparejado53',color='#238b45')
    axs[1,1].set(xticks=pos,xticklabels=labels,title='Media de la ventana fijada: 51–90 ms',ylabel='°/s · semidiferencia calculada')
    for a in [axs[0,1],axs[1,1]]:
        for v in [-.02,.02]:a.axhline(v,color='#555555',lw=.8,ls='--')
    for a in axs.flat:
        a.grid(alpha=.18);a.legend(fontsize=7)
    for a in [axs[0,0],axs[0,1],axs[1,0]]:a.set_xlabel('Tiempo de este ensayo (ms)')
    fig.suptitle('Campaña53 · Control de cantidad sensorial · Etapas4/5 pendientes',fontsize=13)
    fig.savefig(H/'COMPARACION.png',dpi=170);plt.close(fig)
    lines=['# Resultado de la campaña53: cantidad JO emparejada','',
           '**Etapas4/5 siguen abiertas.** Esta campaña compara transferencia neuronal; registra giro sin aplicarlo al cuerpo y no ensaya perturbaciones mecánicas.','',
           '| Condición | Giro calculado original52 | Giro calculado emparejado53 | Diferencia DNb05 emparejada | Criterio material heredado |',
           '|---|---:|---:|---:|---|']
    for i,label in enumerate(labels):
        c=result['contrasts'][str(i)]
        lines.append(f"| {label} | {old_metrics[str(i)]['yaw_deg_s']:.6f} °/s | {c['half_L_minus_R_yaw_deg_s']:.6f} °/s | {c['half_L_minus_R_DNb_q']:.9g} | {'Persiste' if c['both_material'] else 'No alcanza ambos mínimos'} |")
    survived=[labels[i] for i in [0,1] if result['contrasts'][str(i)]['both_material']]
    lines+=['', 'Valores: semidiferencia entre campos L/R, promediada en51–90ms. La referencia52 es histórica y expuesta, no otra réplica nueva.', '',
            ('El efecto material persiste en '+', '.join(survived)+'. La diferencia de suma no basta para explicar ese contraste; no identifica todavía un controlador direccional útil.' if survived else 'El efecto no conserva ambos mínimos en ninguna condición. No es robusto bajo esta normalización; eso no establece que la cantidad sea su causa única.'),'',
            f"Cantidad común fijada: {result['common_total']:.9f} unidades internas. Máxima diferencia entre sumas consumidas: {max(c['max_pair_total_difference'] for c in result['contrasts'].values()):.9g}; límite prospectivo {2e-7*result['common_total']:.9g}.",
            'La normalización conserva identidades y soporte. No empareja simultáneamente L2, número de células activas ni conectividad. Se registró el vector FP32 real y se comprobó ausencia de discrepancias en las evaluaciones del kernel.','',
            f"Cualificación:2ms con normalización desactivada, campos previos/propietarios/eventos exactos frente a52. Cuatro brazos de90ms: total {queue['committed_ms']}ms CNS. CPU de trabajadores {queue['CPU_s']:.3f}s; cola {queue['queue_wall_s']:.3f}s. Preparación y análisis se contabilizan aparte.",'',
            'La interacción con olor, las trayectorias temporales, los grupos anatómicos y hashes se reconstruyen en RESULTADOS.json. Un signo constante no es requisito general de control; estas señales tampoco constituyen por sí solas orientación, iniciación por olor ni aprendizaje.','',
            '![Comparación de datos](COMPARACION.png)','',
            'Reproducción analítica: `python -B -O verify53.py`. Requiere NumPy, arrays de esta carpeta y referencias52 incluidas; no inicia CNS/GPU. No volver a lanzar la cola sobre los directorios cerrados.']
    (H/'RESULTADOS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(H/'RESULTADOS.md'),'figure':str(H/'COMPARACION.png'),'material_conditions':survived}))


if __name__=='__main__':main()
