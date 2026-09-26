"""Generate all numerical report entries from independently verified arrays."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from verify_clock import verify,raw,geometry


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=HERE/'run_02')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=verify(a.run)
    donor=json.loads((HERE/'FIRST_WIND_FORCE.json').read_text())
    lines=['# Campaña41 — diagnóstico físico después del viento','',
        '**Clasificación: '+r['classification']+'. Etapas4/5 abiertas.**','',
        'Reproducción del cuerpo MuJoCo y la prótesis con órdenes congeladas de una sola vida expuesta de campaña40. No se cargó ni simuló el cerebro. Las intervenciones son instrumentos del evaluador; no son políticas propuestas para el organismo.','',
        '| Condición | Error final a fuente (°) | Cambio desde1020ms (°) | Mejora frente a identidad (°) |',
        '|---|---:|---:|---:|']
    labels={'identity':'Órdenes originales y reinicio original','zero_yaw_after_wind':'Giro nulo después del viento','zero_forward_after_wind':'Avance nulo después del viento'}
    for name,x in r['arms'].items():
        lines.append(f"| {labels[name]} | {x['final_abs_error_deg']:.6f} | {x['postwind_abs_error_change_deg']:+.6f} | {r['improvement_vs_identity_deg'].get(name,0):+.6f} |")
    x=r['arms']['identity']
    lines.extend(['',f"Descomposición geométrica desde1020ms: Δdirección a la fuente={x['delta_bearing_deg']:+.6f}°, Δyaw corporal={x['delta_yaw_deg']:+.6f}° y Δerror firmado={x['delta_signed_error_deg']:+.6f}°. Se cumple Δerror=Δdirección−Δyaw. Son términos geométricos; no porcentajes aditivos de causalidad.",''])
    for name,effect in r['improvement_vs_identity_deg'].items():
        lines.append(f"- {labels[name]}: mejora={effect:+.6f}°; supera el mínimo prospectivo de0,5°: **{'sí' if r['material_improvement'][name] else 'no'}**.")
    lines.extend(['','El primer replay continuo falló identidad: era exacto hasta1000ms y divergía tras el reinicio histórico. La sonda sin integración midió fuerza generalizada nula en la primera llamada de viento con datos MuJoCo fríos, frente a norma '+f"{donor['consistent_force_norm']:.15g}"+' con cinemática consistente en un estado auxiliar. El replay reparado reproduce el reinicio y su primera llamada nula; no modifica retrospectivamente la campaña40. Su negativo de navegación permanece.',
       '',f"La identidad reparada tiene errores máximos qpos={r['identity_errors']['qpos']:.3g}, qvel={r['identity_errors']['qvel']:.3g}, yaw={r['identity_errors']['yaw_deg']:.3g}°. El brazo sin viento quedó **NO EJECUTADO**: el intento fallido consumió una de las cuatro ejecuciones. No se amplió el presupuesto ni se relajaron criterios.",'',
       f"Presupuesto consumido: {r['budget_consumed']['wall_with_failed_identity_s']:.3f}/600s de proceso de los replays (incluye el fallo);8000ms corporales y cuatro intentos. Cero organismos, GPU, entrenamiento o ajustes de ganancia. Tres condiciones útiles; no cohorte de semillas ni confirmación reservada.",'',
       'Los efectos pueden interactuar. Las órdenes permanecen congeladas al cambiar la trayectoria; por ello no se predice la respuesta neural al retirar avance o giro. Detener el avance puede contener el error angular sin acercarse al objetivo: es un diagnóstico, no navegación. El efecto mecánico emparejado de retirar viento sigue pendiente.','',
       'A sigue prioritaria: identificar una interfaz conjunta avance/giro con estímulos independientes y controles sensoriales emparejados. B (CNS→VNC→MN) queda acotada; C (músculos y seis patas) posterior. No integrar patas ni rescatar el filtro con otro umbral sobre esta vida.','',
       'Fuentes, contrato, fallo, reparación, trazas, modelo MuJoCo y estados del controlador están incluidos. El paquete reproduce este experimento físico; no contiene el cerebro completo ni acredita aprendizaje o equivalencia biológica.'])
    with a.out.open('x') as f:f.write('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=np.asarray(json.loads((HERE/'inputs/donor_GAUSSIAN_SPEC.json').read_text())['minus']['source_mm'])
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for name in r['arms']:
        t=raw(a.run/name/'trace.npz');g=geometry(t['qpos'],source)
        axes[0].plot(t['step'],g['abs_error_deg'],label=labels[name])
        axes[1].plot(t['qpos'][:,0]*10,t['qpos'][:,1]*10,label=labels[name])
    axes[0].axvspan(1000,1020,color='gray',alpha=.25)
    axes[0].set(xlabel='Tiempo (ms)',ylabel='Error a fuente (°)');axes[0].legend(fontsize=8)
    axes[1].scatter(*source,marker='*',s=120,color='black',label='Fuente fija')
    axes[1].set(xlabel='x (mm)',ylabel='y (mm)');axes[1].axis('equal')
    fig.savefig(a.out.with_suffix('.png'),dpi=160);plt.close(fig)


if __name__=='__main__':main()
