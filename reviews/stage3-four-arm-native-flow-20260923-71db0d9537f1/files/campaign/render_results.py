"""Render numeric tables only from the validated four-arm receipts."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
NAMES={'sham':'Sin olor','odor_left':'Izquierdo',
       'odor_right':'Derecho','uniform':'Uniforme'}

def main():
    close=json.loads((HERE/'CLOSE.json').read_text())
    data=json.loads((HERE/'LONG_DIAGNOSTIC.json').read_text())
    if not close['all_source_manifests_exact'] or not close['all_preodor_states_exact']:
        raise ValueError('Unverified common start')
    lines=['# Resultados verificados — cuatro brazos de 400 ms','',
           'Δyaw en grados desde el comienzo del estímulo; positivo = izquierda. '+
           'Los contrastes restan el sham del mismo instante.','','| Condición | 250 ms | 320 ms | 400 ms | Δyaw−sham a 400 ms | Comando integrado a 400 ms |',
           '|---|---:|---:|---:|---:|---:|']
    for key,label in NAMES.items():
        x=data['arms'][key];c=data['contrasts_to_sham'].get(key,{}).get('400',{})
        cell='—' if key=='sham' else f"{c['yaw_minus_sham_deg']:+.6f}"
        lines.append('| '+label+' | '+' | '.join(f"{x['windows'][str(ms)]['yaw_delta_deg']:+.6f}" for ms in (250,320,400))+
                     f" | {cell} | {x['command_integral_trial_deg']:+.6f} |")
    admission='SÍ' if close['stage3_admission'] else 'NO'
    lines+=['',f"Clasificación mecánica: **{close['classification']}**; admisión de etapa 3: **{admission}**.",
            f"Tiempo de cuatro corridas: {close['wall_four_runs_s']:.3f} s de un máximo predeclarado de {close['wall_budget_s']:.0f} s.",
            'Los cuatro brazos completaron 40 ms de preparación y 400 ms de prueba, '+
            'con fuente, estado previo y exposición sensorial comprobados. '+
            'El error máximo del target focal reconstruido en el kernel no certifica la fidelidad del estado completo.','',
            '| Condición | Target PN L−R a 320 ms | Target DNa02 no cero (440 muestras × 2) | Error máximo target nativo |',
            '|---|---:|---:|---:|']
    for key,label in NAMES.items():
        x=data['arms'][key]
        lines.append(f"| {label} | {x['windows']['320']['native_PN_target_L_minus_R']:+.8f} | {x['DNa02_native_target_nonzero_samples']} | {x['native_target_max_error']:.3g} |")
    lines+=['','El lector motor de este protocolo usa DNb05, no DNa02. '+
            'El test de dos motores a 20 ms de la campaña previa sigue fallando en estado KC oculto; '+
            'el resultado anterior no se reclasifica.','']
    (HERE/'RESULTS.md').write_text('\n'.join(lines))
    print(HERE/'RESULTS.md')

if __name__=='__main__':main()
