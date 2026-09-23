"""Render current PN629 conclusions only from the verified numerical closure."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
def main():
    r=json.loads((HERE/'CLOSE.json').read_text())
    body=json.loads((HERE/'body_factorial_01/RESULT.json').read_text())
    lines=['# Intervención PN629: resultados reconstruidos','',
           'Se desactivó sólo la sustitución general de629 destinos del PN izquierdo antes de40ms de preparación. Se conservaron466 destinos dinámicos, anatomía y lector motor. Positivo significa izquierda.','',
           '| Condición | 250ms | 320ms | 400ms | Comando integrado400ms |',
           '|---|---:|---:|---:|---:|']
    for name in ('sham','odor_right','odor_left','uniform'):
        if name not in r['runs']:continue
        w=r['runs'][name]['windows']
        lines.append('| '+name+' | '+' | '.join(f"{w[str(t)]['yaw_delta_deg']:+.6f}°" for t in (250,320,400))+f" | {w['400']['command_integral_deg']:+.6f}° |")
    lines+=['',f"Cambio del derecho frente al padre: {r['right_change_vs_parent_deg']:+.6f}°. Derecho menos sham del candidato: {r['right_minus_child_sham_deg']:+.6f}°.",
            f"Decisión de esta intervención: **{r['classification']}**. Etapa3: **NO admitida**.",'',
            f"El factorial corporal reproduce exactamente las dos trayectorias propias. Cambio total del sham: {body['total_change_deg']:+.6f}°; cambio de comandos sobre cuerpo padre: {body['command_effect_at_parent_deg']:+.6f}°; cambio de estado preparado con comandos padre: {body['prepared_state_effect_at_parent_deg']:+.6f}°; interacción: {body['interaction_deg']:.3g}°.",'',
            f"Corridas de organismo: {r['wall_all_runs_s']:.3f}/{r['wall_budget_s']:.0f}s. Cuatro replays corporales: {body['wall_total_s']:.3f}/240s; sin nueva simulación cerebral.",'',
            'Las preparaciones entre olores y fuentes ejecutadas se compararon localmente. La neutralidad del observador se verificó a1ms. Estas comprobaciones no certifican equivalencia numérica de400ms.','',
            'A: la sustitución PN629 contribuye al sesgo; su suficiencia se decide por el criterio congelado. B: convergencia posterior/lector. C: historia neural basal. Los otros dos mecanismos no quedan descartados por esta intervención.','',
            'No se reclasifican los negativos históricos, no se llama homologación completa al cambio y no se afirma equivalencia neurobiológica.']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
