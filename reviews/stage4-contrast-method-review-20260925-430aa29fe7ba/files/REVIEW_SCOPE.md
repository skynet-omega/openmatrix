# Revisión acotada de navegación, campaña36

El organismo de control sigue en ejecución. Este paquete no contiene un resultado nuevo de la pareja ni demuestra navegación. Incluye protocolo congelado, implementación real del replay, comparadores y trazas completas del donante de1s, además de análisis de lector calculado localmente. No incluye los checkpoints pesados ni permite reejecutar todo el organismo de forma independiente.

Leer DECISION.md para las alternativas y límites, PLAN.json para presupuesto y criterios. run_replay.py y replay_boundary.py implementan la intervención; verify_pair.py y chatgpt_verificar_cintas_cd.py verifican el resultado. El segundo verificador es portable para comparar trazas; todavía no existe una traza de ablación que entregarle. PREPARATION_EQUALITY.json es un recibo local de588 arrays, no su reproducción independiente en este paquete.

La fuente completa anterior y el donante también están publicados en https://github.com/skynet-omega/openmatrix/tree/34268755d88d45a179c7141cf504218c14a19f5a/reviews/stage4-one-second-negative-20260925-35f02107a658 . Las dependencias del runner de organismo se consultan allí y en SOURCE_LOCK.json; el ZIP actual no es un instalador del motor. El motor Neurocore es otra campaña y no está operando estos brazos.

Para calcular sólo métricas del donante en una extracción limpia con NumPy, usando el código del revisor externo:
```python
from pathlib import Path
from chatgpt_verificar_cintas_cd import cargar, metricas
import json
import numpy as np
z = cargar(Path('donor/traces.npz'))
fuente = np.asarray(json.loads(Path('donor/CAMPOS.json').read_text())['minus']['source_mm'], float)
print(metricas(z, fuente))
```
El cálculo de sensibilidad de ganancias figura con su código exacto en DECISION.md; cambiar la ruta del NPZ a donor/traces.npz. Ninguna ganancia del organismo se modificó ni seleccionó para aprobar una etapa.
