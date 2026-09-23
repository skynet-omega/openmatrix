# Revisión externa del código publicado

Conversación 6ab06db7-9908-83e9-a515-58c9e6e18a1a, turno 50ed61a0-359b-4ce6-b578-b021dea3081c. El revisor declara dos fixtures CPU; reproducción local necesaria antes de aceptar el hallazgo. No se ha descargado su ZIP sandbox.

**Conservaría el núcleo ADD/SET y la reparación LIF observada, pero corregiría una incoherencia concreta del registro `Ledger` antes de considerar cerrada esa interfaz.** No requiere cambiar tolerancias ni repetir la campaña completa.

En `reset_ports.py`, el flujo evoluciona hasta cada evento, mantiene \(s\) continuo y aplica `SET(post)` o `ADD(jump)` a \(q\). El orden simultáneo conserva el orden de entrada. `lif_event_metadata.py` exporta el valor posterior calculado por la fuente, no una saturación introducida por el solver.  

### Defecto reproducido: un SET puede desaparecer antes del puerto

**Ejecuté en CPU las clases publicadas `Ledger` y `Waveform` del vendor**, extraídas sin modificaciones de copias verificadas mediante sus hashes Git. Con \(q_0=0,8\):

`Ledger.add([20e-6], [0], [0.0], post_values=[0.7])`

termina con **cero eventos y `post_values=[]`**. `Ledger` conserva el filtrado heredado `jump>0`, aunque un SET no necesita un salto auxiliar positivo.  

La prueba mixta publicada utiliza precisamente SET con salto cero, pero pasa un `SimpleNamespace` directamente a `ResetFilterPorts`; **no atraviesa `Ledger`**. Por eso no detecta esta diferencia entre interfaces. 

**Corrección mínima:** validar y conservar eventos según su operación; un SET válido no debe descartarse por `jump==0`. Añadir ese caso al recorrido `Ledger→ResetFilterPorts`, sin nuevas cargas corporales.

**Esto no demuestra corrupción de las corridas publicadas ni refuta el kernel ADD/SET.** Es un defecto del enlace para una operación ya representable, no una exigencia futura de química, DAE o compatibilidad universal.

:chatgpt-content-reference{index="5"}[Reproductor CPU y alcance — 5,8 KB](sandbox:/mnt/data/MATRIX_ADD_SET_REVISION_CPU_77611a3.zip)

**Lectura:** completos `reset_ports.py`, `reset_adapter.py`, `lif_event_metadata.py`, `check_event_contract.py`, `macro_abc/README.md` y `verification_vendor/event_waveform.py`; parcialmente el índice y `VERIFIED.json`, incluidos los resúmenes de 1 y 20 ms. **Ejecuté dos fixtures CPU de esas clases; no CUDA, `event_coupling` instalado, arrays del organismo ni verificación del ZIP.** El enlace IR–organismo sigue pendiente; esta reparación no lo acredita.

