# Lecturas GPU→CPU y contrato de puertos para un runtime nativo

Análisis de sólo lectura, 24-09-2026. Perfil: `motor_nuevo/epoch_cost_20260923/profile_on_01/PROFILE.json`, SHA-256 `d72addc51e380bb4637250fbf05c6120bcb2b7401f858778a15ebbbedef43197`. Corrida `causal_cuda`, sham, observador apagado, 19 pasos de 1 ms perfilados después del primer paso de calentamiento. El JSON conserva tiempos y recuentos por función, **pero descarta el grafo de llamadores de `cProfile`**. Las categorías que siguen combinan esos recuentos con los fuentes congelados; no son tiempos exclusivos por dueño.

## Qué se puede atribuir

| Hecho | 19 ms perfilados | Por época CNS (304) | Interpretación |
|---|---:|---:|---|
| `cupy.ndarray.get` | 18.240 llamadas; 7,0748 s de tiempo propio | 60 | Transferencia y/o espera de GPU; el perfil no separa ambas. |
| `cupy.asnumpy` | 15.200 llamadas; 0,9514 s acumulados | 50 | Hasta 50 de los 60 `get` pueden venir de este envoltorio. No sumar 0,9514 a 7,0748. |
| `ProjectedKcBatch.host` | 7.904 llamadas; 0,5294 s acumulados | 26 | Usa `asnumpy` si el backend es CUDA; subconjunto de la fila anterior. |
| `get` fuera de `asnumpy` | 3.040 por diferencia de recuentos | 10 | Reconstrucción de código congruente con los tres dueños siguientes. |

Los 10 `get` directos por época coinciden exactamente con: **uno** de `OrganismAdapter.step` al copiar el estado CNS completo; **dos** de `StageGraph.run`, que lee el vector de 11 valores para aceptar cada una de las dos etapas PN; **siete** de `DeviceCell.advance`, que lee `status`, `counts`, `maximum_error`, `ec`, `et`, `ej` y `ep`. Sus funciones se llamaron 304, 608 y 304 veces respectivamente. La corrida declaró cero retrocesos PN y el `DeviceCell` se instaló en todos los intervalos. Es una atribución fuerte por identidad de recuentos y fuentes, **no una medición directa de los tiempos de cada una**; podrían existir otras rutas condicionales compensadas. Los archivos relevantes tienen hashes: `organism_adapter.py` `355666d4f0510ec5d2cb86843c76c65d2035e617271aa21b1d9dba364a18ded6`, `device_cell.py` `00ec0aec5d9d4692690bfdd4c6477048de731d7538179bcebf3cfa7c8d745d45`, `graph_stage.py` `9ea3998e32d33c82b6f67e8ca391b555c6b314239197722cfc57fc5a54999811`.

El acumulado de **todo** `asnumpy` es sólo 0,9514 s. Como ese acumulado contiene el coste de sus llamadas hijas, al menos `7,0748 - 0,9514 = 6,1233 s` de tiempo propio de `get` ocurrieron fuera de `asnumpy`, bajo la semántica del perfil. Esa cifra no mide cuánto puede ahorrarse: un `get` puede esperar el cálculo CUDA pendiente. El volumen lógico del único `g.x.get` es `359.373 × 8 × 304 = 874,00 MB` en 19 ms; el mismo estado se sube al empezar cada época. Los controles PN son sólo 11 FP64 por etapa, pero pueden forzar sincronización. No hay medición de bytes PCIe, ancho de banda real ni energía.

Las otras 50 llamadas a `asnumpy` por época no se pueden distribuir exactamente sin las aristas de llamador. De ellas, 26 están en `ProjectedKcBatch.host`. El código llama a `host` para exponer `q`, contar/validar eventos, y serializar ocho campos (`delta`, `gates`, `q`, `counts`, `last_siz`, `previous_slope`, `trough`, `clipped`) mediante `state_dict`. `ParentSnapshotFrames` captura el estado evolutivo antes del predictor que luego se descarta; esos datos tienen función de **rollback**, no de telemetría. PN también convierte voltaje, compuertas y carga al importar/exportar la propuesta al objeto histórico y toma observables para el controlador eléctrico. Hay estados/lecturas repetidos que podrían permanecer en VRAM con propiedad y snapshots transaccionales, pero no se deben borrar hasta reproducir sus consumidores y el checkpoint.

El perfil comienza y termina en `obj.step()`. `observer` estaba apagado; `observer.sample_last`, `d.captura`, la escritura de trazas y los checkpoints al final de la corrida se ejecutaron **fuera** del tramo perfilado. Así que estos 18.240 `get` no son evidencia de que la telemetría sea el cuello de botella. La copia total del CNS y las exportaciones PN/KC satisfacen contratos científicos de estado y orden para el diseño actual; una implementación nativa puede cambiar **dónde** residen esos estados sin perderlos. `status`, errores y eventos son necesarios para decidir, aunque la lectura de cada arreglo a Python no lo es si decide un controlador nativo.

## Puertos nativos propuestos

La interfaz no codifica PN/KC ni anatomía en el scheduler. Los modelos registran campos, leyes y rutas; el motor registra dueño, reloj, formato, estado y precedencia. C++ conserva el lazo de control y CUDA las operaciones de datos. MuJoCo continúa en CPU mediante su API nativa, con un lote compacto de sensores y comandos en cada frontera física declarada.

```cpp
enum class Residence { Device, Host, Shared };
enum class EventOp { Set, Add };
struct PortId { uint32_t value; };
struct PortDesc {
  PortId id; uint32_t owner; uint32_t dtype; uint64_t elements;
  Residence residence; uint32_t clock_domain; uint32_t schema_version;
};
struct Event {
  int64_t time_ns; uint64_t sequence; PortId destination;
  EventOp op; double value; uint32_t source;
};
struct Exchange {
  int64_t t0_ns, t1_ns; const PortId* inputs; size_t n_inputs;
  const PortId* outputs; size_t n_outputs;
};
class Runtime {
 public:
  void register_model(const ModelDescriptor&); // campos, operadores y límites
  void begin_transaction(int64_t t0_ns, int64_t t1_ns);
  void bind_inputs(const Exchange&, const BufferView*); // lote y versión
  TrialResult propose(const Exchange&); // sin publicación externa
  void commit(const TrialResult&);       // estado + eventos atómicos
  void rollback();                       // restaura también RNG y dueños
  OutputBatch read_ports(const PortId*, size_t); // sólo puertos solicitados
  void checkpoint(CheckpointWriter&);   // estado completo y hashes
};
```

Contrato causal de una época de 125 µs: guardar en dispositivo un snapshot transaccional de CNS/PN/KC y productores; ejecutar el predictor de medio CNS y obtener sólo los puertos intermedios; restaurar **antes** de avanzar PN media época; avanzar CNS aceptado con los eventos físicos que cortan el tiempo en sus marcas exactas; publicar KC/axones y completar PN con observables de fin de época. La prioridad `SET`/`ADD`, el orden estable `(time_ns, sequence)`, las cotas de capacidad y el rechazo por desbordamiento pertenecen al scheduler. Una lectura del cuerpo en CPU se hace únicamente en la frontera de acoplamiento y se transfiere en un lote; el cuerpo recibe el comando que corresponde al mismo reloj, nunca un comando futuro del predictor descartado. El snapshot del evaluador incluye estado CNS, PN, KC, cuerpo, sensores pendientes, pesos plásticos, eventos, RNG, relojes, parámetros y esquema; scratch y buffers temporales se reconstruyen sin convertirse en memoria operativa.

Un primer prototipo debe mantener los mismos 16 intercambios CNS/ms y la misma física, reemplazar sólo puertos individuales por un lote nativo, y comparar hashes de checkpoints/órdenes de eventos con el padre. Después puede medirse si una época común elimina lecturas sin alterar las trayectorias. Añadir un modelo nuevo de estado y eventos sin editar el scheduler es la prueba mínima de generalidad.

## Techo de mejora y decisión

El perfil completo sumó 59,8875 s para 19 ms. Si se eliminara **todo** el tiempo propio de `get`, y todo lo demás quedara idéntico, el techo de aceleración sería `59,8875 / (59,8875 - 7,0748) = 1,134×`; esto es optimista porque las esperas pueden trasladarse al siguiente punto de sincronización. Eliminar sólo las conversiones `asnumpy` tiene techo aún menor: `59,8875 / (59,8875 - 0,9514) = 1,016×`, porque su tiempo acumulado ya está incluido en otras filas. No sumar tiempos acumulados de `DeviceCell.advance` (5,54 s), PN (6,50 s), KC (5,92 s), ni `NativeGraph.advance` (40,255 s): hay anidamiento. Si todo `NativeGraph.advance` costara cero, el resto todavía consumiría 19,6324 s por 19 ms, frente a la meta de 1,14 s por 19 ms.

Conclusión de ingeniería: agrupar puertos y mover control a C++ es necesario para un runtime general y puede quitar esperas/copies, pero no justifica atribuirle el salto de ~52×. La prueba que importa mide **organismo completo** con la misma política de eventos, precisión y checkpoint, mientras una línea paralela reduce las evaluaciones globales o certifica una integración multirritmo. Una versión sin los 18.240 `get` que preserve sólo una trayectoria breve pero pierda PN/KC/body o replay exacto queda descartada.
