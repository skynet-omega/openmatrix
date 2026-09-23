# Stage3: evidencia compacta para revisión externa

La campaña19 declara **CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA** para orientación funcional del modelo PN629-off con lector efectivo DNb05 y cuerpo de rodillos. Conserva siete corridas completas, dos intentos incompletos y un poscierre CPU. El contrato estricto original no quedó cumplido retrospectivamente. El gate histórico de estado oculto KC permanece **FAIL**.

Este paquete conserva fuentes reales, contratos, flujo PN crudo, trazas, eventos descriptivos, fuerzas de apoyo y recibos. El inventario explícito indica ruta original, destino, tamaño y SHA256. Una copia del operador efectivo sustituye únicamente duplicados binariamente idénticos; las ocho identidades constan en el registro de lecturas poscierre. No se siguen enlaces simbólicos ni se incluyen credenciales.

`verify_compact.py` recalcula desde las trazas los cuatro pares nativo/referencia, dirección y márgenes, relojes y lectores, PN desde operador, apoyo mecánico, retirada angular y continuación observada. Compara esos números con los resultados originales y enlaza por hashes la cola, contrato, verificador, resultado completo y poscierre. No ejecuta las funciones integrales que requieren estados ausentes.

Los checkpoints de sesión de más de 100 MB se excluyen. Su igualdad semántica completa al preparar/restaurar se apoya aquí en el recibo poscierre, que sí fue reconstruido localmente usando esos archivos antes de empaquetar. **El paquete no permite repetir esa comprobación integral ni reproducir el organismo.** Los hashes no sustituyen archivos ni prueban por sí solos igualdad semántica. Tampoco contiene el cierre transitivo completo del organismo, assets corporales, anatomía o entorno GPU. No demuestra equivalencia biológica, marcha natural, navegación a fuente, aprendizaje, un motor general ni etapa4.

Los originales dentro de `AXIOMA_ASTRA/campanas/` conservan su contenido y fechas lógicas. Algunas notas prospectivas, como el README poscierre, describen el estado anterior a ejecutarse; para saber qué ocurrió se consultan los recibos reales incluidos. El intento uniforme16 se interrumpió sin RESULT ni trazas finales; la continuación17 falló antes del primer paso por una identidad corporal obsoleta. La reparación19 corrige esa identidad sin cambiar ecuaciones o tolerancias. No se rellenan archivos faltantes.

Después de verificar y concatenar las partes de `ARCHIVE.json`, extraer el ZIP en una carpeta nueva. Desde esa extracción, con Python y NumPy:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B verify_compact.py > VERIFY_LOCAL.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -I -B -O verify_compact.py > VERIFY_LOCAL_OPTIMIZED.json
```

Estas órdenes importan sólo el código científico incluido en la extracción y las bibliotecas estándar/NumPy instaladas. No arrancan GPU/organismo ni modifican las campañas. El resultado dice expresamente `full_checkpoint_equivalence_recomputed: false` y no emite una nueva admisión Stage3. No existe un modo completo autocontenido: requeriría los checkpoints, núcleo, datos y dependencias que el inventario declara ausentes.

La preparación usa `instrumentos/openmatrix/publish.py` con un manifiesto directo de originales. Se prepara localmente, sin publicación ni mensajes externos. Las pruebas sintéticas ejercitan rechazo de enlaces, recorrido de rutas y credenciales tanto en texto como en NPZ. El presupuesto prospectivo figura en PLAN.json.
