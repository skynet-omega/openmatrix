# Contexto y capacidad directa de DNg100

Ronda CPU del 28-09-2026. **No se superaron etapas4/5.** La capacidad algebraica directa de descendentes/ascendentes existe, pero el saldo recibido en las ventanas observadas es inhibitorio. Las MBON anotadas no tienen aristas directas hacia estas DNg100. Ninguna cota demuestra un estado poblacional alcanzable ni una reparación biológica.

- [Decisión, alternativas y criterio de parada](DECISION.md).
- [Plan anterior al cálculo](PLAN.md), [contrato](CONTRACT.json) y [congelación](FROZEN.json).
- [Fuentes primarias y revisión conceptual](FUENTES.md).
- [Revisión de diseño Motor](aporte_motor/REVISION_DISENO.md).
- [Reconstrucción independiente final de Motor](aporte_motor/REVISION_FINAL.md): todas las tablas reconstruidas desde NPZ originales y Parquet, sin ejecutar el adaptador ni su codec.
- [Ubicación del resultado original](RESULT_PATH.json), [decisión mecánica](cycle_result.json).

La receta verificada no ejecuta CNS ni GPU:

```bash
cd /home/daroch/AXIOMA_FLYWIRE/matrix
./matrix workbench run dng-context-capacity
./matrix workbench cycle show 9604950ab76e402e971f48d31e3eea55
./matrix lab find 'DNg100 contexto capacidad'
```

Con entradas y fuentes idénticas se reutiliza el resultado. `NEGATIVE` responde a la hipótesis «todos los grupos son directamente insuficientes», no significa que el verificador falló. No se cambia el nombre ni los criterios para obtener otro veredicto.

El cálculo principal consumió 2,0918296 s CPU. Motor empleó 3,3713267 s CPU instrumentados entre diseño y reconstrucción independiente. Pasaron las 783 pruebas del laboratorio en 40,10 s de pared, incluidas las nueve nuevas sobre signos, máscaras, dominio, orden de reducción, fases y rechazo de criterios alterados. La adquisición original es la campaña49; no se repitió. Publicación y reproducción quedan en sus recibos al cierre.
