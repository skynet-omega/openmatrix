# Diagnóstico del FAIL de rechazo

La única corrida diagnóstica preservó [la traza hexadecimal cruda](REJECTION_TRACE_RAW_01.txt): los ensayos GPU y CPU coinciden en `h`, tiempo usado y estado hasta el índice 9. En el índice 10, `h` GPU fue `0x1.0c6f7a0b5ed9p-21` y `h` CPU `0x1.0c6f7a0b5ed8cp-21`. La diferencia aparece al evaluar el resto hasta el final del intervalo, no antes de un evento.

El test original creó el final del oráculo como literal `2000e-9` = `0x1.0c6f7a0b5ed8dp-19`. El controlador nativo, igual que `graph_control_v2.cpp`, lo crea desde el argumento entero `duration_ns*1e-9` = `0x1.0c6f7a0b5ed8ep-19`: un ULP más. Ese último bit cambia un subpaso en el umbral sintético `h>500 ns`. Una reconstrucción **sólo CPU** con la misma expresión de tiempo que el contrato predice exactamente 7 aceptaciones, 6 rechazos, siguiente propuesta 500 ns y mínimo 251 ns, los valores observados en GPU. El resultado previo `FAIL_FROZEN_GATE` permanece: el oráculo de aquel archivo no se reescribe.

Se prepara una repetición prospectiva del fixture con el único cambio `end = duration_ns*1e-9` dentro de una función que recibe `duration_ns`; el controlador y su binario son idénticos. Pasar ese fixture probaría rechazo y frontera de evento **sólo en el modelo escalar**, no rollback total del organismo ni rendimiento.
