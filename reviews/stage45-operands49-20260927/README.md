# Campaña 49: entradas consumidas y continuación

[Resultado y límites](RESULTADOS.md) · [Figura](RESULTADOS.png) · [Decisión siguiente](DECISION.md)

Se registraron cinco ventanas prefijadas en cada una de dos continuaciones de 50 ms con el olor apagado. Las 26.736 sumas de seis neuronas se reconstruyen exactamente. La diferencia residual en DNb05 no viene acompañada de diferencias corporales o propioceptivas entre historias. DNg100 no genera avance; no se ensayó navegación ni viento. Las etapas 4/5 siguen abiertas.

El archivo `PLAN.json` conserva el presupuesto anterior a la ronda. Los contratos posteriores documentan reparaciones del observador y del verificador, sin cambiar la intervención científica. Los fallos y la revisión no autoral se conservan. `CIERRE.json` distingue pasos comprometidos, intentados y costes efectivamente medidos.

## Reproducción del diagnóstico desde una extracción limpia

Requiere Python 3 y NumPy, sin GPU ni importaciones del árbol original:

```bash
python3 verify49.py
python3 -O verify49.py
python3 -O test_verifier_corruptions.py --output corrupciones_nuevas.json
```

Ambos modos reconstruyen todas las ventanas científicas seleccionadas y el informe. El paquete almacena los operandos mediante XOR reversible, sin cuantización; `observation_codec.py` reconstruye cada tensor y comprueba su hash. Los NPZ originales permanecen en el laboratorio. `CODEC_ROUNDTRIP.json` registra la identidad exacta.

`VERIFIER_CORRUPTIONS.json` conserva once alteraciones deliberadas realmente rechazadas, incluida la ley positiva. `aporte_motor/OBSERVER_REVIEW.json` conserva los contraejemplos que motivaron la reparación. Los archivos con `preserved` son versiones fallidas, no puntos de entrada vigentes.

## Alcance del paquete

Esta entrega reproduce el **diagnóstico completo de las ventanas capturadas**, no vuelve a simular el cerebro y el cuerpo. Incluye fuentes del observador, verificador, restaurador, contratos, datos, figura, fallos y revisión. Los estados científicos completos de cada ejecución siguen guardados localmente en sus directorios únicos, con RNG, propietarios y hashes.

Para volver a ejecutar los organismos hacen falta además los estados originales de 48, el cargador anatómico y los recursos del cuerpo histórico, las bibliotecas nativas y el entorno GPU identificados por `EXECUTED_SOURCES.json`. No están todos contenidos en esta cápsula: el paquete completo del organismo sigue pendiente. Los comandos efectivamente ejecutados están en los contratos y logs; no se afirma reproducción completa del CNS desde esta entrega.
