# Evidencia e interpretación de la campaña 42

## Señal bajo dos fuentes espejo

La pareja estricta izquierda/derecha de 400 ms compartió preparación. Se compararon canales homólogos por ventana, sin escalar un campo contra otro de unidades diferentes. Cada RMS del par se contrastó con la discrepancia del mismo campo entre motor nativo y referencia estricta. En ms 101–400, la entrada L/R tuvo RMS de diferencia entre fuentes 0,384434 frente a discrepancia numérica máxima 9,57×10⁻⁹. ORN izquierda/derecha dieron 0,181254/0,199310 frente a ≤6,62×10⁻⁹; PN legado 0,012962 frente a 4,18×10⁻⁹; DNb05 leído 0,000141274 frente a 1,38×10⁻⁹. Estos son números del modelo, no tasas medidas en mosca ni tamaños de efecto comparables entre campos. El perfil completo incluye transmisión PN y ventanas más tempranas.

La diferencia de mando fue positiva en 399 ms, cero en 1 ms y nunca negativa. Su integral firmada y la integral absoluta fueron ambas 0,057746766°. Por tanto, el integral pequeño no oculta pulsos pareados opuestos que se cancelen. Sí puede coexistir con órdenes individuales que cambian de signo: el brazo de fuente izquierda integró −0,002624° en ms 301–400. La pareja muestra una dependencia fuente→lector→orden, no corrección de rumbo bajo feedback.

## Viento y filtro en la vida de 2 s

En ms 1021–2000 la fuente permaneció a la derecha del contraste L−R sensorial. El mando crudo integró +0,014937° y el entregado por EMA/relé +2,190000°. En ms 1601–2000 el crudo integró −0,027848° —sentido localmente correctivo— mientras el aplicado aún integró +0,555000°. Estas observaciones no señalan una sinapsis defectuosa: la recurrencia del filtro y el lector se reconstruyeron exactamente, y la actividad que llega al lector varía.

La única intervención nueva fue física y preregistrada. El brazo original acabó con error 27,558233° y distancia 0,766791 mm; el brazo de mando crudo, con 24,275701° y 0,758525 mm. La mejora angular fue 3,282532° y superó el mínimo prospectivo de 0,5°; ambos cuerpos conservaron apoyo. El brazo crudo empeoró 4,861538° respecto al final del viento. El control de giro nulo de la campaña 41 acabó en 24,248712°, de modo que el mando crudo apenas cambió esa referencia. Esto justifica descartar el filtro actual como solución de orientación en esta vida; no prueba que el CNS sea incapaz de aprender, ni que quitar el filtro en una vida online reproduzca el mismo resultado.

La puntuación usa bearing de fuente fija, posición real del cuerpo y yaw de cuaternión. En la identidad, el bearing cambió −5,905064° y el yaw +2,239007° después del viento. Con orden cruda, el bearing cambió −4,792123° y el yaw +0,069415°. Es una descomposición cinemática de trayectorias distintas, no contribuciones causales aditivas. Retirar el avance en campaña 41 mejoró todavía más el error, pero detenerse no es una política de navegación.

## Qué decidir y qué no inferir

Las explicaciones fuertes «el CNS no recibió diferencia izquierda/derecha» y «toda la diferencia se canceló en el tiempo» quedan refutadas para la pareja espejo examinada. La discrepancia nativo/referencia comprueba concordancia numérica, no variabilidad entre semillas. Una diferencia entre fuentes tampoco prueba que el estado represente error angular. La hipótesis de que la señal llegada a DNb05 sea **insuficiente para esta demanda geométrica** sigue abierta. El replay sustituye la secuencia completa de salida crudo→aplicado y muestra un efecto físico adverso de ese bloque; no separa EMA, umbral y saturación entre sí, ni demuestra qué habría hecho el CNS bajo una nueva trayectoria. El caso sin viento permanece sin ejecutar; no se asigna todo el deterioro al pulso.

La decisión próxima debe distinguir estado neural posviento, interfaz de lectura y coordinación avance/giro bajo una entrada que cambie causalmente el lado del olor. Una prueba corta y emparejada puede aportar esa información. No se promueve lector, VNC ni marcha; el objetivo de fondo sigue siendo un sustrato general y la navegación es una presión diagnóstica local.
