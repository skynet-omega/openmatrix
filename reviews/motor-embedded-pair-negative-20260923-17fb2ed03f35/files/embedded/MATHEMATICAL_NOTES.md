# Dos evaluaciones y actualización espacial: alcance matemático

Derivación deCodex sobre la interfaz efectiva `y'=b(t,y)⊙(a(t,y)−y)`. Es análisis de métodos conocidos, no resultado experimental ni método nuevo. La revisión primaria de [Hochbruck yOstermann,2010](https://na.math.kit.edu/download/papers/acta-final.pdf) distingue orden clásico y orden rígido; su análisis no convierte automáticamente este operador biológico compuesto en un problema semilineal con matriz constante.

## B: par Euler/punto medio exponencial

En un segmento sin eventos interiores, sean `a0,b0` los coeficientes iniciales y `f0=b0⊙(a0−y)`:

```
yE = y + (1−exp(−h b0)) ⊙ (a0−y)
yM = y + (1−exp(−h b0/2)) ⊙ (a0−y)
aM,bM = coefficients(t+h/2, yM)
y2 = y + (1−exp(−h bM)) ⊙ (aM−y)
```

Para coeficientes suaves, `yM=y+h f0/2+O(h²)`. La expansión de `y2` da `y+h f0+h²(f_t+Df·f0)/2+O(h³)`: orden clásico2 local. Las tasas variables se reevalúan; no se congelan durante todo el método. Esto no prueba orden rígido uniforme ni una cota global del organismo. Los puertos prescritos deben proyectarse al inicio, mitad y final con sus tiempos correctos; los saltos siguen siendo fronteras obligatorias.

Si cada tasa es no negativa, los estados iniciales y cada objetivo evaluado están dentro de una caja, y la proyección respeta esa misma caja, ambas actualizaciones son combinaciones convexas: conservan la caja en aritmética exacta. No asumir esas premisas para todo plugin; comprobar valores y dominio sigue siendo necesario. No corregir una salida inválida por clipping del integrador.

`y2−yE` estima una diferencia de orden2 del método inferior, no el error de orden3 de la solución aceptada del padre. No conserva su divisor3. La pareja puede requerir muchos más intentos: con dos evaluaciones sólo ahorra frente a seis si el aumento de intentos y controles no consume la ventaja. Como ambos métodos son exactos con coeficientes constantes, no basta probar ese caso para medir utilidad. Las primeras mediciones deben usar los coeficientes variables y proyecciones reales. Ningún error antiguo autoriza aceptar un paso nuevo.

## C: una cota espacial requiere acoplamiento

En el bloque gradual simplificado `q'=Dτ(T(Ws+d)−q)`, `s'=Ds(q−s)`, un cambio pequeño en cada `s` puede acumularse por muchas aristas. Una cota inicial necesita `|W|·εs` y una cota de la derivada deT, además de la recurrencia posterior. Con cotas válidas de esas derivadas, un sistema comparador de errores tiene bloques `[-Dτ, Dτ·L_T·|W|; Ds, −Ds]`. La diagonal negativa de relajación debe conservarse; sustituir todo por una norma escalar puede destruir la utilidad de la cota.

Ese comparador no cubre automáticamente tasas dependientes del estado, receptores, PN espaciales o eventos: cada extensión necesita términos/cotas de su contrato. Una derivada calculada sólo en el estado nominal no es una cota de intervalo. Si falta una cota, se usa el avance completo. Los128.261 cambios observados en transmisión descartan justificar omisiones sólo con un porcentaje supuesto de neuronas que disparan; no descartan un método aproximado con influencia acotada.

ParaA, B yC la decisión sigue siendo empírica: coste del organismo con la misma consulta, error observado/refinamiento, eventos, dominio, rollback y continuidad. Las fuentes y parámetros de la confirmación16 permanecen congelados.
