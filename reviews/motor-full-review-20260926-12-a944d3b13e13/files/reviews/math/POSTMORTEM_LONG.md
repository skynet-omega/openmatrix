# Diagnóstico de la pareja continua de 2 s

**Se mantiene FAIL.** El único desacuerdo de mando aplicado está en el **paso 1911**, intervalo simulado **1910–1911 ms**. La causa inmediata comprobable es el umbral discontinuo del motor: las dos trayectorias neuronales dejan el filtro a lados opuestos de `0.0005 rad/s`. La reconstrucción desde las entradas guardadas reproduce exactamente ambos motores. No se encontró un defecto de implementación del filtro; los datos no permiten atribuir de manera exclusiva la diferencia neuronal anterior a RK3(2), FP32 o la discretización de eventos, ni descartar todos los defectos del núcleo.

Revisión CPU acotada a datos guardados y fuentes congeladas. No se ejecutaron GPU, organismos ni nuevas vidas; no se cambiaron fuentes, verificadores, parámetros o criterios. La recurrencia escalar que se evalúa abajo sólo reconstruye observaciones existentes.

## Localización y cadena causal

El contrato está implementado en [motor_wind.py:50](/home/daroch/AXIOMA_ASTRA/campanas/etapa45_navigation_wind_20260925_40/motor_wind.py:50):

`d = DN_q_usada − DN_baseline`; `raw = tanh(250*(d[2]−d[3]))*radians(5)`; `f ← f + α*(raw−f)`, con `α = −expm1(−0.001/0.2) = 0.004987520807317687`. Aplica `copysign(radians(5),f)` cuando `abs(f) >= 0.0005`; en otro caso, cero. `f` comienza en cero. [motor_wind_fixed.py:19](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/motor_wind_fixed.py:19) modifica el mapeo del viento y hereda esta misma regla.

| Paso | Filtro estable, rad/s | Filtro revisado, rad/s | Yaw estable/revisado, °/s |
|---|---:|---:|---:|
| 1910 | 0.0005195256222131168 | 0.0005204352207133743 | +5 / +5 |
| **1911** | **0.0004995591588058994** | **0.0005004683573549568** | **0 / +5** |
| 1912 | 0.00047964618695625107 | 0.00048055503579637614 | 0 / 0 |

Los márgenes respecto al umbral en 1911 son **−4.4084119410058454e−7** y **+4.6835735495682e−7 rad/s**. No es una ambigüedad de un ULP en la comparación: `ulp(0.0005) = 1.0842021724855044e−19`. La diferencia entre filtros es `9.091985490574045e−7 rad/s`; el salto de salida es `0.08726646259971647 rad/s` durante 1 ms.

En ese paso, revisado menos estable de `DN_q_usada[2:4]` es `[3.672216553063379e−8, −1.3557718281376197e−9]`. Eso produce una diferencia de `raw = 8.294081179928935e−7 rad/s`. La diferencia del filtro procede principalmente de su historia: `(1−α)*Δf_1910 = 9.050618588111535e−7`, mientras `α*Δraw_1911 = 4.13669024624776e−9`. No es un error puntual de lectura del DN en 1911.

- DN posterior ya difiere en el paso 1; DN usado, `raw` y filtro difieren desde el paso 2. El desfase de una muestra está respetado.
- Se reconstruyeron los **2000 pasos de cada brazo** desde DN usado y baseline, conservando el orden de operaciones del código. `raw`, filtro, salida umbralizada y avance coinciden **exactamente**, error máximo cero. `command_yaw_rate_rad_s` coincide en cada brazo con su salida umbralizada. Las 21 transiciones de mando coinciden salvo el apagado de +5°/s, que ocurre en 1911 y 1912 respectivamente.
- `qpos`, `qvel`, posición y yaw son idénticos hasta 1910 inclusive y divergen en 1911. Los sensores pendientes divergen en 1911 y los usados en 1912, como exige el acoplamiento. Esto localiza la primera bifurcación física después del desacuerdo de mando.
- Los 800 subpasos del viento, sus poses y fuerzas generalizadas son idénticos entre brazos. El pulso de pasos 1001–1020 no presenta un desacuerdo de implementación que explique este FAIL.

La integral del mando adicional es `5°/s * 0.001 s = 0.005°`. El máximo desacuerdo de yaw observado es `0.004999941139573139°`; al final es `0.004978614882645616°`. La posición difiere como máximo `2.910327318406125e−5 mm`; contactos y avance siguen siendo idénticos. Estas diferencias pequeñas satisfacen sus respectivos límites, pero no anulan el requisito independiente de mando idéntico cada tick.

## Qué permite concluir

La discontinuidad del lector motor convierte diferencias neuronales pequeñas y persistentes en un tick de acción distinto. El filtro previo no muestra un fallo ni una amplificación inestable: reproduce literalmente la recurrencia y su diferencia máxima (`9.27780900770005e−7 rad/s`) es menor que la de `raw` (`1.159292670876392e−6 rad/s`). La diferencia neuronal tampoco nace del feedback corporal: existe mucho antes de 1911. **Después de 1911 los brazos reciben entradas sensoriales diferentes**, por lo que las diferencias neuronales posteriores ya incluyen ese efecto del circuito cerrado.

[RUNTIME_PAIR2000.json](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/RUNTIME_PAIR2000.json) acredita los ejecutores previstos, intercambio de 125 µs, restauración y contabilidad. No certifica convergencia ni equivalencia neuronal. Los eventos confirmados tienen igual cuenta total, pero difieren las identidades presentes en 12 bloques; los predictores difieren en 15 bloques. Esto tampoco identifica por sí solo un defecto de implementación o el origen dominante del sesgo DN.

**Recomendación acotada:** cerrar esta confirmación con el FAIL funcional de [PAIR2000.json](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/PAIR2000.json). Conservar por separado la aceleración de pared **1.912718×** (6241.904 s frente a 3263.368 s), sin promover la candidata como equivalente bajo este contrato. Los registros no justifican una reparación concreta del núcleo ni mover umbral, ganancia, constante del filtro o criterio para rescatar la vida. Una futura atribución del error requiere separar precisión e integrador y contrastar convergencia sobre entradas comunes; queda fuera de este diagnóstico. El paso 1911 ya está expuesto y no puede usarse para ajustar parámetros y luego llamarlo confirmación independiente.

## Evidencia y reproducción de la comprobación escalar

Los hashes de ambas trazas coinciden con los de `PAIR2000.json`. Se verificó también la identidad congelada de `motor_wind.py`, `motor_wind_fixed.py`, `gaussiano_400.py`, runner, PLAN y verificador.

| Archivo | SHA256 |
|---|---|
| SOURCES.json | `0ee67395796b3a3399581a42667e5f1130d8744499124725e7f25bca5bdd5bea` |
| PAIR2000.json | `245f4e5fc1b03c4637b4721d89bc113368ec7fa6534ead0923407273f2334b6e` |
| RUNTIME_PAIR2000.json | `3e421aeea6db813cd9248540797d83bad11e86c0891b134e938c91fa6e9e1dbd` |
| stable_2000ms_01/traces.npz | `63d82665862fb94c2cd0c8045929db59d5cc90835332e9fc5d2528d0e016ec14` |
| reviewed_2000ms_01/traces.npz | `35dc06d816051d00f0da0441aecf666807116ed51631b933c20f74bef9cba1ce` |

Sólo lee datos y reconstruye el motor escalar; no importa el organismo:

```bash
cd /home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12
/home/daroch/miniconda3/envs/GPU/bin/python -B - <<'PY'
import math
import numpy as np
alpha = -math.expm1(-.001/.2)
amp = math.radians(5.)
for engine in ('stable', 'reviewed'):
    with np.load(f'{engine}_2000ms_01/traces.npz', allow_pickle=False) as z:
        f = 0.; values = []
        for q, baseline in zip(z['DN_q_usada'], z['DN_baseline']):
            d = np.asarray(q-baseline, dtype=float)
            raw = float(np.tanh(250.*(d[2]-d[3]))*amp)
            f += alpha*(raw-f)
            applied = math.copysign(amp,f) if abs(f)>=.0005 else 0.
            forward = float(np.clip(.2+np.mean(d[:2]),0.,.5))
            values.append([raw, f, applied, forward])
        fields = ('neural_command_raw_rad_s','motor_filter_state_rad_s',
                  'motor_filter_applied_rad_s','command_forward_mm_s')
        for i, key in enumerate(fields):
            if not np.array_equal(np.asarray(values)[:,i],z[key]):
                raise ValueError(engine+': reconstruction differs '+key)
        if not np.array_equal(z['command_yaw_rate_rad_s'],z[fields[2]]):
            raise ValueError(engine+': applied command differs')
        print(engine, 'exact reconstruction, 2000 steps', values[1910])
PY
```
