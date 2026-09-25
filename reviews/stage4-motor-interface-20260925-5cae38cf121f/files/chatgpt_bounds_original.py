from __future__ import annotations

import numpy as np


def cota_monotona(
    valores: np.ndarray,
    pesos: np.ndarray,
    tope: float = 5.0,
) -> dict[str, float]:
    """Cotas de sum(pesos * f(valores)) para UN MISMO lector f.

    f es estático, impar, monótono no decreciente y |f| <= tope.
    Con pesos en segundos y tope en grados/s, devuelve grados de mando.

    Individual:
        cota_monotona(x, dt)

    Pareada, sinD menos D, usando el MISMO lector en ambos brazos:
        cota_monotona(
            np.concatenate((x_sinD, x_D)),
            np.concatenate((dt_sinD, -dt_D)),
        )

    Agrupa empates exactos de |x|, incluidos los compartidos entre brazos.
    Los ceros no contribuyen porque la imparidad exige f(0) = 0.
    Son cotas sobre señales congeladas, no sobre giro corporal.
    """
    valores = np.asarray(valores, dtype=np.float64)
    pesos = np.asarray(pesos, dtype=np.float64)

    if not (
        valores.ndim == 1
        and valores.shape == pesos.shape
        and valores.size > 0
    ):
        raise ValueError(
            "Valores y pesos deben ser vectores no vacíos del mismo tamaño."
        )
    if not (np.isfinite(valores).all() and np.isfinite(pesos).all()):
        raise ValueError("Valores o pesos no finitos.")
    if not (np.isfinite(tope) and tope > 0):
        raise ValueError("Tope inválido.")

    activos = valores != 0
    if not np.any(activos):
        return {
            "min_grados_mando": 0.0,
            "max_grados_mando": 0.0,
        }

    x = valores[activos]
    w = pesos[activos]

    # np.unique ordena las amplitudes y asigna el mismo grupo a cada empate.
    _, grupos = np.unique(np.abs(x), return_inverse=True)
    contribuciones = np.bincount(
        grupos,
        weights=w * np.sign(x),
    )

    # Cada cola suma pesos * signo(x) para |x| >= su amplitud.
    colas = np.cumsum(contribuciones[::-1], dtype=np.float64)[::-1]

    # Incluir cero corresponde al lector admisible f(x) = 0.
    return {
        "min_grados_mando": float(tope * min(0.0, float(colas.min()))),
        "max_grados_mando": float(tope * max(0.0, float(colas.max()))),
    }
