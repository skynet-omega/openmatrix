from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FRACCION_RANGO = 0.25
INICIO_NS = 20_000_000
FIN_NS = 60_000_000

OBJETIVOS = {
    "DNp09_bilateral": (10783, 11177),
    "DNa03_izquierda": (519624,),
    "DNa03_derecha": (10975,),
}


def exigir(condicion: bool, mensaje: str) -> None:
    if not condicion:
        raise ValueError(mensaje)


def vector_real(valor: Any, nombre: str) -> np.ndarray:
    salida = np.asarray(valor, dtype=np.float64)
    exigir(
        salida.ndim == 1 and salida.size > 0,
        f"{nombre}: vector vacío o forma inválida",
    )
    exigir(
        bool(np.isfinite(salida).all()),
        f"{nombre}: valores no finitos",
    )
    return salida


@dataclass(frozen=True)
class Pulso:
    ids: tuple[int, ...]
    corriente_adicional: np.ndarray
    target_inicial: np.ndarray
    target_objetivo_instantaneo: np.ndarray


def preparar_pulso(
    ids: tuple[int, ...],
    corriente_total_previa: np.ndarray,
    umbral: np.ndarray,
    ganancia: np.ndarray,
) -> Pulso:
    """I incluye sinapsis y drive preexistente, pero NO resta theta.

    Los vectores contienen solo las células de intervención, en orden de ids.
    El objetivo se refiere al instante inicial con entrada de red congelada;
    no garantiza target constante tras cambiar la actividad recurrente.
    El snapshot debe corresponder al dueño nativo del estado, no a tasas
    publicadas float32 ni a una media de un milisegundo.
    """
    corriente = vector_real(corriente_total_previa, "corriente_total_previa")
    theta = vector_real(umbral, "umbral")
    gain = vector_real(ganancia, "ganancia")

    exigir(
        corriente.shape == theta.shape == gain.shape,
        "Formas incompatibles",
    )
    exigir(
        len(ids) == corriente.size and len(set(ids)) == len(ids),
        "Identidades inválidas",
    )
    exigir(
        all(
            isinstance(i, (int, np.integer)) and not isinstance(i, bool)
            for i in ids
        ),
        "IDs no enteros",
    )
    exigir(
        bool((gain > 0).all()),
        "Se requiere ganancia positiva en esta ley no visual",
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        argumento = gain * (corriente - theta)
        inicial = np.maximum(0.0, np.tanh(argumento))
        objetivo = inicial + FRACCION_RANGO * (1.0 - inicial)

        exigir(
            bool((objetivo < 1.0).all()),
            "Sin rango resoluble: target saturado; no aumentar dosis",
        )

        adicional = (np.arctanh(objetivo) - argumento) / gain
        reconstruido = np.maximum(
            0.0,
            np.tanh(gain * ((corriente + adicional) - theta)),
        )

    exigir(
        bool(np.isfinite(adicional).all() and (adicional > 0).all()),
        "Pulso no resoluble",
    )
    exigir(
        bool(np.allclose(reconstruido, objetivo, rtol=1e-10, atol=1e-12)),
        "Cancelación numérica en unidades de entrada: no inyectar",
    )

    arreglos = [adicional.copy(), inicial.copy(), objetivo.copy()]
    for arreglo in arreglos:
        arreglo.setflags(write=False)

    return Pulso(tuple(int(i) for i in ids), *arreglos)


def corriente_en_intervalo(
    pulso: Pulso,
    inicio_ns: int,
    fin_ns: int,
) -> np.ndarray:
    """Señal constante en un intervalo aceptable; t=0 es inicio del ensayo.

    El host debe hacer coincidir los eventos con fronteras del integrador.
    Devuelve copia: no modifica ni sustituye el drive original del host.
    """
    exigir(
        isinstance(inicio_ns, (int, np.integer))
        and isinstance(fin_ns, (int, np.integer)),
        "Se requieren relojes enteros en nanosegundos",
    )
    exigir(0 <= inicio_ns < fin_ns, "Intervalo inválido")
    exigir(
        not any(
            inicio_ns < frontera < fin_ns
            for frontera in (INICIO_NS, FIN_NS)
        ),
        "Dividir el intervalo en la frontera del pulso",
    )

    if INICIO_NS <= inicio_ns and fin_ns <= FIN_NS:
        return pulso.corriente_adicional.copy()

    return np.zeros_like(pulso.corriente_adicional)
