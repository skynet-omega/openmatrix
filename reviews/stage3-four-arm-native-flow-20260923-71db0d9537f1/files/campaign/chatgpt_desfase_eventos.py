"""Lectura de EVENT_AUDIT del snapshot 2ed2781; no simulador.
Emparejamiento ordinal por neurona, no estimación de latencia biológica.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

BLOQUE_NS = 125000

def exigir(condicion, mensaje):
    if not condicion:
        raise ValueError(mensaje)

def cargar(ruta):
    contenido = ruta.read_bytes()
    datos = json.loads(contenido)
    eventos = defaultdict(list)
    bloques = []
    predictores = 0
    cuantizacion = 0.0
    for bloque in datos["blocks"]:
        duracion = bloque["duration_ns"]
        exigir(duracion in (BLOQUE_NS // 2, BLOQUE_NS),
               "Duración ajena al contrato revisado")
        if duracion == BLOQUE_NS // 2:
            predictores += len(bloque["events"])
            continue
        inicio = bloque["start_elapsed_ns"]
        exigir(type(inicio) is int and inicio >= 0, "Reloj inválido")
        bloques.append(inicio)
        for e in bloque["events"]:
            fila, identidad = e["row"], e["neuron_id"]
            exigir(type(fila) is int and type(identidad) is int,
                   "Fila/ID deben ser enteros")
            exigir(fila >= 0 and identidad >= 0, "Fila/ID negativos")
            tiempo = float(e["time_s"]) * 1e9
            exigir(math.isfinite(tiempo) and 0 <= tiempo <= duracion,
                   "Evento fuera del bloque")
            redondeado = round(tiempo)
            cuantizacion = max(cuantizacion, abs(tiempo-redondeado))
            post = e["post_q"]
            exigir(post is None or math.isfinite(float(post)),
                   "Post no finito")
            clave = (e["producer"], fila, identidad)
            eventos[clave].append((inicio+redondeado, post))
    exigir(bloques and all(a < b for a, b in zip(bloques, bloques[1:])),
           "Bloques aceptados vacíos, duplicados o desordenados")
    for lista in eventos.values():
        lista.sort(key=lambda x: x[0])  # Conserva orden en marcas iguales.
    return eventos, bloques, {
        "sha256": hashlib.sha256(contenido).hexdigest(),
        "eventos_predictores": predictores,
        "eventos_aceptados": sum(map(len, eventos.values())),
        "redondeo_max_ns": cuantizacion,
    }

def comparar(ruta_a, ruta_b):
    a, bloques_a, meta_a = cargar(ruta_a)
    b, bloques_b, meta_b = cargar(ruta_b)
    exigir(bloques_a == bloques_b, "Cronología de bloques distinta")
    diferencias, tiempos, posts = [], [], []
    peor = None
    for clave in sorted(set(a) | set(b)):
        aa, bb = a.get(clave, []), b.get(clave, [])
        if len(aa) != len(bb):
            diferencias.append({
                "productor_fila_id": list(clave),
                "n_A": len(aa), "n_B": len(bb)
            })
            continue
        for orden, ((ta, pa), (tb, pb)) in enumerate(zip(aa, bb)):
            delta = abs(ta-tb)
            tiempos.append(delta)
            if pa is not None and pb is not None:
                posts.append(abs(float(pa)-float(pb)))
            if peor is None or delta > peor["desfase_ns"]:
                peor = {
                    "productor_fila_id": list(clave),
                    "orden": orden, "t_A_ns": ta, "t_B_ns": tb,
                    "desfase_ns": delta
                }
    return {
        "A": meta_a, "B": meta_b,
        "diferencias_recuento": diferencias,
        "eventos_emparejados": len(tiempos),
        "max_desfase_ns": max(tiempos) if tiempos else None,
        "max_diferencia_post": max(posts) if posts else None,
        "peor": peor,
        "alcance": "Desfase numérico de eventos; no latencia de circuito.",
        "admisión_etapa3": False
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("causal", type=Path)
    p.add_argument("referencia", type=Path)
    args = p.parse_args()
    print(json.dumps(comparar(args.causal, args.referencia),
                     indent=2, ensure_ascii=False, allow_nan=False))
