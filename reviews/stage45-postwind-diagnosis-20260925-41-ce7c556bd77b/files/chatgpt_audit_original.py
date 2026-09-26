"""Auditoría cinemática: no carga campañas ni demuestra feedback neural.
Posiciones en metros, ángulos en radianes y tiempo físico en segundos.
Las muestras de pose deben describir el mismo punto/eje usados por el error.
"""
import numpy as np


def reloj(t, nombre: str = "tiempo") -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if (t.ndim != 1 or t.size < 2 or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0)):
        raise ValueError(f"{nombre}: exigir muestras finitas y crecientes")
    return t


def relojes(t_mj_s, t_neural_ms) -> dict:
    """Pares del MISMO evento; no emparejar por longitud ni corregir desfases."""
    p = reloj(t_mj_s, "MuJoCo")
    n = reloj(t_neural_ms, "neural") / 1000.0
    if p.shape != n.shape:
        raise ValueError("Falta correspondencia explícita entre eventos")
    x, y = p - p.mean(), n - n.mean()
    escala = float((x @ y) / (x @ x))
    offset = float(n.mean() - escala * p.mean())
    return dict(escala_neural_fisica=escala, offset_s=offset,
                residuo_max_s=float(np.max(np.abs(n - escala*p - offset))),
                deriva_s=float((n[-1]-n[0]) - (p[-1]-p[0])),
                dt_mj_min_max_s=(float(np.diff(p).min()),
                                 float(np.diff(p).max())))


def descomponer(t_s, xy_m, yaw_rad, fuente_xy_m, ventanas,
                error_firmado_rad=None, distancia_min_m: float = 1e-9) -> dict:
    """ventanas: [(inicio_s, fin_s), ...]. No usar error absoluto.
    unwrap exige muestreo suficiente; no puede detectar vueltas ocultas.
    """
    t = reloj(t_s)
    p, y, s = [np.asarray(a, dtype=float)
               for a in (xy_m, yaw_rad, fuente_xy_m)]
    if p.shape != (len(t), 2) or y.shape != t.shape or s.shape != (2,):
        raise ValueError("Formas requeridas: xy=(N,2), yaw=(N,), fuente=(2,)")
    if not all(np.isfinite(a).all() for a in (p, y, s)):
        raise ValueError("Pose/fuente no finita")
    if not np.isfinite(distancia_min_m) or distancia_min_m <= 0:
        raise ValueError("distancia_min_m debe ser positiva")
    r = s - p
    if np.any(np.linalg.norm(r, axis=1) <= distancia_min_m):
        raise ValueError("Bearing indefinido o mal condicionado junto a fuente")
    b = np.unwrap(np.arctan2(r[:, 1], r[:, 0]))
    y = np.unwrap(y)
    e = b - y
    e -= 2*np.pi*np.floor((e[0] + np.pi) / (2*np.pi))
    discrepancia = None
    if error_firmado_rad is not None:
        obs = np.asarray(error_firmado_rad, dtype=float)
        if obs.shape != t.shape or not np.isfinite(obs).all():
            raise ValueError("Error firmado incompatible")
        discrepancia = float(np.rad2deg(np.max(np.abs(
            np.angle(np.exp(1j*(obs-e)))))))
    filas = []
    for a, z in ventanas:
        if not (t[0] <= a < z <= t[-1]):
            raise ValueError("Ventana fuera del registro: no extrapolar")
        db, dy, de = [float(np.diff(np.interp([a, z], t, v))[0])
                      for v in (b, y, e)]
        filas.append(dict(inicio_s=a, fin_s=z, geometria_deg=np.rad2deg(db),
                          giro_deg=np.rad2deg(-dy), delta_error_deg=np.rad2deg(de),
                          bordes_interpolados=not (np.any(t == a) and np.any(t == z))))
    return dict(ventanas=filas, discrepancia_error_deg=discrepancia,
                alcance="Cinemática; no identifica feedback neural")


def integral_zoh(t_bordes_s, valor_intervalo, inicio: float, fin: float) -> float:
    """N-1 valores aplicados en [t[i],t[i+1]); NO muestras post-step."""
    t = reloj(t_bordes_s)
    u = np.asarray(valor_intervalo, dtype=float)
    if u.shape != (len(t)-1,) or not np.isfinite(u).all():
        raise ValueError("Exigir un valor finito por intervalo físico")
    if not (t[0] <= inicio < fin <= t[-1]):
        raise ValueError("Ventana inválida")
    dt = np.maximum(0.0, np.minimum(t[1:], fin) - np.maximum(t[:-1], inicio))
    return float(u @ dt)
