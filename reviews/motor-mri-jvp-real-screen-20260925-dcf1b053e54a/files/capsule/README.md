# Bloque real: criba MRI por filas y JVP

Incluye código, planes, recibos y arrays numéricos de un bloque sham MaleCNS de 125 µs. La lectura principal está en [MOTOR_DECISION.md](review/MOTOR_DECISION.md). Los archivos `data/` permiten recalcular el defecto MRI muestreado, la semilla por filas y una prueba local de linealización del operador efectivo.

Desde esta carpeta extraída:

```bash
python3 code/verify_mri_jvp_capsule.py --root . --out verificacion.json
python3 -O code/verify_mri_jvp_capsule.py --root . --out verificacion_opt.json
```

Se necesita NumPy; no se necesita GPU para estos dos comandos. Los códigos de `probe_*` documentan cómo se ejecutaron las consultas sobre el organismo original, pero esta cápsula **no** contiene el organismo completo ni reconstruye el operador desde cero. Los recibos locales compararon la salida del organismo con el control bit a bit; esa comparación no se reejecuta en la cápsula portátil. Los costes por arista son cribas condicionales, no tiempos medidos de un nuevo motor.
