# Retirada de opción inválida, después de las cuatro cargas

La copia del runner aún ofrecía guard_nominal: instalaba NominalGraph, pero scoped_graph lo sustituía por NativeGraph. Esa opción no se usó en ninguna carga de esta ronda. Se retira para evitar una selección silenciosamente ignorada. Argparse la rechaza antes de cargar organismo o crear salida; cero cargas nuevas. Los cuatro brazos guard y sus fuentes ejecutadas conservadas no cambian. Esta corrección de interfaz no atribuye nuevos resultados numéricos al código editado.

```diff
--- executed/run_scoped.py
+++ current/run_scoped.py
@@ -10,16 +10,12 @@
 from runtime_session import RuntimeSession
 
 def main():
- p=argparse.ArgumentParser();p.add_argument('--variant',choices=['guard','fine1562','guard_nominal'],required=True);p.add_argument('--ms',type=int,choices=[20,50],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['baseline','scoped'],required=True);a=p.parse_args()
+ p=argparse.ArgumentParser();p.add_argument('--variant',choices=['guard','fine1562'],required=True);p.add_argument('--ms',type=int,choices=[20,50],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['baseline','scoped'],required=True);a=p.parse_args()
  RunStorage(a.out);obj=session=None;undo_metadata=undo_ports=undo_graph=None;start=time.perf_counter();rows=[];times=[];status='INCOMPLETE';error=None
  try:
   import cupy as cp,pandas as pd
   obj,d,plan,Field,field,ports=load(a.out/'inputs');b=obj.core.hybrid
   # Preserve125us coupling; only the prospective scoped arm may remove global cuts.
-  if a.variant=='guard_nominal':
-   import organism_adapter
-   from nominal_policy import NominalGraph
-   organism_adapter.NativeGraph=NominalGraph
   import reset_adapter
   undo_ports=reset_adapter.prepare()
   session=RuntimeSession(b,'causal_cuda');undo_metadata=reset_adapter.attach(session)
```
