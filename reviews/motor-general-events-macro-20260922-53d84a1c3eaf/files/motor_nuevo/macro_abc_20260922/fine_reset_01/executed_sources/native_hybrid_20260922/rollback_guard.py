"""Fail closed after a strict physical rollback replaces captured GPU owners.

The whole runtime must be torn down/reinstalled before retry. Private graph
rollback alone cannot rebind every inherited membrane, port and body owner.
"""
import types

def install(brain):
 old_restore=brain._restore_joint
 had_advance='advance' in brain.__dict__;old_advance=brain.__dict__.get('advance')
 def invalid(*args,**kw):raise RuntimeError('Physical rollback replaced execution owners; rebuild native runtime before retry')
 def restore(self,*args,**kw):
  try:return old_restore(*args,**kw)
  finally:self.advance=invalid;self._native_rebuild_required=True
 brain._restore_joint=types.MethodType(restore,brain)
 def undo():
  brain._restore_joint=old_restore
  if brain.__dict__.get('advance') is invalid:
   if had_advance:brain.advance=old_advance
   else:del brain.advance
  brain.__dict__.pop('_native_rebuild_required',None)
 return undo
