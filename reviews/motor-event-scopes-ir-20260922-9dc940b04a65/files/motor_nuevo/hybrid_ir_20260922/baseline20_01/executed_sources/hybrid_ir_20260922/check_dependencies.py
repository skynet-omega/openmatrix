"""Small declared-model fixtures; source models must expand intermediate reads."""
from pathlib import Path
import json,numpy as np
from dependency_ir import EventProgram,Read

def require(ok,message):
 if not ok:raise AssertionError(message)

def main():
 # q is prescribed and jumps. s is prescribed but continuous; x'=s-x.
 filtered=EventProgram(3,[Read('own',[0,1,2],[0,1,2]),Read('release',[0],[1]),Read('feedback',[1],[2])],[0,1])
 require(filtered.classify([0])['continuous_free_rhs'],'Filtered source classified incorrectly')
 # Same topology, but an endogenous filter s'=q-s now reads the discontinuity.
 live=EventProgram(3,filtered.reads,[0])
 require(live.classify([0])['affected_rows']==[1],'Live reader lost')
 # Another model with dense concentration coupling; no neural labels in core.
 chemistry=EventProgram(5,[Read('reaction',[0,1],[2,3,4],False)],[])
 require(chemistry.classify([1])['affected_rows']==[2,3,4],'Many-to-many dependency lost')
 require(chemistry.classify([4])['continuous_free_rhs'],'Spurious dependency')
 # Corruption controls must fail; a malformed declaration cannot authorize smoothing.
 rejected=0
 for bad in ([-1],[3],[0.5],[[0]]):
  try:filtered.classify(bad)
  except ValueError:rejected+=1
 require(rejected==4,'Invalid jump declaration accepted')
 require(filtered.identity()!=live.identity(),'Replaced derivative not in identity')
 result={'status':'PASS','models':3,'invalid_jump_controls':rejected,'live_reader_fallback':True,'continuous_source_history_retained':True,'scope':'Dependency semantics only, not convergence or proof of model declarations.'}
 (Path(__file__).resolve().parent/'DEPENDENCY_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
if __name__=='__main__':main()
