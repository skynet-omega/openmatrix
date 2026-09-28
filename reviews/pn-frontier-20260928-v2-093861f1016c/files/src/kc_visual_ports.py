"""Conservative visual-input relocation; retain six-component parent interface."""
import numpy as np


class VisualComponents(tuple):
    def __new__(cls,values,visual_ge,visual_gi,gamma_mask):
        if len(values)!=6:raise ValueError('Require six inherited conductance components')
        obj=super().__new__(cls,values)
        obj.visual_ge=np.asarray(visual_ge);obj.visual_gi=np.asarray(visual_gi);obj.gamma_mask=np.asarray(gamma_mask,dtype=bool)
        return obj

    def without_visual(self,remove_ge,remove_gi):
        """Diagnostic cuts remove both total current and its spatial label."""
        ge,gi,age,agi,pn,apl=self
        values=[ge-remove_ge,gi-remove_gi]
        visual=[self.visual_ge-remove_ge*self.gamma_mask,self.visual_gi-remove_gi*self.gamma_mask]
        if any(np.min(v)<-1e-12 for v in values+visual):raise ValueError('Visual cut exceeds tagged current')
        return type(self)((np.maximum(values[0],0.),np.maximum(values[1],0.),age,agi,pn,apl),
            np.maximum(visual[0],0.),np.maximum(visual[1],0.),self.gamma_mask)


class VisualKcInputs:
    def __init__(self,base,enabled):self.base=base;self.enabled=enabled
    def __getattr__(self,name):return getattr(self.base,name)
    def split(self,components,**kwargs):
        ge,gi,regional=self.base.split(components,**kwargs)
        if self.enabled:
            if not isinstance(components,VisualComponents):raise ValueError('Lost visual conductance labels')
            j=self.dynamic_index
            for domains,visual in [(ge,components.visual_ge[j]),(gi,components.visual_gi[j])]:
                total=domains.sum(axis=1)
                domains[:,0]-=visual;domains[:,1]+=visual
                if np.min(domains)<-1e-12:raise ValueError('Visual input exceeds soma proxy')
                np.maximum(domains,0.,out=domains)
                np.testing.assert_allclose(domains.sum(axis=1),total,atol=1e-13,rtol=1e-12)
        return ge,gi,regional
