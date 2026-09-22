"""Persist extra output tails around the same finePN/Ca and receptor source."""
import hashlib
import numpy as np
from pn_cholinergic_online_source import CholinergicOnlineSource,SCHEMA as PARENT_SCHEMA
from pn_electrical_output_port import AdditionalElectricalOutput

SCHEMA='PN_additional_electrical_output_source_v1'

class ElectricalOutputSource(CholinergicOnlineSource):
    @classmethod
    def adopt(cls,base,spec):
        if type(base) is not CholinergicOnlineSource:raise ValueError('Exact ACh source parent required')
        obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        obj.extra_output=AdditionalElectricalOutput(**spec);obj._cholinergic_identity=base.identity
        np.testing.assert_array_equal(obj.extra_output.route.site_ids,base.pn.calcium_port.chemistry.ids)
        if obj.extra_output.origin_ns>base.time_ns:raise ValueError('Future output origin')
        obj.identity=hashlib.sha256((SCHEMA+base.identity+obj.extra_output.identity).encode()).hexdigest()
        return obj

    def state_dict(self):
        base=CholinergicOnlineSource.state_dict(self);base['identity']=self._cholinergic_identity
        return dict(schema=SCHEMA,identity=self.identity,base=base,extra_output=self.extra_output.state_dict())

    def load_state_dict(self,s):
        if (set(s)!={'schema','identity','base','extra_output'} or s['schema']!=SCHEMA or s['identity']!=self.identity
            or s['base']['schema']!=PARENT_SCHEMA or s['base']['identity']!=self._cholinergic_identity
            or s['base']['base']['time_ns']!=s['extra_output']['time_ns']):raise ValueError('Wrong combined output/source clock or identity')
        self.extra_output.validate_state(s['extra_output'])
        chem=s['base']['base']['pn']['calcium']['chemistry'];tail=s['extra_output']
        generated=(np.asarray(chem['slow'])-np.asarray(chem['fast']))/self.extra_output.norm-(np.asarray(tail['slow'])-np.asarray(tail['fast']))/self.extra_output.norm
        if not np.isfinite(generated).all() or np.any(generated<0):raise ValueError('Saved additional output is below its inherited source tail')
        CholinergicOnlineSource.load_state_dict(self,dict(s['base'],identity=self.identity))
        self.extra_output.load_state_dict(s['extra_output']);self.additional_output_nS()

    def additional_output_nS(self):
        if self.extra_output.time_ns!=self.time_ns:raise ValueError('Partial output interval cannot be observed')
        return self.extra_output.output_nS(self.pn.calcium_port.chemistry.activation())

    def advance(self,dt_ns,before,after,**kwargs):
        old=self.state_dict()
        try:
            result=CholinergicOnlineSource.advance(self,dt_ns,before,after,**kwargs)
            if result['accepted']:
                self.extra_output.advance(dt_ns);self.additional_output_nS()
            return result
        except BaseException:self.load_state_dict(old);raise
