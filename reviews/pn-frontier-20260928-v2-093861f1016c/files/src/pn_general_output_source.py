"""General consumer tails around the same single PN/Ca source."""
import hashlib
import numpy as np
from pn_electrical_output_source import ElectricalOutputSource, SCHEMA as BASE_SCHEMA
from pn_general_output_port import GeneralOutputPort

SCHEMA = 'PN_general_output_source_v1'


class GeneralOutputSource(ElectricalOutputSource):
    @classmethod
    def adopt(cls, base, spec):
        if type(base) is not ElectricalOutputSource:
            raise ValueError('Exact electrical-output source required')
        obj = cls.__new__(cls); obj.__dict__.update(base.__dict__)
        obj.general_output = GeneralOutputPort(**spec)
        np.testing.assert_array_equal(obj.general_output.site_ids, base.pn.calcium_port.chemistry.ids)
        if obj.general_output.origin_ns > base.time_ns:
            raise ValueError('Future general output origin')
        obj._electrical_identity = base.identity
        obj.identity = hashlib.sha256((SCHEMA + base.identity + obj.general_output.identity).encode()).hexdigest()
        return obj

    def state_dict(self):
        base = ElectricalOutputSource.state_dict(self)
        base['identity'] = self._electrical_identity
        return dict(schema=SCHEMA, identity=self.identity, base=base,
                    general_output=self.general_output.state_dict())

    def load_state_dict(self, state):
        if (set(state) != {'schema', 'identity', 'base', 'general_output'}
                or state['schema'] != SCHEMA or state['identity'] != self.identity
                or state['base']['schema'] != BASE_SCHEMA
                or state['base']['identity'] != self._electrical_identity
                or state['base']['extra_output']['time_ns'] != state['general_output']['time_ns']):
            raise ValueError('Wrong general source identity or clock')
        self.general_output.validate_state(state['general_output'])
        chem = state['base']['base']['base']['pn']['calcium']['chemistry']
        tail = state['general_output']
        generated = ((np.asarray(chem['slow']) - np.asarray(chem['fast'])) -
                     (np.asarray(tail['slow']) - np.asarray(tail['fast']))) / self.general_output.norm
        if not np.isfinite(generated).all() or np.any(generated < 0):
            raise ValueError('General output below inherited source tail')
        ElectricalOutputSource.load_state_dict(self, dict(state['base'], identity=self.identity))
        self.general_output.load_state_dict(state['general_output'])
        self.general_transmission()

    def general_transmission(self):
        if self.general_output.time_ns != self.time_ns:
            raise ValueError('Partial general output interval cannot be observed')
        return self.general_output.output(self.pn.calcium_port.chemistry.activation())

    def advance(self, dt_ns, before, after, **kwargs):
        old = self.state_dict()
        try:
            result = ElectricalOutputSource.advance(self, dt_ns, before, after, **kwargs)
            if result['accepted']:
                self.general_output.advance(dt_ns)
                self.general_transmission()
            return result
        except BaseException:
            self.load_state_dict(old)
            raise
