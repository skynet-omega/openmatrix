"""Explicit generalized PN state bridge for existing local receptor owners.

The common voltage/calcium state schema describes state layout, not fine-grid
fidelity. Identity retains the full G/M operator and mass integrator. The CNS
mass manifest supplies the coordinate contract and guarded projection recipe.
"""
from pn_mass_calcium import MassCalciumPN
from pn_calcium_coupled import CalciumCoupledFinePN


class MassPnSourceBridge(MassCalciumPN):
    def state_dict(self):
        return CalciumCoupledFinePN.state_dict(self)

    def load_state_dict(self, saved):
        return CalciumCoupledFinePN.load_state_dict(self, saved)
