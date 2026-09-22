"""New model descriptions created AFTER CORE_FREEZE. Never imports CUDA or an integrator."""
import numpy as np
from development import base,scalar,state


def fixture(counts=(12,12,12),seed=0):
    nr,nv,nc=counts;rng=np.random.default_rng(seed)
    spec=base(nr,min(nr,24));rate=spec['populations'][0]
    rate['states']['v']['initial']=(.1+.03*np.sin(np.arange(nr)*.17)).tolist()
    visual={'id':'transducer','count':nv,'states':{},'parameters':{'omega':scalar('1/s',25),'tau':scalar('s',.006)},
            'inputs':{'light':scalar('1',.4)},'outputs':{'signal':{'unit':'1','expr':'p6'}}}
    for i in range(7):visual['states'][f'p{i}']=state('1',.04,1,('(light+0.1*sin(omega*t)-p0)' if i==0 else f'(p{i-1}-p{i})')+'/tau')
    chem={'id':'chemistry','count':nc,'states':{},'parameters':{'k':scalar('1/s',12),'cref':scalar('mM',1)},
          'inputs':{'modulator':scalar('1',0)},'outputs':{'signal':{'unit':'1','expr':'c0/cref'}}}
    for i in range(12):
        chem['states'][f'c{i}']=state('mM',.5 if i==0 else .5/11,1,f'k*(1+modulator)*(c{(i-1)%12}-c{i})')
    spec['populations'] += [visual,chem]
    def connect(src,sn,target,tn,port,weight):
        rows=np.arange(tn);cols=(rows*7+seed)%sn
        return {'source':[src,'release' if src=='cells' else 'signal'],'target':[target,port],'pattern':'coo','row':rows.tolist(),'col':cols.tolist(),'weights':weight}
    spec['connections'] += [connect('cells',nr,'transducer',nv,'light',.25),connect('transducer',nv,'cells',nr,'w',.15),
                           connect('cells',nr,'chemistry',nc,'modulator',.1),connect('chemistry',nc,'cells',nr,'w',.05)]
    # All three populations participate in a closed, continuously evaluated loop.
    spec['description']='Synthetic mixed ODE: recurrent network + seven-state transducer + conserved twelve-species chemistry. No species-specific fit, no complete brain, no spiking/delay/body.'
    return spec
