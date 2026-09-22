"""Development fixtures, explicitly mathematical engineering models, not fitted brains."""
import copy

def scalar(unit,value): return {'unit':unit,'value':value}
def state(unit,initial,scale,rhs): return {'unit':unit,'initial':initial,'scale':scale,'rhs':rhs}

def base(n=32,degree=8):
    return {'version':1,'description':'Synthetic recurrent rate population, dimensionless, not a biological reconstruction',
            'populations':[{'id':'cells','count':n,'states':{
                'v':state('1',.1,1,'(-v+w*syn+drive)/tau'),
                'syn':state('1',.05,1,'(tanh(v)-syn)/taus')},
                'parameters':{'tau':scalar('s',.02),'taus':scalar('s',.01),'drive':scalar('1',.1)},
                'inputs':{'w':scalar('1',0)},'outputs':{'release':{'unit':'1','expr':'syn'}}}],
            'connections':[{'source':['cells','release'],'target':['cells','w'],'pattern':'ring','offsets':list(range(1,degree+1)),'weight':.2/degree}]}

def hh(n=4):
    # Classical HH current balance, milliseconds converted explicitly to seconds.
    # hh.m/vtrap is expressed by the generic exprel primitive (analytic removable singularity).
    p={'id':'axon','count':n,'states':{
        'v':state('mV',-65,100,'(current-gna*m**3*h*(v-ena)-gk*q**4*(v-ek)-gl*(v-el))/cm'),
        'm':state('1',.0529,1,'am*(1-m)-bm*m'),
        'h':state('1',.5961,1,'ah*(1-h)-bh*h'),
        'q':state('1',.3177,1,'aq*(1-q)-bq*q')},
       'parameters':{'cm':scalar('s',.001),'gna':scalar('1',120),'gk':scalar('1',36),'gl':scalar('1',.3),
        'ena':scalar('mV',50),'ek':scalar('mV',-77),'el':scalar('mV',-54.387),
        'v40':scalar('mV',40),'v65':scalar('mV',65),'v35':scalar('mV',35),'v55':scalar('mV',55),
        'v10':scalar('mV',10),'v18':scalar('mV',18),'v20':scalar('mV',20),'v80':scalar('mV',80),
        'r1':scalar('1/s',1000),'r4':scalar('1/s',4000),'r07':scalar('1/s',70),'r01':scalar('1/s',100),'r0125':scalar('1/s',125)},
       'inputs':{'current':scalar('mV',10)},'derived':{
        'am':{'unit':'1/s','expr':'r1/exprel(-(v+v40)/v10)'},
        'bm':{'unit':'1/s','expr':'r4*exp(-(v+v65)/v18)'},
        'ah':{'unit':'1/s','expr':'r07*exp(-(v+v65)/v20)'},
        'bh':{'unit':'1/s','expr':'r1/(1+exp(-(v+v35)/v10))'},
        'aq':{'unit':'1/s','expr':'r01/exprel(-(v+v55)/v10)'},
        'bq':{'unit':'1/s','expr':'r0125*exp(-(v+v65)/v80)'}},
       'outputs':{'voltage':{'unit':'mV','expr':'v'}}}
    return {'version':1,'description':'Classical HH axon engineering fixture, not measured other-species connectome','populations':[p],'connections':[]}
