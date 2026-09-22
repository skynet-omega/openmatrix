"""Declarative smooth ODE IR. No anatomical dispatch and no arbitrary code execution."""
from __future__ import annotations
import ast, copy, hashlib, json, math, re
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.special import exprel

class Unsupported(ValueError): pass

def require(ok, message):
    if not ok: raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

FUNCS = {'exp': np.exp, 'expm1': np.expm1, 'exprel': exprel, 'log': np.log,
         'tanh': np.tanh, 'sin': np.sin, 'cos': np.cos, 'sqrt': np.sqrt}
IDENT = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')

def combine(a, b, sign=1):
    result = dict(a)
    for key, val in b.items(): result[key] = result.get(key, 0) + sign*val
    return {k:v for k,v in result.items() if v != 0}

def dimension(node, env):
    if isinstance(node, ast.Constant):
        require(type(node.value) in (int,float) and math.isfinite(node.value), 'finite numeric literals only')
        return {}
    if isinstance(node, ast.Name):
        require(node.id in env, 'unknown symbol: '+node.id)
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd,ast.USub)):
        return dimension(node.operand, env)
    if isinstance(node, ast.BinOp):
        a,b=dimension(node.left,env),dimension(node.right,env)
        if isinstance(node.op,(ast.Add,ast.Sub)):
            # Literal zero is valid in any physical unit.
            if isinstance(node.left,ast.Constant) and node.left.value==0: return b
            if isinstance(node.right,ast.Constant) and node.right.value==0: return a
            require(a==b, f'incompatible dimensions in addition: {a}, {b}')
            return a
        if isinstance(node.op,ast.Mult): return combine(a,b)
        if isinstance(node.op,ast.Div): return combine(a,b,-1)
        if isinstance(node.op,ast.Pow):
            require(isinstance(node.right,ast.Constant) and not b, 'constant numeric power only')
            return {k:v*node.right.value for k,v in a.items() if v*node.right.value!=0}
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in FUNCS:
        require(len(node.args)==1 and not node.keywords, 'unary mathematical functions only')
        a=dimension(node.args[0],env)
        if node.func.id=='sqrt': return {k:v/2 for k,v in a.items()}
        require(not a, 'transcendental function requires dimensionless argument')
        return {}
    raise ValueError('unsupported expression syntax: '+ast.dump(node)[:100])

def unit(text):
    # Only products/powers of named base units. Numeric scale factors are unsupported.
    def visit(n):
        if isinstance(n,ast.Name) and IDENT.fullmatch(n.id): return {n.id:1}
        if isinstance(n,ast.Constant) and n.value==1 and type(n.value) in (int,float):return {}
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Mult,ast.Div)):
            return combine(visit(n.left),visit(n.right),-1 if isinstance(n.op,ast.Div) else 1)
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Pow):
            power=n.right
            sign=1
            if isinstance(power,ast.UnaryOp) and isinstance(power.op,ast.USub):sign=-1;power=power.operand
            require(isinstance(power,ast.Constant) and type(power.value) in (int,float) and math.isfinite(power.value),'unit power must be numeric')
            return {k:v*sign*power.value for k,v in visit(n.left).items() if v*power.value!=0}
        raise ValueError('unit scale factors/conversions are not implemented; use explicit parameters')
    return visit(ast.parse(text,mode='eval').body)

class FP64Literals(ast.NodeTransformer):
    def visit_Constant(self,node):
        require(type(node.value) in (int,float),'numeric literal required')
        try:value=float(node.value)
        except (OverflowError,ValueError) as e:raise ValueError('literal outside FP64 domain') from e
        require(math.isfinite(value),'literal outside finite FP64 domain')
        return ast.copy_location(ast.Constant(value=value),node)

def integer_array(values,name):
    # Validate original elements before any coercion can truncate or round them.
    a=np.asarray(values,dtype=object);require(a.ndim==1,name+' must be a vector')
    result=[]
    for x in a:
        require(not isinstance(x,(bool,np.bool_)) and isinstance(x,(int,float,np.integer,np.floating)),name+' must contain integers')
        if isinstance(x,(float,np.floating)):require(np.isfinite(x) and x==math.floor(x),name+' fractional/nonfinite value')
        v=int(x);require(-(2**63)<=v<2**63,name+' outside int64 range');result.append(v)
    return np.asarray(result,dtype=np.int64)

def cuda_expr(node):
    if isinstance(node,ast.Constant): return repr(float(node.value))
    if isinstance(node,ast.Name): return 'v_'+node.id
    if isinstance(node,ast.UnaryOp): return ('-' if isinstance(node.op,ast.USub) else '+')+'('+cuda_expr(node.operand)+')'
    if isinstance(node,ast.BinOp):
        a,b=cuda_expr(node.left),cuda_expr(node.right)
        if isinstance(node.op,ast.Pow): return 'pow('+a+','+b+')'
        op={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'}[type(node.op)]
        return '('+a+op+b+')'
    if isinstance(node,ast.Call):
        return ('om_exprel' if node.func.id=='exprel' else node.func.id)+'('+cuda_expr(node.args[0])+')'
    raise ValueError('unsupported CUDA expression')

@dataclass
class Expression:
    text: str
    env: dict
    expected: dict
    def __post_init__(self):
        self.node=ast.fix_missing_locations(FP64Literals().visit(ast.parse(self.text,mode='eval').body))
        actual=dimension(self.node,self.env)
        zero=isinstance(self.node,ast.Constant) and self.node.value==0
        require(zero or actual==self.expected,f'equation units {actual} != {self.expected}: {self.text}')
        self.code=compile(ast.Expression(self.node),'<validated-equation>','eval')
        self.cuda=cuda_expr(self.node)
    def evaluate(self, values):
        return eval(self.code,{'__builtins__':{},**FUNCS},values)

class Model:
    """All numerical storage is SoA, compiled only from a validated descriptor."""
    def __init__(self, spec, prepare_mass_solver=True):
        self.spec=copy.deepcopy(spec)
        require(set(spec)<= {'version','populations','connections','mass','clamps','capabilities','description'}, 'unknown top-level field')
        require(spec.get('version')==1, 'unsupported IR version')
        for cap in spec.get('capabilities',[]):
            if cap not in ('smooth_ode','graded_ports','scheduled_edits'): raise Unsupported('capability not implemented: '+cap)
        self.pops=[];self.by_id={};self.slices={};self.units=[]
        self.parameters=[];self.initial=[];self.scale=[];self.port_bias=[]
        ns=no=ni=0
        for specpop in spec['populations']:
            require(set(specpop)<={'id','count','cell_ids','states','parameters','inputs','outputs','derived'},'unknown population field')
            name=specpop['id'];n=specpop['count']
            require(isinstance(name,str) and name not in self.by_id,'duplicate population id')
            require(type(n) is int and n>0, 'positive population count required')
            ids=integer_array(specpop['cell_ids'],'cell_ids') if 'cell_ids' in specpop else np.arange(n,dtype=np.int64)
            require(ids.shape==(n,) and len(np.unique(ids))==n,'cell identities must be unique')
            p={'id':name,'n':n,'ids':ids,'states':{},'params':{},'inputs':{},'outputs':{},'derived':{}}
            env={'t':{'s':1}}
            for group in ('states','parameters','inputs','derived','outputs'):
                for k,desc in specpop.get(group,{}).items():
                    require(IDENT.fullmatch(k) and k not in env and k not in FUNCS,'duplicate/reserved/invalid symbol: '+k)
                    require(isinstance(desc,dict),'field descriptions required')
                    allowed={'unit','initial','scale','rhs'} if group=='states' else {'unit','value'} if group=='parameters' else {'unit','value'} if group=='inputs' else {'unit','expr'}
                    require(set(desc)<=allowed,'unknown field property')
                    dim=unit(desc['unit'])
                    if group=='outputs':
                        # Outputs cannot depend on ports/derived port algebra: global coupling remains an ODE.
                        outenv={kk:vv for kk,vv in env.items() if kk in ('t',) or kk in p['states'] or kk in p['params']}
                        expr=Expression(desc['expr'],outenv,dim)
                        p['outputs'][k]=(slice(no,no+n),expr);no+=n
                    elif group=='derived':
                        expr=Expression(desc['expr'],env.copy(),dim);p['derived'][k]=expr
                    elif group=='states':
                        require(desc['scale']>0 and math.isfinite(desc['scale']),'positive finite state error scale')
                        p['states'][k]=slice(ns,ns+n);self.slices[(name,k)]=slice(ns,ns+n)
                        initial=np.broadcast_to(np.asarray(desc['initial'],dtype=float),(n,)).copy()
                        self.initial.extend(initial);self.scale.extend([desc['scale']]*n);self.units.extend([dim]*n);ns+=n
                    elif group=='parameters':
                        value=np.broadcast_to(np.asarray(desc['value'],dtype=float),(n,)).copy()
                        offset=len(self.parameters);self.parameters.extend(value);p['params'][k]=slice(offset,offset+n)
                    else:
                        p['inputs'][k]=slice(ni,ni+n);ni+=n
                        self.port_bias.extend(np.broadcast_to(np.asarray(desc.get('value',0),dtype=float),(n,)))
                    if group != 'outputs': env[k]=dim
            require(p['states'],'each population needs dynamic states')
            p['rhs']={k:Expression(desc['rhs'],env.copy(),combine(unit(desc['unit']),{'s':1},-1)) for k,desc in specpop['states'].items()}
            self.pops.append(p);self.by_id[name]=p
        require(ns>0,'empty model')
        self.n=ns;self.nout=no;self.nin=ni
        self.initial=np.asarray(self.initial);self.scale=np.asarray(self.scale,dtype=np.float64);self.parameters=np.asarray(self.parameters);self.port_bias=np.asarray(self.port_bias)
        for a in (self.initial,self.scale,self.parameters,self.port_bias): require(np.isfinite(a).all(),'nonfinite descriptor')
        rows=[];cols=[];weights=[]
        for conn in spec.get('connections',[]):
            require(set(conn)<={'source','target','pattern','offsets','weight','row','col','weights'},'unknown connection property (delays/events unsupported)')
            sp,sv=conn['source'];tp,tv=conn['target'];a=self.by_id[sp];b=self.by_id[tp]
            ao,expr=a['outputs'][sv];bi=b['inputs'][tv]
            sourceunit=unit(next(p for p in spec['populations'] if p['id']==sp)['outputs'][sv]['unit'])
            targetunit=unit(next(p for p in spec['populations'] if p['id']==tp)['inputs'][tv]['unit'])
            require(sourceunit==targetunit,'port unit mismatch: explicit conversion required')
            if conn.get('pattern')=='ring':
                require(a['n']==b['n'],'ring pattern requires equal sizes')
                offsets=integer_array(conn['offsets'],'offsets');require(len(offsets)>0,'empty ring offsets');rr=np.tile(np.arange(b['n']),len(offsets));cc=np.concatenate([(np.arange(b['n'])+(int(o)%a['n']))%a['n'] for o in offsets])
                ww=np.full(rr.size,conn['weight'])
            elif conn.get('pattern')=='coo':
                rr=integer_array(conn['row'],'connection rows');cc=integer_array(conn['col'],'connection columns');ww=np.broadcast_to(conn['weights'],rr.shape)
                require(rr.ndim==1 and cc.shape==rr.shape,'COO shape mismatch')
            else: raise Unsupported('connection pattern must be ring or coo')
            require(((rr>=0)&(rr<b['n'])).all() and ((cc>=0)&(cc<a['n'])).all(),'connection index out of range')
            require(np.isfinite(ww).all(),'nonfinite connection')
            rows.append(rr+bi.start);cols.append(cc+ao.start);weights.append(ww)
        join=lambda x,dtype:np.concatenate(x).astype(dtype) if x else np.empty(0,dtype)
        self.connection=sparse.csr_matrix((join(weights,float),(join(rows,np.int64),join(cols,np.int64))),shape=(ni,no))
        self.connection.sum_duplicates();self.connection.sort_indices()
        self.mass=sparse.eye(ns,format='csc')
        if 'mass' in spec:
            m=spec['mass'];require(set(m)=={'row','col','values'},'mass must be a constant COO matrix')
            mr=integer_array(m['row'],'mass rows');mc=integer_array(m['col'],'mass columns')
            require(mr.shape==mc.shape and ((mr>=0)&(mr<ns)&(mc>=0)&(mc<ns)).all(),'mass index out of range')
            self.mass=sparse.csc_matrix((m['values'],(mr,mc)),shape=(ns,ns),dtype=np.float64);self.mass.sum_duplicates()
            require(np.isfinite(self.mass.data).all(),'nonfinite mass')
            for r,c in zip(*self.mass.nonzero()): require(self.units[r]==self.units[c],'mass mixes state dimensions')
        diag=self.mass.diagonal();self.diagonal_mass=(self.mass-sparse.diags(diag)).nnz==0
        self.mass_diag=diag
        if self.diagonal_mass: require((diag>0).all(),'diagonal mass must be positive')
        self.clamp_mask=np.zeros(ns,dtype=bool);self.clamp_values=np.zeros(ns)
        for clamp in spec.get('clamps',[]):
            require(set(clamp)=={'population','state','cells','value'},'invalid clamp')
            pop=self.by_id[clamp['population']];sl=pop['states'][clamp['state']]
            lookup={int(v):i for i,v in enumerate(pop['ids'])}
            idx=np.array([sl.start+lookup[int(c)] for c in integer_array(clamp['cells'],'clamp cell ids')],dtype=np.int64)
            require(np.isfinite(clamp['value']),'nonfinite clamp')
            self.clamp_mask[idx]=True;self.clamp_values[idx]=clamp['value'];self.initial[idx]=clamp['value']
        if not self.diagonal_mass and self.clamp_mask.any():
            m=self.mass.tolil();idx=np.flatnonzero(self.clamp_mask)
            for i in idx: m.rows[i]=[int(i)];m.data[i]=[1.]
            self.mass=m.tocsc()
        self.lu=None
        if not self.diagonal_mass and prepare_mass_solver:
            if ns>4096: raise Unsupported('CPU LU reference limited to 4096 states; choose the implicit GPU operator for larger problems')
            try: self.lu=splu(self.mass)
            except RuntimeError as e: raise Unsupported('singular mass/DAE not supported') from e
        self.layout=[{'id':p['id'],'count':p['n'],**{g:list(p[g]) for g in ('states','params','inputs','outputs','derived')}} for p in self.pops]
        self.descriptor_hash=digest(self.spec)
        self.identity=digest({'numeric_IR':'fp64-v2','descriptor':self.spec,'ordered_layout':self.layout})

    def raw_rhs(self,t,y):
        require(np.isfinite(y).all(),'nonfinite input state')
        out=np.empty(self.nout);contexts=[]
        for p in self.pops:
            v={'t':t,**{k:y[s] for k,s in p['states'].items()},**{k:self.parameters[s] for k,s in p['params'].items()}}
            for k,(sl,expr) in p['outputs'].items(): out[sl]=expr.evaluate(v)
            contexts.append(v)
        require(np.isfinite(out).all(),'nonfinite output')
        inp=self.port_bias+self.connection@out
        require(np.isfinite(inp).all(),'nonfinite input port')
        f=np.empty(self.n)
        for p,v in zip(self.pops,contexts):
            v.update({k:inp[s] for k,s in p['inputs'].items()})
            for k,expr in p['derived'].items():
                v[k]=expr.evaluate(v);require(np.isfinite(v[k]).all(),'nonfinite derived state')
            for k,expr in p['rhs'].items(): f[p['states'][k]]=expr.evaluate(v)
        require(np.isfinite(f).all(),'nonfinite derivative')
        f[self.clamp_mask]=0
        return f

    def rhs(self,t,y):
        f=self.raw_rhs(t,y)
        if self.diagonal_mass:return f/self.mass_diag
        require(self.lu is not None,'mass solver not prepared')
        return self.lu.solve(f)

    def cuda_source(self):
        from autodiff import derivative
        require(self.diagonal_mass,'GPU non-diagonal mass is not implemented')
        lines=['#include <math_constants.h>','__device__ double om_exprel(double x) { return fabs(x)<1e-5 ? 1.0+x*(0.5+x*(1.0/6+x*(1.0/24+x/120))) : expm1(x)/x; }']
        lines.append('__device__ double om_dexprel(double x) { return fabs(x)<1e-4 ? 0.5+x*(1.0/3+x*(1.0/8+x/30)) : (x+(x-1)*expm1(x))/(x*x); }')
        for j,p in enumerate(self.pops):
            for phase in ('output','rhs'):
                lines.append(f'extern "C" __global__ void {phase}_{j}(const double* y,const double* par,const double* inp,double* out,const double* clock,double tf,const double* mass,const bool* clamp,int* flag,double* jac,int need_diag) {{ int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>={p["n"]}) return; double v_t=clock[0]+tf*clock[1];')
                for group,src in (('states','y'),('params','par')):
                    for k,sl in p[group].items(): lines.append(f'double v_{k}={src}[{sl.start}+i]; if(!isfinite(v_{k})) atomicOr(flag,1);')
                if phase=='output':
                    for k,(sl,expr) in p['outputs'].items(): lines.append(f'double r_{k}={expr.cuda}; if(!isfinite(r_{k})) atomicOr(flag,1); out[{sl.start}+i]=r_{k};')
                else:
                    for k,sl in p['inputs'].items(): lines.append(f'double v_{k}=inp[{sl.start}+i]; if(!isfinite(v_{k})) atomicOr(flag,1);')
                    for k,expr in p['derived'].items(): lines.append(f'double v_{k}={expr.cuda}; if(!isfinite(v_{k})) atomicOr(flag,1);')
                    for k,expr in p['rhs'].items():
                        sl=p['states'][k];lines.append(f'double r_{k}={expr.cuda}; if(!isfinite(r_{k})) atomicOr(flag,1); out[{sl.start}+i]=clamp[{sl.start}+i] ? 0.0 : r_{k}/mass[{sl.start}+i];')
                        lines.append('if(need_diag) {')
                        denv={k:'1.0'}
                        for dk,de in p['derived'].items():
                            dn='d_'+k+'_'+dk;lines.append(f'double {dn}={derivative(de.node,denv)};');denv[dk]=dn
                        lines.append(f'double dv={derivative(expr.node,denv)}; if(!isfinite(dv)) atomicOr(flag,1); jac[{sl.start}+i]=clamp[{sl.start}+i] ? 0.0 : dv/mass[{sl.start}+i]; }}')
                lines.append('}')
        return '\n'.join(lines)
