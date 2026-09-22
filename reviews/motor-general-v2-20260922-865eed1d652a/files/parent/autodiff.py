"""Forward symbolic differentiation of the already validated arithmetic AST."""
import ast
from model import cuda_expr

def add(a,b):return b if a=='0.0' else a if b=='0.0' else '('+a+'+'+b+')'
def mul(a,b):return '0.0' if a=='0.0' or b=='0.0' else b if a=='1.0' else a if b=='1.0' else '('+a+'*'+b+')'
def derivative(node,env):
    if isinstance(node,ast.Constant):return '0.0'
    if isinstance(node,ast.Name):return env.get(node.id,'0.0')
    if isinstance(node,ast.UnaryOp):
        d=derivative(node.operand,env);return d if isinstance(node.op,ast.UAdd) or d=='0.0' else '-('+d+')'
    if isinstance(node,ast.BinOp):
        a,b=cuda_expr(node.left),cuda_expr(node.right);da,db=derivative(node.left,env),derivative(node.right,env)
        if isinstance(node.op,ast.Add):return add(da,db)
        if isinstance(node.op,ast.Sub):return add(da,'-('+db+')' if db!='0.0' else db)
        if isinstance(node.op,ast.Mult):return add(mul(da,b),mul(a,db))
        if isinstance(node.op,ast.Div):
            n=add(mul(da,b),'-('+mul(a,db)+')' if db!='0.0' else '0.0')
            return '0.0' if n=='0.0' else '('+n+'/('+b+'*'+b+'))'
        if isinstance(node.op,ast.Pow):
            power=float(node.right.value)
            if power==0:return '0.0'
            return mul(mul(repr(power),'pow('+a+','+repr(power-1)+')'),da)
    if isinstance(node,ast.Call):
        a=cuda_expr(node.args[0]);da=derivative(node.args[0],env)
        if da=='0.0':return da
        name=node.func.id
        factor={'exp':f'exp({a})','expm1':f'exp({a})','exprel':f'om_dexprel({a})','log':f'(1.0/({a}))',
                'tanh':f'(1-tanh({a})*tanh({a}))','sin':f'cos({a})','cos':f'(-sin({a}))','sqrt':f'(0.5/sqrt({a}))'}[name]
        return mul(factor,da)
    raise ValueError('unsupported derivative node')
