"""No neural run: guard must refuse exhausted/invalid aggregate accounting."""
from budget_guard import remaining

plan={'budget':{'neural_processes_max':3,'cpu_s_aggregate_max':3000,'wall_s_aggregate_max':2700,
                'wall_s_per_process_max':900,'neural_ms_max':360}}
def rejects(records):
    try:remaining(plan,records)
    except RuntimeError:return
    raise RuntimeError('Forbidden continuation admitted')

row={'cpu_s':3169.303591,'wall_s':466.9552197599842,'attempted_ms':120}
rejects([row])
rejects([{**row,'cpu_s':3000}])
rejects([{**row,'cpu_s':float('nan')}])
rejects([{**row,'cpu_s':-1}])
rejects([{**row,'cpu_s':1,'wall_s':2700}])
rejects([{**row,'cpu_s':1,'attempted_ms':360}])
rejects([{**row,'cpu_s':1}]*3)
accepted=remaining(plan,[{**row,'cpu_s':200}])
if accepted!={'cpu_s':2800.0,'wall_s':900,'neural_ms':240}:raise RuntimeError('Wrong remaining aggregate')
print('8 resource-accounting checks passed; zero neural steps')
