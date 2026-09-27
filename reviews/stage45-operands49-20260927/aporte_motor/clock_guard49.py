"""Exact, lineage-scoped replacement of two cold-load clock predicates.

The stored MuJoCo time is never changed. The prepared40 anchor and25us schedule
predict its floating-point recurrence exactly. All other constructor checks
and all integration methods remain the historical ones.
"""
import ast
import copy
import hashlib
import inspect
import json
import math
import textwrap
from pathlib import Path

ANCHOR_TIME = 44.486000000027154
ANCHOR_STEPS = 1779440
DT = 2.5e-5
MAX_STEPS = ANCHOR_STEPS + 3200*40
ANCHOR_MANIFEST_SHA256 = 'af1eea807e43e01a504c0488c1be3d608727e0157ecc08cf2d1cc31d7c8ece9d'
OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
ROOT = Path('/home/daroch/AXIOMA_ASTRA')
ANCHOR = ROOT/'campanas/etapa45_navigation_wind_20260925_40/navigation_minus_filtered_wind_03/prepared_state'
HERE = Path(__file__).resolve().parent
PREDICATE = ast.parse('abs(obj.data.time-obj.steps*obj.dt)>1e-10', mode='eval').body


def expected_time(steps):
    if type(steps) is not int or not ANCHOR_STEPS <= steps <= MAX_STEPS:
        raise ValueError('Clock outside qualified49 lineage/horizon')
    value = ANCHOR_TIME
    for _ in range(steps-ANCHOR_STEPS):
        value += DT
    return value


def clock_violation(obj):
    actual = float(obj.data.time)
    if not math.isfinite(actual):
        return True
    if obj.steps < ANCHOR_STEPS:
        # Pre-anchor loaders keep their historical criterion verbatim.
        return abs(actual-obj.steps*obj.dt)>1e-10
    if obj.dt != DT:
        return True
    try:
        return actual != expected_time(obj.steps)
    except ValueError:
        return True


def transformed_function(source, class_name):
    """Replace exactly one known test, preserving every other AST node."""
    module = ast.parse(source)
    cls = next(x for x in module.body if isinstance(x, ast.ClassDef) and x.name == class_name)
    original = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'from_state')
    candidate = copy.deepcopy(original)
    matches = [x for x in ast.walk(candidate) if isinstance(x, ast.If) and
               ast.dump(x.test) == ast.dump(PREDICATE)]
    if len(matches) != 1:
        raise ValueError('Unexpected constructor clock predicate')
    replacement = ast.Call(func=ast.Name(id='_clock_violation49', ctx=ast.Load()),
                           args=[ast.Name(id='obj', ctx=ast.Load())], keywords=[])
    matches[0].test = ast.copy_location(replacement, matches[0].test)
    # Check structurally that restoring this predicate recovers the entire method.
    proof = copy.deepcopy(candidate)
    altered = [x for x in ast.walk(proof) if isinstance(x, ast.If) and
               isinstance(x.test, ast.Call) and isinstance(x.test.func, ast.Name) and
               x.test.func.id == '_clock_violation49']
    if len(altered) != 1:
        raise ValueError('More than one adapted test')
    altered[0].test = copy.deepcopy(PREDICATE)
    if ast.dump(proof) != ast.dump(original):
        raise ValueError('Constructor changed outside its clock predicate')
    candidate.decorator_list = []
    result = ast.fix_missing_locations(ast.Module(body=[candidate], type_ignores=[]))
    return result


def install():
    """Local reversible overlay; its receipt exposes the actual executed change."""
    from guarded_tibia_body import GuardedTibiaBody
    from rh_tarsal_body import RHTarsalBody
    if hashlib.sha256((ANCHOR/'MANIFEST.json').read_bytes()).hexdigest() != ANCHOR_MANIFEST_SHA256:
        raise ValueError('Prepared anchor identity changed')
    lock = json.loads((ROOT/'campanas/etapa45_composicion_20260927_48/SOURCES.json').read_text())
    restored = []
    receipt = dict(schema='exact_clock_recurrence_overlay49_v1', anchor=str(ANCHOR),
                   anchor_manifest_sha256=ANCHOR_MANIFEST_SHA256, anchor_time=ANCHOR_TIME,
                   anchor_steps=ANCHOR_STEPS, dt=DT, max_steps=MAX_STEPS,
                   scope='Only two loader predicates; prior lineages retain old check; no integration/time changes',
                   overlay_path=str(Path(__file__).resolve()),
                   overlay_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), methods=[])
    try:
        for cls in (GuardedTibiaBody, RHTarsalBody):
            descriptor = cls.__dict__['from_state']
            function = descriptor.__func__
            path = OLD/'src'/('guarded_tibia_body.py' if cls is GuardedTibiaBody else 'rh_tarsal_body.py')
            source = path.read_text()
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != lock[str(path)] or Path(inspect.getsourcefile(function)).resolve() != path:
                raise ValueError('Historical method already changed or overlaid')
            tree = transformed_function(source, cls.__name__)
            namespace = dict(function.__globals__)
            namespace['_clock_violation49'] = clock_violation
            exec(compile(tree, str(Path(__file__).resolve()), 'exec'), namespace)
            replacement = namespace['from_state']
            replacement.__module__ = __name__
            restored.append((cls, descriptor))
            cls.from_state = classmethod(replacement)
            receipt['methods'].append(dict(class_name=cls.__name__, method='from_state',
                original_source=str(path), original_sha256=actual,
                replacement_ast_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
                changed_predicates=1, scientific_state_modified=False))
    except BaseException:
        for cls, descriptor in reversed(restored):
            cls.from_state = descriptor
        raise
    def undo():
        for cls, descriptor in reversed(restored):
            cls.from_state = descriptor
    return receipt, undo
