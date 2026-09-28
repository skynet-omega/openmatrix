"""Planning memory is read-only evidence retrieval, never scientific execution."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import pytest

from problem_context import brief_context, concepts, retrieve, tokens

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('context_fixture_catalog', ROOT/'scripts/code_catalog.py')
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)


def write(root, path, value):
    p = root/path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value) if isinstance(value, (list, dict)) else value)
    return p


@pytest.fixture
def library(tmp_path):
    root = tmp_path/'matrix'
    root.mkdir()
    write(root, 'INDEX.md', '# Current\nDNa02 geometry is provisional. No identified passive parameters.\n')
    write(root, 'config/olfactory_mobility_plan_v1.json', {
        'active_priority':[3], 'stage_status':{'3':'open'},
        'next_work':'work/old_named_but_current/NEXT_WORK.json',
        'next_stage2_work':'work/CTr/NEXT_WORK.json'})
    write(root, 'work/old_named_but_current/NEXT_WORK.json', {
        'objective':'DNa02 passive transfer needs mapped input and output ports',
        'do_not_repeat':['Transfer local5 passive parameters to DNa02 as measurements'],
        'boundary':'DNa02 geometry alone does not identify physiology'})
    write(root, 'work/CTr/NEXT_WORK.json', {
        'objective':'Identify CTr torsion before force changes',
        'entry':'work/CTr/README.md',
        'closed':'Do not repeat the same CTr force scaling; that negative applies to the tested posture'})
    write(root, 'work/new_date_but_historical/NEXT_WORK.json', {'objective':'Promote DNa02 now'})
    db = root/'catalogo/catalogo.sqlite';db.parent.mkdir()
    with sqlite3.connect(db) as con:
        con.execute('CREATE VIRTUAL TABLE documents USING fts5(path,page UNINDEXED,kind UNINDEXED,title,text)')
        for path, title, text in [
            ('work/CTr/README.md','CTr frame','CTr frame match failed at51degrees; no force retuning justified.'),
            ('work/negative/README.md','DNa02 passive comparison','DNa02 spatial transfer failed this criterion. Failure is conditional on this input and soma observer.'),
            ('work/swc/README.md','SWC radius units','SWC radius in NeuTu differs from old neuclease convention. Verify exporter before applying unit correction.'),
            ('work/uncurated/README.md','Passive transfer mechanism','Passive transfer with no special curated card. This uncurated result matters.'),
            ('work/spectral/README.md','Spectral analysis','spectral constraints describe photoreceptors, not the queried motor joint.')]:
            write(root,path,text)
            con.execute('INSERT INTO documents VALUES (?,?,?,?,?)',(str(root/path),0,'Documento',title,text))
        # Many availability records must not hide the scientific negative.
        for i in range(15):
            con.execute('INSERT INTO documents VALUES (?,?,?,?,?)',
                (str(root/'config/holdings.json'),i,'Disponibilidad','DNa02 file','DNa02 available'))
    write(root, 'src/passive.py', '''"""Passive transfer in nF/nS/mV/pA; requires declared input and output ports."""
class NativePassiveCable:
    """Passive cable, not identified DNa02 physiology."""
    def impedance(self):
        return 1
''')
    write(root, 'src/geometry.py', '''def parse_swc(radius):
    """SWC radius units depend on exporter; no silent convention transfer."""
    return radius
''')
    write(root, 'src/ctr.py', 'def ctr_frame():\n    """CTr motor frame."""\n    return 0\n')
    write(root, 'src/spectral.py', 'def spectral_ratio():\n    return 0\n')
    write(root, 'work/sealed/tool.py', 'def ctr_candidate():\n    """CTr motor precedent."""\n    return 1\n')
    write(root, 'work/sealed/MANIFEST.json', {})
    catalog.build(root,root/'catalogo/code.sqlite')
    return root


def test_ctr_word_boundary_does_not_retrieve_spectral(library):
    r=retrieve('CTr',library)
    assert r['evidence'] and r['code_candidates']
    assert not any('spectral' in x['path'] for x in r['evidence']+r['code_candidates'])
    assert any('same CTr force scaling' in x['text'] for x in r['scoped_decisions'])
    assert r['evidence'][0]['path'].endswith('work/CTr/README.md')
    assert r['evidence'][0]['current_reference'].endswith('work/CTr/NEXT_WORK.json')


@pytest.mark.parametrize('observed', ['COMPLETE', 'FAILED', 'RUNNING', None])
def test_experiment_status_comes_from_receipt_without_mutating_declaration(library, observed):
    path=library/'config/olfactory_mobility_plan_v1.json'
    plan=json.loads(path.read_text())
    plan['active_experiment']=dict(id='trial',status='RUNNING',execution='work/trial/execution.json')
    path.write_text(json.dumps(plan))
    if observed is not None:
        write(library,'work/trial/execution.json',dict(status=observed))
    result=retrieve('DNa02',library)
    experiment=brief_context(result)['state']['experiment']
    assert experiment['execution_status']==(observed or 'UNKNOWN')
    assert result['current_state']['experiment']['declared_status']=='RUNNING'
    assert 'scope' not in experiment and 'id' not in experiment
    assert json.loads(path.read_text())==plan
    if observed in ('COMPLETE','FAILED'):
        assert experiment['declared_status']=='RUNNING'
        assert any('status mismatch' in w for w in result['warnings'])
    if observed is None:
        assert any('status unavailable' in w for w in result['warnings'])


def test_free_query_finds_uncurated_mechanism_and_separates_availability(library):
    r=retrieve('Quiero transferencia pasiva DNa02',library)
    assert any('uncurated' in x['path'] for x in r['evidence'])
    assert any('NativePassiveCable' in x['symbol'] for x in r['code_candidates'])
    assert any('negative' in x['path'] for x in r['evidence'])
    assert all(x['kind']!='Disponibilidad' for x in r['evidence'])
    assert len(r['availability'])<=3


def test_named_mechanism_beats_incidental_entity_list(library):
    write(library,'src/generic_cable.py','class PassiveCable:\n    """Passive transfer on a forest."""\n    pass\n')
    write(library,'src/incidental.py','def analyze():\n    """DNa02 passive transfer are mentioned in an unrelated analysis."""\n    pass\n')
    catalog.build(library,library/'catalogo/code.sqlite')
    rows=retrieve('DNa02 transferencia pasiva',library,limit=12)['code_candidates']
    paths=[x['path'] for x in rows]
    assert paths.index(str(library/'src/generic_cable.py')) < paths.index(str(library/'src/incidental.py'))


def test_exact_snapshot_pool_prefers_active_copy_and_keeps_history(library):
    source = 'def fit():\n    """Release uncertainty."""\n    return 1\n'
    canonical = write(library, 'src/reusable.py', source)
    snapshots = []
    for i in range(35):
        snapshots.append(write(library, f'work/release_uncertainty_{i}/source_snapshot/model.py', source))
    different = write(library, 'work/earlier/model.py', source.replace('return 1', 'return 2'))
    catalog.build(library, library/'catalogo/code.sqlite')
    rows = retrieve('release uncertainty earlier', library, limit=20)['code_candidates']
    row = next(x for x in rows if x['path'] == str(canonical))
    # More copies than the preliminary pool; resolve by hash before truncation.
    assert row['freshness'] == 'content_matches_index'
    assert row['same_indexed_content_at']
    assert set(row['same_indexed_content_at']).issubset({str(p) for p in snapshots})
    assert not any(x['path'] in row['same_indexed_content_at'] for x in rows)
    assert any(x['path'] == str(different) for x in rows)
    assert str(different) not in row['same_indexed_content_at']
    assert 'not inferred' in row['discovery']


def test_registered_api_survives_incidental_catalogue_matches_and_brief(library):
    source = 'def evaluate():\n    """Release uncertainty with temporal memory."""\n    return 1\n'
    canonical = write(library, 'src/comparison.py', source)
    write(library, 'work/release_uncertainty/snapshot/comparison.py', source)
    for i in range(25):
        write(library, f'work/release_uncertainty_temporal_{i}/catalogue.py',
              f'def catalogue():\n    """release uncertainty temporal"""\n    return {i}\n')
    # Generic capability text must not displace a more relevant actual entrypoint.
    caps = [dict(id=f'generic_{i}', capability='Release uncertainty temporal',
                 state='release uncertainty temporal', entrypoints=[]) for i in range(8)]
    caps.append(dict(id='comparison', capability='Paired comparison', state='Development only',
        entrypoints=[dict(path='src/comparison.py', symbols=['evaluate'],
            purpose='Temporal paired prediction', limits=['Not identified conductances.'])]))
    write(library, 'config/tooling_capabilities_v1.json', {'capabilities': caps})
    catalog.build(library, library/'catalogo/code.sqlite')
    result = retrieve('release uncertainty temporal', library)
    assert result['tools'][0]['id'] == 'comparison'
    assert result['code_candidates'][0]['path'] == str(canonical)
    assert result['code_candidates'][0]['declared_entrypoint']
    brief = brief_context(result)
    assert brief['apis'][0]['source'] == str(canonical)
    assert brief['apis'][0]['kind'] == 'declared_API_reference'
    assert brief['apis'][0]['contract_source'].endswith('tooling_capabilities_v1.json')
    assert 'consult contract' in brief['apis'][0]['applicability']
    assert len({r['source'] for r in brief['apis']}) == len(brief['apis'])
    assert brief['budget']['used_chars'] <= 6000


def test_live_registration_without_code_refresh_remains_discoverable(library):
    canonical = write(library, 'src/new_release.py', 'def evaluate():\n    """Release memory."""\n    return 0\n')
    write(library, 'config/tooling_capabilities_v1.json', {'capabilities': [dict(
        id='new_release', capability='Release memory', entrypoints=[dict(
            path='src/new_release.py', symbols=['evaluate'], purpose='Release memory',
            tests=['tests/test_release.py'], limits=['Synthetic only.'])])]})
    index = library/'catalogo/code.sqlite'
    before = index.read_bytes()
    context = retrieve('release memory', library)
    brief = brief_context(context)
    assert brief['apis'][0]['source'] == str(canonical)
    assert brief['apis'][0]['kind'] == 'declared_API_reference'
    assert not any(x['path'] == str(canonical) for x in context['code_candidates'])
    assert index.read_bytes() == before


def test_tool_usage_example_does_not_claim_subject_implementation(library):
    write(library, 'src/search.py', 'def retrieve():\n    """Read catalogue."""\n    return []\n')
    write(library, 'config/tooling_capabilities_v1.json', {'capabilities': [dict(
        id='memory', capability='Search tools and evidence', entrypoints=[dict(
            path='src/search.py', symbols=['retrieve'], purpose='Search existing evidence',
            usage=["./matrix context CTr"], references=['work/CTr/README.md'])])]})
    catalog.build(library, library/'catalogo/code.sqlite')
    result = retrieve('CTr', library)
    assert all(not t['entrypoints'] for t in result['tools'])
    assert result['code_candidates'][0]['path'].endswith('/src/ctr.py')
    assert brief_context(result)['apis'][0]['source'].endswith('/src/ctr.py')


def test_current_pointer_wins_over_folder_date_and_negatives_remain_scoped(library):
    r=retrieve('DNa02',library)
    assert r['current_next_work'][0]['path'].endswith('old_named_but_current/NEXT_WORK.json')
    assert 'Promote DNa02 now' not in json.dumps(r)
    assert r['scoped_decisions'] and not any(x['universal_prohibition'] for x in r['scoped_decisions'])
    assert r['advisory_only'] and not r['executes_science']


def test_bilingual_radius_query_retains_exporter_condition(library):
    r=retrieve('unidades radio SWC',library)
    assert any('neuclease' in x['excerpt'] and 'NeuTu' in x['excerpt'] for x in r['evidence'])
    assert any(x['symbol']=='parse_swc' for x in r['code_candidates'])
    assert 'spectral' not in tokens('CTr')


def test_spanish_excitation_query_finds_distributed_context_tool_without_execution(library):
    path=library/'config/tooling_capabilities_v1.json'
    data={'capabilities':[]}
    data['capabilities'].append(dict(id='distributed_context',capability='Recurrent population context',
        state='Observed excitatory and inhibitory source projection; no causal qualification',
        entrypoints=[dict(id='distributed:audit',path='src/distributed_context_audit.py',
                         purpose='Excitation inhibition and distributed population context',symbols=['audit_counts'])]))
    path.write_text(json.dumps(data))
    result=retrieve('falta excitación contexto recurrente poblaciones',library)
    assert any(t['id']=='distributed_context' and t['entrypoints'] for t in result['tools'])
    assert result['advisory_only'] and not result['executes_science']


def test_latest_reference_is_recovered_without_becoming_a_qualification(library):
    path=library/'work/old_named_but_current/NEXT_WORK.json'
    data=json.loads(path.read_text());data['latest_DNa02_geometry']='work/new_geom/README.md'
    path.write_text(json.dumps(data))
    write(library,'work/new_geom/README.md','DNa02 SWC radius units provisional; disconnected soma fragment retained.')
    r=retrieve('unidades radio SWC',library)
    row=next(x for x in r['evidence'] if x['path'].endswith('new_geom/README.md'))
    assert row['current_reference']==str(path)
    assert 'provisional' in row['excerpt'] and r['advisory_only']


def test_retrieval_never_imports_payload_and_does_not_mutate_indexes(library):
    marker=library/'IMPORTED'
    write(library,'src/effects.py',f'from pathlib import Path\nPath({str(marker)!r}).write_text("BAD")\n# DNa02\n')
    catalog.build(library,library/'catalogo/code.sqlite')
    dbs=list((library/'catalogo').glob('*.sqlite'))
    original={p:p.read_bytes() for p in dbs}
    retrieve('DNa02',library)
    assert not marker.exists()
    assert original=={p:p.read_bytes() for p in dbs}
    assert not (library/'runs').exists()


def test_changed_code_and_sealed_code_do_not_claim_validation(library):
    path=library/'src/ctr.py';path.write_text(path.read_text()+'\n# changed\n')
    r=retrieve('CTr',library)
    stale=next(x for x in r['code_candidates'] if x['path']==str(path))
    assert stale['freshness']=='changed_since_index'
    sealed=next(x for x in r['code_candidates'] if '/sealed/' in x['path'])
    assert sealed['seal'] and 'not inferred' in sealed['discovery']


def test_live_card_replaces_stale_indexed_decision(library):
    path='work/card/README.md';write(library,path,'Current correction says this result is not admitted')
    old=dict(id='card',problem='DNa02',status='DNa02 DNa02 promoted',files=[f'matrix/{path}'])
    with sqlite3.connect(library/'catalogo/catalogo.sqlite') as c:
        c.execute('INSERT INTO documents VALUES (?,?,?,?,?)', (str(library/path),0,'Ficha','DNa02',json.dumps(old)))
    new={**old,'status':'Correction: not admitted','limit':'No DNa02 calibration'}
    write(library,'config/scientific_catalog_current_parts.json',{'parts':[new]})
    r=retrieve('DNa02',library)
    rows=[x for x in r['evidence'] if x['path']==str(library/path)]
    assert len(rows)==1 and rows[0]['reported_decision']['status']=='Correction: not admitted'


def test_missing_indexes_report_partial_coverage_without_building(tmp_path):
    r=retrieve('DNa02',tmp_path)
    assert len(r['warnings'])==2
    assert not list(tmp_path.iterdir())
    assert r['code_candidates']==[] and r['evidence']==[]


def test_no_fts_fallback_keeps_boundaries(library):
    with sqlite3.connect(library/'catalogo/code.sqlite') as c:
        c.execute('DROP TABLE file_fts')
    r=retrieve('CTr',library)
    assert any('fallback' in w for w in r['warnings'])
    assert r['code_candidates'] and not any('spectral' in x['path'] for x in r['code_candidates'])


def test_external_small_source_is_not_read(library,tmp_path):
    path=tmp_path/'outside.md';path.write_text('SECRET LIVE VALUE DNa02')
    with sqlite3.connect(library/'catalogo/catalogo.sqlite') as c:
        c.execute('INSERT INTO documents VALUES (?,?,?,?,?)',(str(path),0,'Documento','DNa02','DNa02 indexed only'))
    r=retrieve('DNa02',library)
    assert 'SECRET LIVE VALUE' not in json.dumps(r)


def test_plan_advice_does_not_change_scientific_identity_or_create_runs(library,monkeypatch,capsys):
    spec=importlib.util.spec_from_file_location('context_workbench_cli',ROOT/'scripts/workbench.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    write(library,'config/workflows/fixture.json',dict(schema='matrix_workflow_v1',id='fixture',
        title='DNa02 passive transfer',scope='development only',steps=[dict(id='compare',component='compare_arrays',inputs={})]))
    write(library,'src/session_io.py','# identity fixture\n')
    write(library,'src/matrix_workbench/science.py','# scientific source\n')
    bench=cli.Workbench(library);before=bench.code_identity()
    monkeypatch.setattr(cli,'ROOT',library)
    monkeypatch.setattr(sys,'argv',['workbench.py','plan','fixture'])
    cli.main();result=json.loads(capsys.readouterr().out)
    assert result['advisory_context']['advisory_only']
    assert result['advisory_context']['schema']=='matrix_problem_brief_v1'
    assert len(json.dumps(result['advisory_context'],ensure_ascii=False,indent=2))+1<=6000
    assert result['steps'][0]['component']=='compare_arrays'
    monkeypatch.setattr(sys,'argv',['workbench.py','plan','fixture','--full-context'])
    cli.main();full=json.loads(capsys.readouterr().out)
    assert full['advisory_context']['schema']=='matrix_problem_context_v1'
    assert full['steps']==result['steps']
    write(library,'src/problem_context.py','# changed retrieval implementation\n')
    assert bench.code_identity()==before
    assert not (library/'runs').exists()


def test_invalid_or_overbroad_query_is_bounded():
    with pytest.raises(ValueError):concepts('qué puedo hacer')
    with pytest.raises(ValueError):concepts(' '.join(f'term{i}' for i in range(17)))


def test_brief_keeps_authoritative_entry_when_evidence_ranking_does_not(library):
    from problem_context import brief_context
    full=retrieve('DNa02',library)
    full['evidence']=[]
    brief=brief_context(full)
    assert brief['navigation_entry']==str(library/'INDEX.md')
    assert len(json.dumps(brief,ensure_ascii=False,indent=2))+1<=6000


def test_brief_next_work_reports_omitted_contract_fields(library):
    from problem_context import brief_context
    full=retrieve('DNa02',library)
    row=full['current_next_work'][0]
    row['declared_next']['next_work_contract']={'limits':'DNa02 remains uncalibrated. ' * 1000}
    brief=brief_context(full,max_chars=12000)
    selected=brief['next_work'][0]
    assert selected['details']['objective']==row['declared_next']['objective']
    assert 'next_work_contract' in selected['omitted_fields']
    assert brief['omitted']['next_work_details']>=1
    assert set(selected['details']) | set(selected['omitted_fields']) == set(row['declared_next'])


def test_brief_budget_state_references_and_no_mutation(library):
    context=retrieve('DNa02 transferencia pasiva',library)
    before=json.dumps(context,sort_keys=True)
    brief=brief_context(context)
    rendered=json.dumps(brief,ensure_ascii=False,indent=2)+'\n'
    assert len(rendered)==brief['budget']['used_chars']<=6000
    assert brief['state']['stages']==context['current_state']['stages']
    assert brief['state']['active_priority']==context['current_state']['active_priority']
    assert brief['question']==context['query']
    assert {x['source'] for x in brief['next_work']}=={x['path'] for x in context['current_next_work']}
    assert len(brief['precedents'])<=3 and len(brief['apis'])<=2
    assert json.dumps(context,sort_keys=True)==before


def test_brief_keeps_whole_negative_or_marks_omission(library):
    context=retrieve('DNa02',library)
    negative='No recruitment under this protocol; it does not exclude a response under a different sensory state.'
    context['scoped_decisions']=[{'source':'evidence/negative.md','text':negative,'basis':'Only protocol A'}]
    result=brief_context(context,10000)
    row=next(x for x in result['precedents'] if x['source']=='evidence/negative.md')
    assert row['details']['statement']==negative
    # A long assertion must never become a convincing, condition-free prefix.
    long_negative=negative+' Important condition.'*2000
    context['scoped_decisions'][0]['text']=long_negative
    result=brief_context(context,6000)
    row=next(x for x in result['precedents'] if x['source']=='evidence/negative.md')
    assert isinstance(row['details'],str) and 'omitted' in row['details']
    assert negative not in json.dumps(result)
    assert result['omitted']['precedent_details']>0


def test_brief_preserves_do_not_repeat_polarity_from_field(library):
    context=retrieve('DNa02',library)
    context['scoped_decisions']=[{'source':'evidence/limit.md','field':'do_not_repeat',
        'text':'Declare the exposed reference independently validated','basis':'Only this exposed reference',
        'universal_prohibition':False}]
    result=brief_context(context,10000)
    row=next(x for x in result['precedents'] if x['source']=='evidence/limit.md')
    assert row['details']['source_field']=='do_not_repeat'
    assert row['details']['statement']=='Declare the exposed reference independently validated'
    assert row['details']['universal_prohibition'] is False


def test_brief_does_not_restate_pretruncated_excerpts(library):
    context=retrieve('DNa02',library)
    for row in context['evidence']:
        row.pop('reported_decision',None)
        row['excerpt']='The manipulation succeeded, except when…'
        row['indexed_excerpt']='The manipulation succeeded, except when…'
    brief=brief_context(context,12000)
    assert 'The manipulation succeeded' not in json.dumps(brief)
    assert any(x['kind']=='evidence_reference' for x in brief['precedents'])


def test_brief_many_large_texts_and_unicode_still_bounded(library):
    context=retrieve('DNa02',library)
    context['warnings']=['Aviso completo: '+('á𝛥'*10000)]*40
    context['evidence']*=100
    context['code_candidates']*=100
    result=brief_context(context,6000)
    assert len(json.dumps(result,ensure_ascii=False,indent=2)+'\n')<=6000
    assert result['omitted']['warnings']==40
    assert len(result['precedents'])<=3 and len(result['apis'])<=2
    assert result['omitted']['notice'] and result['full_context']['command']=='./matrix context'


def test_brief_essential_state_too_large_fails_explicitly(library):
    context=retrieve('DNa02',library)
    context['current_state']['stages']['3']='Unresolved; '+('complete condition '*1000)
    with pytest.raises(ValueError,match='Essential context needs'):
        brief_context(context,6000)
    for invalid in (0,2047,True,6000.0):
        with pytest.raises(ValueError,match='at least 2048'):
            brief_context(context,invalid)


def test_brief_cli_budget_matches_actual_stdout(library,capsys):
    from problem_context import main
    main(['DNa02','--root',str(library),'--brief','--max-chars','6000'])
    actual=capsys.readouterr().out
    result=json.loads(actual)
    assert result['schema']=='matrix_problem_brief_v1'
    assert len(actual)==result['budget']['used_chars']<=6000


def test_verbose_next_steps_cannot_starve_relevant_executable_api(library):
    context=retrieve('DNa02',library)
    context['current_state']['stages']={str(i):('Status and scope. '*12) for i in range(7)}
    context['current_next_work'][0]['declared_next']['next_decision']='Complete scientific limitation. '*70
    context['tools']=[dict(source='config/tools.json',entrypoints=[dict(path='src/diagnosis.py',
        symbols=['diagnose'],usage=['./matrix atlas diagnose'],relevance=100,limits=['Two reviewed protocols only'])])]
    result=brief_context(context,6000)
    assert result['apis'][0]['usage']==['./matrix atlas diagnose']
    assert result['state']['stages']==context['current_state']['stages']
    assert result['precedents'] and result['budget']['used_chars']<=6000


def install_scope_alert(library):
    import hashlib
    # Deliberately absent from the full-text index and current latest_* links.
    path = write(library, 'work/old_spatial/README.md',
                 'Passive transfer differs by location; does not establish firing or inhibition.')
    config = library/'config/olfactory_mobility_plan_v1.json'
    data = json.loads(config.read_text())
    data['model_scope_alerts'] = [dict(id='spatial_scope', query_terms=['DNa02'],
        statement='An isopotential negative does not refute spatial integration; passive transfer does not establish recruitment.',
        evidence=[dict(path=str(path.relative_to(library)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())])]
    config.write_text(json.dumps(data))
    return path


def test_scope_alert_survives_unindexed_evidence_and_crowded_brief(library):
    install_scope_alert(library)
    context=retrieve('DNa02 conductance',library,limit=1)
    context['scoped_decisions'] *= 30
    context['current_next_work'][0]['declared_next']['next_decision']='Other optional history. '*1000
    result=brief_context(context,6000)
    alert=result['model_scope_alerts'][0]
    assert alert['statement'].endswith('does not establish recruitment.')
    assert alert['evidence'][0]['identity']=='content_matches_receipt'
    assert result['budget']['used_chars']<=6000
    for query in ['CTr','DNa020']:
        assert not retrieve(query,library)['current_state']['model_scope_alerts']


def test_scope_alert_detects_changed_and_missing_evidence(library):
    path=install_scope_alert(library)
    path.write_text('Changed interpretation')
    result=brief_context(retrieve('DNa02',library))
    assert result['model_scope_alerts'][0]['evidence'][0]['identity']=='changed_since_receipt'
    path.unlink()
    result=retrieve('DNa02',library)
    assert result['current_state']['model_scope_alerts'][0]['evidence'][0]['identity']=='missing_or_unreadable'
    assert any('requires review' in warning for warning in result['warnings'])


def test_scope_alert_cannot_silently_disappear_to_fit_budget(library):
    install_scope_alert(library)
    context=retrieve('DNa02',library)
    context['current_state']['model_scope_alerts'][0]['statement']='Complete scoped finding. '*500
    with pytest.raises(ValueError,match='model scope alerts were not truncated'):
        brief_context(context,6000)


def test_default_plan_uses_explicit_recipe_question_when_title_omits_entity(library,monkeypatch,capsys):
    install_scope_alert(library)
    spec=importlib.util.spec_from_file_location('scope_workbench_cli',ROOT/'scripts/workbench.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    write(library,'config/workflows/fixture.json',dict(schema='matrix_workflow_v1',id='fixture',
        title='Conductance diagnostic',context_query='DNa02 conductance',scope='development only',
        steps=[dict(id='compare',component='compare_arrays',inputs={})]))
    write(library,'src/session_io.py','# identity fixture\n')
    monkeypatch.setattr(cli,'ROOT',library)
    monkeypatch.setattr(sys,'argv',['workbench.py','plan','fixture'])
    cli.main();result=json.loads(capsys.readouterr().out)
    assert result['advisory_context']['model_scope_alerts'][0]['id']=='spatial_scope'
    monkeypatch.setattr(sys,'argv',['workbench.py','plan','fixture','--context-query','CTr'])
    cli.main();result=json.loads(capsys.readouterr().out)
    assert not result['advisory_context']['model_scope_alerts']
def test_workflow_status_requires_complete_declared_coverage():
    from problem_context import receipt_execution_status
    receipt = {'schema':'matrix_workflow_execution_v1',
               'recipe':{'steps':[{'id':'a'},{'id':'b'}]},
               'steps':{'a':{'status':'COMPLETE'},'b':{'status':'COMPLETE'}}}
    assert receipt_execution_status(receipt) == 'COMPLETE'
    receipt['steps']['b']['status'] = 'FAILED'
    assert receipt_execution_status(receipt) == 'FAILED'
    del receipt['steps']['b']
    assert receipt_execution_status(receipt) == 'UNKNOWN'
    assert receipt_execution_status({'steps':{'a':{'status':'COMPLETE'}}}) == 'UNKNOWN'

