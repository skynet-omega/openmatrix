"""Read-only, lexical problem retrieval over MATRIX's existing indexes.

This is planning advice, not an approval, scientific qualification, simulator or
new catalogue. Indexed code is never imported. Dynamic retrieval is deliberately
outside matrix_workbench's scientific source identity and execution/cache path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata


STOP = set("a al algo antes como con contra cual cuando de del el en es esta este esto etapa etapas hacer hay la las lo los mas me mismo necesito no o para por problema puedo que quiero se si sin sobre su sus un una usar y the of to for and is with how can from model modelo".split())
# Small, explicit bilingual vocabulary, not inferred scientific equivalences.
VOCABULARY = [
    {"pasivo", "pasiva", "passive"},
    {"transferencia", "transferir", "transfer", "transferred", "transferible"},
    {"radio", "radios", "radius", "radii"},
    {"unidad", "unidades", "unit", "units"},
    {"geometria", "geometry", "geometric", "geometrica"},
    {"morfologia", "morfologico", "morphology", "morphological"},
    {"impedancia", "impedance"},
    {"entrada", "entradas", "input", "inputs"},
    {"salida", "salidas", "output", "outputs"},
    {"sensorial", "sensory"},
    {"retorno", "feedback"},
    {"contexto", "context"},
    {"distribuido", "distribuida", "distributed"},
    {"poblacion", "poblaciones", "population", "populations"},
    {"recurrente", "recurrencia", "recurrent", "recurrence"},
    {"excitacion", "excitadora", "excitation", "excitatory"},
    {"inhibicion", "inhibidora", "inhibition", "inhibitory"},
    {"torsion", "torsional"},
    {"arbol", "bosque", "forest", "tree"},
    {"soma", "somatic", "somatico"},
    {"cable", "electrotonico", "electrotonic"},
]
TEXT_LIMIT = 262144
ROOT = Path(__file__).resolve().parents[1]


def tokens(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return set(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def concepts(query):
    words = tokens(query) - STOP
    groups = []
    for word in sorted(words):
        group = next((v for v in VOCABULARY if word in v), {word})
        if not any(group == g for g in groups):
            groups.append(group)
    if not groups:
        raise ValueError("Indicar al menos un término de contenido")
    if len(groups) > 16:
        raise ValueError("Consulta demasiado amplia: usar hasta 16 conceptos de contenido")
    return groups


def match(text, groups):
    words = tokens(text)
    return sum(bool(words & group) for group in groups)


def score(path, title, text, groups):
    coverage = match(path + ' ' + title + ' ' + text, groups)
    return (9 * match(title, groups) + 6 * match(path, groups) + 4 * match(text, groups)
            + 15 * max(0, coverage - 1))


def excerpt(text, groups, limit=1200):
    blocks = [b.strip() for b in re.split(r"\n\s*\n|\n", text) if b.strip()]
    ranked = sorted(enumerate(blocks), key=lambda x: (-match(x[1], groups), x[0]))
    selected = sorted(i for i, b in ranked[:4] if match(b, groups))
    value = " … ".join(blocks[i] for i in selected) if selected else text.strip()
    return value[:limit]


def read_small(root, path):
    """Only current small text inside root; never open indexed biological payloads."""
    p = Path(path)
    p = p if p.is_absolute() else root / p
    try:
        p = p.resolve()
        p.relative_to(root)
        if p.suffix.lower() not in {".md", ".json", ".py"} or p.stat().st_size > TEXT_LIMIT:
            return None
        raw = p.read_bytes()
        if len(raw) > TEXT_LIMIT:
            return None
        return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, UnicodeError):
        return None


def read_json(root, path, warnings):
    item = read_small(root, path)
    if item:
        try:
            return json.loads(item[0])
        except json.JSONDecodeError:
            warnings.append("Invalid JSON: " + str(path))
    return {}


def absolute(root, path):
    p = Path(path)
    if not p.is_absolute():
        # Scientific catalogue cards use paths relative to the parent project.
        if p.parts and p.parts[0] == root.name:
            p = Path(*p.parts[1:])
        p = root / p
    return str(p.resolve())


def search_path(root, path):
    p = Path(absolute(root, path))
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def connect(path, warnings):
    if not path.is_file():
        warnings.append("Index unavailable: " + str(path))
        return None
    try:
        con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        return con
    except sqlite3.Error as exc:
        warnings.append("Index unreadable: " + str(exc))
        return None


def fts_query(groups):
    return " OR ".join('"' + term + '"' for term in sorted(set().union(*groups)))


def scientific_rows(root, groups, warnings):
    con = connect(root / "catalogo/catalogo.sqlite", warnings)
    if con is None:
        return [], []
    rows, holdings = [], []
    try:
        # Separate pools prevent availability cards from crowding out evidence.
        for available in (False, True):
            inventory = "('Disponibilidad','Archivo','Archivo ZIP','ZIP anidado')"
            clause = "kind IN " + inventory if available else "kind NOT IN " + inventory
            sql = ("SELECT path,page,kind,title,text FROM documents WHERE documents MATCH ? AND "
                   + clause + " ORDER BY bm25(documents,5,0,0,7,1) LIMIT 36")
            found = []
            for subset in [groups] + [[g] for g in groups]:
                found.extend(dict(r) for r in con.execute(sql, (fts_query(subset),)))
            (holdings if available else rows).extend(found)
    except sqlite3.Error as exc:
        warnings.append("Scientific index query failed: " + str(exc))
    finally:
        con.close()
    return rows, holdings


def receipt_execution_status(receipt):
    """Read aggregate execution only when every declared workflow step is present."""
    if receipt.get('status'):
        return receipt['status']
    if receipt.get('schema') != 'matrix_workflow_execution_v1':
        return 'UNKNOWN'
    expected = [s.get('id') for s in receipt.get('recipe', {}).get('steps', [])]
    steps = receipt.get('steps', {})
    if not expected or len(set(expected)) != len(expected) or set(steps) != set(expected):
        return 'UNKNOWN'
    statuses = [s.get('status') for s in steps.values()]
    if all(s == 'COMPLETE' for s in statuses):
        return 'COMPLETE'
    if 'FAILED' in statuses:
        return 'FAILED'
    return 'UNKNOWN'


def current_context(root, groups, warnings):
    plan_path = "config/olfactory_mobility_plan_v1.json"
    plan = read_json(root, plan_path, warnings)
    predictor = read_json(root, "config/predictor_program_v2.json", warnings)
    index = read_small(root, "INDEX.md")
    state = dict(source=absolute(root, plan_path), active_priority=plan.get("active_priority"),
                 stages=plan.get("stage_status", {}), execution_hold=plan.get("execution_hold", {}))
    declared = plan.get('active_experiment') or plan.get('last_experiment')
    if declared:
        receipt_path = declared.get('execution')
        if not receipt_path and declared.get('cycle'):
            receipt_path = f"runs/workbench/decisions/{declared['cycle']}/state.json"
        receipt = read_json(root, receipt_path, warnings) if receipt_path else {}
        observed = receipt_execution_status(receipt)
        state['experiment'] = dict(id=declared.get('id', declared.get('cycle')),
            declared_status=declared.get('status'), execution_status=observed,
            receipt=absolute(root, receipt_path) if receipt_path else None,
            recorded_outcome=declared.get('outcome'),
            scope='Execution status is read from its receipt; it is not scientific admission.')
        if observed == 'UNKNOWN':
            warnings.append('Experiment execution status unavailable; do not infer it from stage prose.')
        elif observed != declared.get('status'):
            warnings.append('Experiment status mismatch: declared '+str(declared.get('status'))+
                            ', receipt '+str(observed)+'. Read the receipt before resuming.')
    state['model_scope_alerts'] = model_scope_alerts(root, plan, groups, warnings)
    from operational_check_report import current_report
    state['operational_tests']=current_report(root)
    if index:
        state["index"] = dict(path=absolute(root, "INDEX.md"), sha256=index[1],
                              excerpt=excerpt(index[0], groups, 1700))
    pointers = {v for obj in (plan, predictor) for k, v in obj.items()
                if "next" in k and isinstance(v, str) and v.endswith("NEXT_WORK.json")}
    next_work, constraints = [], []
    for path in sorted(pointers):
        content = read_small(root, path)
        obj = read_json(root, path, warnings)
        if not obj:
            warnings.append("Current NEXT_WORK unreadable: " + path)
            continue
        row = dict(path=absolute(root, path), basis="explicit current configuration pointer, not date",
                   sha256=content[1] if content else None,
                   relevance=score(path, "", json.dumps(obj, ensure_ascii=False), groups))
        fields = ("objective", "next_decision", "next_work_contract", "reusable_start", "boundary",
                  "first_bounded_comparison", "gate", "next", "stage3_criterion")
        row["declared_next"] = {k: obj[k] for k in fields if k in obj}
        row['linked_evidence'] = []
        def links(value):
            if isinstance(value, dict):
                for v in value.values():
                    links(v)
            elif isinstance(value, list):
                for v in value:
                    links(v)
            elif isinstance(value, str) and value.endswith('.md') and '\n' not in value:
                if read_small(root, value):
                    row['linked_evidence'].append(absolute(root, value))
        links({k:v for k,v in obj.items() if k in {'entry','next_work_contract','transfer_readiness'}
               or k.startswith(('_latest', 'latest_'))})
        row['linked_evidence'] = sorted(set(row['linked_evidence']))[:16]
        next_work.append(row)
        for key, value in obj.items():
            if key in {"do_not_repeat", "closed", "boundary", "limits", "scope", "non_claims"}:
                values = value if isinstance(value, list) else [value]
                for v in values:
                    text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    constraints.append(dict(source=row["path"], field=key, text=text,
                        relevance=score(path, "", text, groups), universal_prohibition=False,
                        basis="declared statement; scope must be read at source"))
    next_work.sort(key=lambda r: (-r["relevance"], r["path"]))
    constraints.sort(key=lambda r: (-r["relevance"], r["source"], r["text"]))
    return state, next_work, [c for c in constraints if c['relevance'] > 0][:10]


def model_scope_alerts(root, plan, groups, warnings):
    """Explicit current model limits, independent of lexical evidence ranking.

    Matching is by complete declared term (not a substring or inferred biological
    equivalence). Receipts verify document identity only, never the claim itself.
    No historical configuration activates an alert.
    """
    rows = []
    query_words = set().union(*groups)
    for record in plan.get('model_scope_alerts', []):
        if not isinstance(record, dict) or not all(k in record for k in
                ('id', 'query_terms', 'statement', 'evidence')):
            raise ValueError('Invalid current model scope alert')
        if not record['query_terms'] or not isinstance(record['statement'], str) or not record['statement']:
            raise ValueError('Model scope alert needs terms and a complete statement')
        if not any(tokens(term) and tokens(term) <= query_words for term in record['query_terms']):
            continue
        refs = []
        for evidence in record['evidence']:
            path, digest = evidence['path'], evidence['sha256']
            live = read_small(root, path)
            status = ('missing_or_unreadable' if not live else
                      'content_matches_receipt' if live[1] == digest else 'changed_since_receipt')
            refs.append(dict(source=absolute(root, path), identity=status))
            if status != 'content_matches_receipt':
                warnings.append('Model scope evidence requires review: ' + path + ' (' + status + ')')
        if not refs:
            raise ValueError('Model scope alert needs evidence references')
        rows.append(dict(id=record['id'], statement=record['statement'], evidence=refs,
                         scope='Declared model limitation; neither causal finding nor execution prohibition'))
    return rows


def documents(root, groups, raw, limit, warnings):
    """Deduplicate actual document/page; small live documents supersede excerpts."""
    ranked, seen = [], set()
    current_links = {absolute(root, r['path']):r['live_context_link'] for r in raw if r.get('live_context_link')}
    for r in raw:
        path = absolute(root, r["path"])
        s = score(search_path(root, path), r.get("title", ""), r.get("text", ""), groups)
        if not s:
            continue
        s += 5 if r.get("kind") == "Ficha" else 0
        s += 12 if path in current_links else 0
        s -= 12 if Path(path).name.lower() == "manifest.json" else 0
        ranked.append((s, path, r))
    # Current explicit cards replace their stale indexed duplicate even if the
    # older wording happens to match more query words.
    live_keys = {(absolute(root, r['path']), r.get('page', 0)) for r in raw if r.get('live_card')}
    ranked = [x for x in ranked if x[2].get('live_card') or
              (x[1], x[2].get('page', 0)) not in live_keys]
    ranked.sort(key=lambda x: (-x[0], x[1], x[2].get("page", 0)))
    result, per_folder = [], {}
    for s, path, r in ranked:
        key = (path, r.get("page", 0))
        if key in seen:
            continue
        seen.add(key)
        folder = str(Path(path).parent)
        if per_folder.get(folder, 0) >= 2:
            continue
        per_folder[folder] = per_folder.get(folder, 0) + 1
        p = Path(path)
        item = dict(path=path, page=r.get("page", 0), title=r.get("title", ""),
                    kind=r.get("kind", ""), relevance=s,
                    basis="indexed historical evidence; neither date nor seal establishes applicability",
                    available=p.is_file(), indexed_excerpt=excerpt(r.get("text", ""), groups))
        # JSON configuration is read only at declared navigation entrypoints.
        # Search hits may name arrays encoded as JSON/TXT: keep indexed text and
        # never open arbitrary response files while gathering planning advice.
        live = read_small(root, path) if p.suffix.lower() in {'.md', '.py'} else None
        if live and r.get("kind") not in {"Disponibilidad", "Ficha"}:
            item.update(excerpt=excerpt(live[0], groups), current_text_sha256=live[1],
                        freshness="small source read live; ranking still uses index")
        else:
            item.update(excerpt=item["indexed_excerpt"], freshness="indexed extraction; content not revalidated")
        if r.get('live_card'):
            item['freshness'] = 'live catalogue card; original evidence not revalidated'
        if path in current_links:
            item['current_reference'] = current_links[path]
        if r.get("kind") == "Ficha":
            try:
                card = json.loads(r["text"])
                item["reported_decision"] = {k: card[k] for k in ("status", "limit", "problem") if k in card}
            except (ValueError, TypeError):
                pass
        result.append(item)
        if len(result) >= limit:
            break
    return result


def code_candidates(root, groups, limit, warnings, tools=()):
    con = connect(root / "catalogo/code.sqlite", warnings)
    if con is None:
        return []
    rows = []
    declared = {search_path(root, e['path']) for t in tools for e in t['entrypoints']
                if e.get('path') and e.get('relevance', 0) > 0}
    try:
        # Query each concept independently so a common entity cannot swamp a
        # mechanism with a different biological name (e.g. passive cable DM1).
        candidates = {}
        for group in groups:
            try:
                found = []
                # Three bounded pools: symbols encoded in filenames, active
                # source, and historical use. Frequency in archived snapshots
                # must not hide a small reusable implementation.
                for query, clause in ((" OR ".join('path:"'+t+'"' for t in sorted(group)), ""),
                                      (fts_query([group]), " AND f.base_role='active_source'"),
                                      (fts_query([group]), "")):
                    found.extend(con.execute("SELECT f.* FROM file_fts JOIN files f ON f.id=file_fts.rowid "
                        "WHERE file_fts MATCH ?" + clause + " ORDER BY bm25(file_fts,8,1,2) LIMIT 28",
                        (query,)).fetchall())
            except sqlite3.OperationalError:
                # Existing AST catalogue may have been built without FTS5.
                expr = "lower(path || char(10) || source || char(10) || semantic)"
                where = " OR ".join("instr(" + expr + ",?)>0" for _ in group)
                found = con.execute("SELECT * FROM files WHERE " + where + " ORDER BY path LIMIT 120", sorted(group)).fetchall()
                warnings.append("Code FTS unavailable: bounded lexical fallback, possibly incomplete")
            for r in found:
                if match(r["path"] + " " + r["source"] + " " + r["semantic"], groups):
                    candidates[r["id"]] = dict(r)
        # Relevant live contracts provide a separate discovery pool. Their
        # implementations can be absent from the FTS shortlist (or newly
        # registered); registration is not a scientific qualification.
        for path in sorted(declared):
            r = con.execute('SELECT * FROM files WHERE path=?', (path,)).fetchone()
            if r is not None:
                candidates[r['id']] = dict(r)
        # Resolve exact indexed copies BEFORE the bounded file shortlist. A
        # long historical path must not outrank its identical active source.
        hashes = sorted({r['sha256'] for r in candidates.values() if r['sha256']})
        for start in range(0, len(hashes), 200):
            batch = hashes[start:start+200]
            sql = ('SELECT * FROM files WHERE base_role=\'active_source\' AND sha256 IN ('
                   + ','.join('?' for _ in batch) + ')')
            for r in con.execute(sql, batch):
                candidates[r['id']] = dict(r)
        by_content = {}
        for f in candidates.values():
            by_content.setdefault(f['sha256'] or ('unhashed', f['id']), []).append(f)
        candidates = {}
        for copies in by_content.values():
            copies.sort(key=lambda f: (not Path(absolute(root, f['path'])).is_file(),
                f['path'] not in declared, f['base_role'] != 'active_source',
                len(Path(f['path']).parts), f['path']))
            preferred = copies[0]
            preferred['same_indexed_content_at'] = [absolute(root, f['path']) for f in copies[1:]]
            candidates[preferred['id']] = preferred
        files = sorted(candidates.values(), key=lambda f: (
            f['path'] not in declared,
            -score(f["path"], "", f["source"][:TEXT_LIMIT], groups)
            - (3 if f["base_role"] == "active_source" else 0), f["path"]))
        for f in files[:max(18, limit * 3)]:
            entries = [dict(e) for e in con.execute("SELECT * FROM entries WHERE file_id=? "
                "AND kind IN ('function','async_function','class','import','call')", (f["id"],))]
            entries.sort(key=lambda e: (-score(f["path"], e["qualname"], e["search_text"], groups)
                                       - (3 if e["kind"] in {"function", "class", "async_function"} else 0), e["line"]))
            best = entries[0] if entries else {}
            path = absolute(root, f["path"])
            live = read_small(root, path)
            freshness = ("content_matches_index" if live and live[1] == f["sha256"] else
                         "changed_since_index" if live else "not_content_verified")
            s = score(f["path"], best.get("qualname", ""), f["source"][:TEXT_LIMIT], groups)
            if best.get('kind') in {'class', 'function', 'async_function'}:
                # Declaring the requested operation carries more evidence of
                # reuse than listing all query words somewhere in a long file.
                s += 12 * match(best.get('qualname', ''), groups)
            s += 3 if f["base_role"] == "active_source" else 0
            rows.append(dict(path=path, line=best.get("line", 1), symbol=best.get("qualname", ""),
                signature=best.get("signature", ""), role=f["role"], seal=f["seal"] or None,
                parse_error=f["error"] or None, indexed_sha256=f["sha256"], freshness=freshness,
                available=Path(path).is_file(), relevance=s,
                excerpt=excerpt(f["source"], groups, 700),
                discovery="lexical candidate; no import or execution; compatibility and validation not inferred"))
            rows[-1]['declared_entrypoint'] = f['path'] in declared
            if f.get('same_indexed_content_at'):
                rows[-1]['same_indexed_content_at'] = f['same_indexed_content_at']
    except sqlite3.Error as exc:
        warnings.append("Code query failed: " + str(exc))
    finally:
        con.close()
    rows.sort(key=lambda r: (not r['declared_entrypoint'], -r["relevance"], r["path"]))
    unique = {}
    for row in rows:
        key = (row['indexed_sha256'], row['symbol'])
        if key in unique:
            unique[key].setdefault('same_indexed_content_at', []).append(row['path'])
        else:
            unique[key] = row
    return list(unique.values())[:limit]


def declared_tools(root, groups, warnings):
    path = "config/tooling_capabilities_v1.json"
    tools = read_json(root, path, warnings).get("capabilities", [])
    results = []
    for tool in tools:
        body = json.dumps(tool, ensure_ascii=False)
        if not match(body, groups):
            continue
        entries = []
        for entry in tool.get("entrypoints", []):
            # Example commands and historical references are not declarations
            # that this API implements their biological subject. In particular,
            # a search tool's example query must not recommend the search tool.
            text = json.dumps({k: entry[k] for k in ('id', 'path', 'symbols', 'purpose')
                               if k in entry}, ensure_ascii=False)
            if match(text, groups):
                selected = {k: entry[k] for k in ("id", "path", "symbols", "purpose", "usage", "references", "tests", "limits", "source_sha256") if k in entry}
                live = read_small(root, entry.get('path', '')) if entry.get('path') else None
                selected['relevance'] = score(entry.get('path', ''), ' '.join(entry.get('symbols', [])),
                                              text + (' ' + live[0] if live else ''), groups)
                selected['ranking_basis'] = 'relevant declared entrypoint; small source text when available; not validation'
                if selected.get('path'):
                    selected['path'] = absolute(root, selected['path'])
                entries.append(selected)
        entries.sort(key=lambda e: (-e['relevance'], e.get('path', '')))
        results.append(dict(id=tool["id"], source=absolute(root, path),
            state=tool.get("state"), gap=tool.get("gap"), decision=tool.get("decision"),
            evidence=tool.get("evidence"), entrypoints=entries[:5],
            relevance=score("", tool.get("capability", ""), body, groups),
            compatibility="declared evidence and limits; not automatically certified for this query"))
    results.sort(key=lambda r: (-max((e['relevance'] for e in r['entrypoints']), default=0),
                               -r["relevance"], r["id"]))
    return results[:5]


def retrieve(query, root=ROOT, limit=6):
    root = Path(root).resolve()
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    groups, warnings = concepts(query), []
    state, next_work, constraints = current_context(root, groups, warnings)
    raw, holdings = scientific_rows(root, groups, warnings)
    campaign_evidence=[]
    if (root/'config/lab_evidence_v1.json').is_file():
        try:
            from lab_evidence import search as lab_search
            for record in lab_search(root,query,min(limit,6)):
                if record['entry']:
                    campaign_evidence.append(dict(path=absolute(root,record['entry']),
                        record_id=record['id'],freshness=record['metadata_integrity'],
                        reported_decision={'fields':record['reported_fields'],
                            'scope':'Literal attributed metadata, not new validation or stage admission'},
                        discovery='incremental laboratory index; raw-array verification separate'))
        except (OSError,ValueError,sqlite3.Error) as exc:
            warnings.append('Laboratory index unavailable: '+str(exc))
    for current in next_work:
        for path in current['linked_evidence']:
            live = read_small(root, path)
            if live:
                raw.append(dict(path=path, page=0, kind='Documento',
                    title=Path(path).parent.name, text=live[0], live_context_link=current['path']))
    # Live existing cards cover recent registrations before a refresh; they do
    # not replace independent full-text and code-index retrieval.
    cards = read_json(root, "config/scientific_catalog_current_parts.json", warnings).get("parts", [])
    for card in cards:
        text = json.dumps(card, ensure_ascii=False)
        if match(text, groups):
            for path in card.get("files", [])[:1]:
                raw.append(dict(path=path, page=0, kind="Ficha", title=card.get("problem", ""), text=text, live_card=True))
    evidence = documents(root, groups, raw, limit, warnings)
    tools = declared_tools(root, groups, warnings)
    code = code_candidates(root, groups, limit, warnings, tools)
    from decision_cycles import recalled_decisions
    memory = recalled_decisions(root, set().union(*groups), min(limit, 3), warnings)
    recorded = []
    for row in memory:
        recorded.append(dict(source=row['source'],field='recorded_diagnostic_decision',
            text=json.dumps({k:row[k] for k in ('question','outcome','scope','next_action','hypotheses','evidence_mode')},ensure_ascii=False),
            relevance=row['relevance'],universal_prohibition=False,basis=row['basis'],
            recorded_outcome=row['outcome'],
            hypotheses=[{'id':h['id'],'outcome':h['outcome']} for h in row['hypotheses']]))
    constraints = recorded + constraints
    return dict(schema="matrix_problem_context_v1", query=query,
        advisory_only=True, executes_science=False, scientific_cache_key_effect="none",
        retrieval=dict(method="bounded lexical FTS plus explicit bilingual aliases and relevant declared entrypoints; "
                              "canonical preference for exact indexed copies; token boundaries; no date preference",
                       concepts=[sorted(g) for g in groups], limit_per_section=limit),
        current_state=state, current_next_work=next_work, scoped_decisions=constraints,
        decision_memory=memory,
        evidence=evidence, campaign_evidence=campaign_evidence, code_candidates=code, tools=tools,
        availability=documents(root, groups, holdings, min(3, limit), warnings),
        warnings=sorted(set(warnings)),
        limits=["Retrieval is incomplete; zero hits are not proof of absence.",
                "Negative results are scoped evidence, not universal prohibitions.",
                "Historical NEXT_WORK and COMPLETE runs are not current priority or scientific qualification.",
                "Indexed text can lag changed files; live state comes from explicit current pointers.",
                "Match does not identify units, ports, preparation, morphology convention or observer compatibility.",
                "Only small local text is read; no biological arrays, downloads, imports of indexed code or simulations."])


def brief_context(context, max_chars=6000):
    """Compact planning advice, with whole statements and an exact JSON budget.

    The budget counts ``json.dumps(result, ensure_ascii=False, indent=2)`` plus
    one terminal newline (Unicode characters, not bytes or model tokens).
    All declared stage states, priority, question and current NEXT_WORK paths
    are essential. If they cannot fit, raise ValueError; never shorten them.
    Other statements are copied whole or replaced by a reference and omission
    status. Retrieval excerpts are not copied: they may already be truncated.
    This pure presentation function performs no IO and does not mutate context.
    """
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 2048:
        raise ValueError('max_chars must be an integer of at least 2048 characters')
    if not isinstance(context, dict) or not isinstance(context.get('query'), str):
        raise ValueError('context must be a retrieval object with a string query')
    try:
        json.dumps(context, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('context must contain finite JSON values') from exc

    current = context.get('current_state', {})
    next_rows = context.get('current_next_work', [])
    # A source is represented once. A scoped decision has priority over an
    # excerpt from the same source; neither becomes an independent observation.
    precedent_rows, seen = [], set()
    decisions = list(context.get('scoped_decisions', []))
    evidence = list(context.get('evidence', []))
    # Keep the ranked decision receipts together: interleaving document hits
    # must not hide a retained failure behind its followup (or vice versa).
    recorded = [r for r in decisions if r.get('recorded_outcome')]
    other = [r for r in decisions if not r.get('recorded_outcome')]
    ordered = context.get('campaign_evidence',[]) + recorded + other[:1] + evidence + other[1:]
    for row in ordered:
        source = row.get('source') or row.get('path')
        if not source or source in seen:
            continue
        seen.add(source)
        evaluation = ({'source_field': row.get('field', 'unspecified'),
                       'statement': row['text'], 'scope_rule': row.get('basis'),
                       'universal_prohibition': row.get('universal_prohibition', False)}
                      if 'text' in row else row.get('reported_decision'))
        reference={'source': source,
            'kind': 'reported_evaluation' if evaluation else 'evidence_reference',
            'details': 'omitted; consult complete source, not the retrieved excerpt'}
        if row.get('recorded_outcome'):
            reference['diagnostic_outcome']=row['recorded_outcome']
            reference['outcome_scope']='declared criteria only; conditions at source, no stage admission'
            if row.get('hypotheses'):
                reference['hypotheses']=row['hypotheses']
                reference['hypothesis_scope']='individual outcomes; never inherit the overall result; read claims and conditions at source'
        precedent_rows.append((reference, evaluation))

    api_rows, seen = [], set()
    # The short view has only two API slots. Prefer relevant current contracts
    # to incidental mentions in snapshots; retain history in full retrieval.
    declared_apis = []
    for tool in context.get('tools', []):
        for row in tool.get('entrypoints', []):
            declared_apis.append((tool, row))
    declared_apis.sort(key=lambda x: (-x[1].get('relevance', 0), x[1].get('path', '')))
    for tool, row in declared_apis:
        source = row.get('path')
        if not source or source in seen:
            continue
        seen.add(source)
        api_rows.append(({'source': source, 'contract_source': tool.get('source'),
            'symbol': row.get('symbols'), 'kind': 'declared_API_reference',
            'usage': row.get('usage', [])[:1],
            'applicability': 'conditions omitted; consult contract before reuse'},
            {'limits': row.get('limits', []), 'tests': row.get('tests', [])}))
    for row in context.get('code_candidates', []):
        if not row.get('path') or row['path'] in seen:
            continue
        seen.add(row['path'])
        api_rows.append(({'source': row['path'], 'line': row.get('line'),
            'symbol': row.get('symbol'), 'kind': 'lexical_implementation_candidate',
            'applicability': 'not inferred; read source and its tests before reuse'},
            {'freshness': row.get('freshness'), 'discovery': row.get('discovery')}))

    out = dict(schema='matrix_problem_brief_v1', advisory_only=True,
        question=context['query'],
        state=copy.deepcopy({k: current.get(k) for k in
            ('source', 'active_priority', 'stages', 'execution_hold', 'experiment')
            if k != 'experiment' or k in current}),
        navigation_entry=current.get('index',{}).get('path'),
        next_work=[{'source': r.get('path'), 'kind': 'declared_current_reference',
                    'details': 'omitted; consult source',
                    'omitted_fields': list(r.get('declared_next', {}))} for r in next_rows],
        model_scope_alerts=copy.deepcopy(current.get('model_scope_alerts', [])),
        precedents=[], apis=[], warnings=[],
        limits=['No state updates, scientific qualification or execution.',
                'A negative result is scoped evidence, not a universal prohibition.',
                'A lexical match does not establish compatible units, preparation, ports or observer.',
                'Potentially truncated retrieval excerpts are not restated as claims.'],
        full_context={'command': './matrix context', 'query': 'use question above without --brief',
                      'note': 'Retrieval is live; preserve the full receipt when exact historical context matters.'},
        omitted={'next_work_details': len(next_rows), 'precedents': len(precedent_rows),
                 'precedent_details': len(precedent_rows), 'apis': len(api_rows),
                 'api_details': len(api_rows), 'warnings': len(context.get('warnings', [])),
                 'additional_records_at_same_source': len(ordered)-len(precedent_rows),
                 'availability_records': len(context.get('availability', [])),
                 'notice': 'Omitted items remain accessible through full context and cited sources.'},
        budget={'max_chars': max_chars, 'used_chars': 0,
                'format': 'json.dumps(ensure_ascii=False,indent=2) plus one LF'})

    if 'experiment' in out['state']:
        experiment = out['state']['experiment']
        out['state']['experiment'] = {k:experiment[k] for k in
            ('execution_status', 'receipt', 'recorded_outcome')}
        if experiment['declared_status'] != experiment['execution_status']:
            out['state']['experiment']['declared_status'] = experiment['declared_status']

    def measured(value):
        while True:
            n = len(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)) + 1
            if value['budget']['used_chars'] == n:
                return n
            value['budget']['used_chars'] = n

    required = measured(out)
    if required > max_chars:
        raise ValueError(f'Essential context needs {required} characters; max_chars={max_chars}. '
                         'Priority, stage states, question, current references and model scope alerts were not truncated.')

    def admit(change):
        nonlocal out
        candidate = copy.deepcopy(out)
        change(candidate)
        if measured(candidate) <= max_chars:
            out = candidate
            return True
        return False

    # Reserve an actionable reference and a precedent before optional prose.
    # Otherwise verbose stage/next descriptions can silently consume every API
    # slot in a real 6000-character brief. Essential state above stays intact.
    selected=[]; admitted_apis=set(); admitted_precedents=set()
    def add_api_reference(i):
        ref,detail=api_rows[i]
        def change(v):
            v['apis'].append(copy.deepcopy(ref));v['omitted']['apis']-=1
        if admit(change):
            admitted_apis.add(i)
            selected.append(('apis',len(out['apis'])-1,detail,'api_details'))
    def add_precedent_reference(i):
        ref,detail=precedent_rows[i]
        def change(v):
            v['precedents'].append(copy.deepcopy(ref));v['omitted']['precedents']-=1
        if admit(change):
            admitted_precedents.add(i)
            selected.append(('precedents',len(out['precedents'])-1,detail,'precedent_details'))
    if api_rows:add_api_reference(0)
    if precedent_rows:add_precedent_reference(0)

    # A complete next statement carries its boundary; omit whole, never slice.
    for i, row in enumerate(next_rows):
        declared = row.get('declared_next', {})
        detail = {k: declared[k] for k in ('objective', 'next_decision', 'boundary') if k in declared}
        if not detail:
            detail = declared
        if detail:
            omitted_fields = [k for k in declared if k not in detail]
            def add_next(v, i=i, detail=detail, omitted_fields=omitted_fields):
                v['next_work'][i]['details'] = copy.deepcopy(detail)
                v['next_work'][i]['omitted_fields'] = list(omitted_fields)
                if not omitted_fields:
                    v['omitted']['next_work_details'] -= 1
            admit(add_next)
    for warning in context.get('warnings', []):
        def add_warning(v, warning=warning):
            v['warnings'].append(copy.deepcopy(warning));v['omitted']['warnings'] -= 1
        admit(add_warning)

    for i in range(min(3,len(precedent_rows))):
        if i not in admitted_precedents:add_precedent_reference(i)
    for i in range(min(2,len(api_rows))):
        if i not in admitted_apis:add_api_reference(i)
    for section, i, detail, counter in selected:
        if detail:
            def add_detail(v, section=section, i=i, detail=detail, counter=counter):
                v[section][i]['details'] = copy.deepcopy(detail)
                v['omitted'][counter] -= 1
            admit(add_detail)
    measured(out)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="+")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument('--brief', action='store_true', help='Bounded planning view; statements are never sliced')
    parser.add_argument('--max-chars', type=int, default=6000,
                        help='Total brief JSON characters including final LF (minimum 2048)')
    args = parser.parse_args(argv)
    try:
        result = retrieve(" ".join(args.query), args.root, args.limit)
        if args.brief:
            result = brief_context(result, args.max_chars)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
