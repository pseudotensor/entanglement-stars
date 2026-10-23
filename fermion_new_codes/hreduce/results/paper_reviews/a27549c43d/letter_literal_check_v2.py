"""Literal document checks only; the copied live aux is an unbound witness."""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parent
letter=ROOT/'REVIEWA_write_letter_LT3_v2_updated.md'
aux=ROOT/'observed_live_build.aux'
text=letter.read_text()
body=text.split('\n\n',1)[1]
labels=dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}',aux.read_text()))
source_labels=set()
for p in (ROOT/'paper_sources').rglob('*.tex'):
    source_labels.update(re.findall(r'\\label\{([^}]+)\}',p.read_text()))
used=re.findall(r'`((?:eq|sec|app):[^`]+)`',text)
rows=[]
for line_no,line in enumerate(text.splitlines(),1):
    for m in re.finditer(r'Eqs?\.\s*((?:\([A-Z]?\d+\)(?:\s*(?:,|and|to)\s*)?)+)\s*\((`eq:[^)]*)\)',line):
        numbers=re.findall(r'\(([A-Z]?\d+)\)',m[1])
        refs=re.findall(r'`(eq:[^`]+)`',m[2])
        actual=[labels.get(ref) for ref in refs]
        rows.append(dict(line=line_no,kind='equation',printed=numbers,labels=refs,aux_values=actual,
                         decision='manual_range_or_count' if len(numbers)!=len(refs) else 'match' if numbers==actual else 'mismatch'))
    for m in re.finditer(r'(?:Sec\.|Section|Appendix)\s+([IVX]+(?:\s+[A-Z])?|[A-D])\s*\(`((?:sec|app):[^`]+)`\)',line):
        actual=labels.get(m[2]);normalized=None if actual is None else actual.replace(r'\,',' ').strip()
        rows.append(dict(line=line_no,kind='section_or_appendix',printed=m[1],labels=[m[2]],aux_values=[actual],
                         decision='match' if m[1]==normalized else 'mismatch'))
result=dict(scope='Literal counts, label existence and adjacent explicit label/number groups only; no claim or physics review',
    source_commit='a27549c43d342d32f702868e2e0aed47344b075f',
    letter_sha256=hashlib.sha256(letter.read_bytes()).hexdigest(),
    aux_sha256=hashlib.sha256(aux.read_bytes()).hexdigest(),
    aux_provenance='Copied unversioned live build at intake; not independently bound to named commit or PDF',
    characters=dict(raw=len(text),without_delivery_heading=len(body),cap=44475,
                    raw_excess=max(0,len(text)-44475),body_excess=max(0,len(body)-44475)),
    unique_used_labels=sorted(set(used)),
    missing_from_committed_tex=sorted(set(used)-source_labels),
    missing_from_observed_aux=sorted(set(used)-labels.keys()),
    adjacent_number_checks=rows,
    lexical_hits=[dict(line=i,token=m.group()) for i,line in enumerate(text.splitlines(),1)
                  for m in re.finditer(r'\b(?:Theorem|Lemma|Proposition|declared)\b|\(G[0-5]\)',line)],
    coverage_limit='Unlabelled references, range/count ambiguities, title/claim/source meaning and PDF binding require the assigned pair; a lexical hit is not itself a scope violation')
(ROOT/'letter_literal_check_v2.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(characters=result['characters'],used_labels=len(set(used)),missing_source_labels=result['missing_from_committed_tex'],
    adjacent_checks=len(rows),decisions={x:sum(r['decision']==x for r in rows) for x in ('match','mismatch','manual_range_or_count')})))
