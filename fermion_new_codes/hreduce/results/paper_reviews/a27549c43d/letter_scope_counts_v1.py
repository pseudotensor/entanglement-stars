"""Requested mechanical scope illustrations; no editorial exclusion is authorized."""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parent
source=ROOT/'REVIEWA_write_letter_LT3_v2.md'
text=source.read_text()
body=text.split('\n\n',1)[1]
marker='**SOURCES:**\n\n- Sec. I, Introduction'
assert body.count(marker)==1
first=body.index(marker)
last=body.index('# Part II.',first)
without_list=body[:first].rstrip()+'\n\n'+body[last:]
without_sources=re.sub(r'[ \t]+\*\*SOURCES:\*\*[^\n]*','',without_list)
without_labels=re.sub(r'`(?:eq|sec|app):[^`]+`','',body)
result=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cap=44475,
    scope='Character counts only; the original printable scope is unchanged and no omission is authorized',
    variants={
      'raw_delivery':dict(characters=len(text),rule='Entire source including delivery heading'),
      'letter_body':dict(characters=len(body),rule='Remove delivery heading and its following blank line only'),
      'body_without_sources_tails_and_partI_source_list':dict(characters=len(without_sources),
          rule='Remove inline SOURCES marker/tails to each line end; remove Part I source-list block up to, but not including, the Part II heading; join preceding paragraph and Part II by two newlines'),
      'body_without_backtick_source_labels_only':dict(characters=len(without_labels),
          rule='Delete only each complete backtick-delimited eq:/sec:/app: label token; preserve all remaining spaces and punctuation')})
for row in result['variants'].values():row.update(excess=max(0,row['characters']-44475),margin=44475-row['characters'])
(ROOT/'letter_scope_counts_v1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['variants'],indent=2))
