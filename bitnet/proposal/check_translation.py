#!/usr/bin/env python3
"""Check structural and literal fidelity of the Chinese companion. No inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tables(text):
    groups=[];current=[]
    for line in text.splitlines()+['']:
        if line.startswith('|'):current.append([x.strip() for x in line.strip('|').split('|')])
        elif current:groups.append(current);current=[]
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional new JSON file outside raw experimental data')
    args=parser.parse_args()
    en=(HERE/'PROPOSAL.md').read_text();zh=(HERE/'PROPOSAL_ZH.md').read_text()
    localized_date=re.sub(r'2026\s*年\s*11\s*月\s*(15|16)\s*日',lambda m: m.group(1)+' November 2026',zh)
    blocks=lambda s:re.findall(r'```[^\n]*\n(.*?)```',s,re.S)
    inline=lambda s:re.findall(r'(?<!`)`([^`\n]+)`(?!`)',re.sub(r'```.*?```','',s,flags=re.S))
    urls=lambda s:re.findall(r'\]\((https?://[^)]+)\)',s)
    refs=lambda s:re.findall(r'\[[ER]\d+\]',s)
    headers=lambda s:re.findall(r'^## (\d+) ',s,re.M)
    a,b=tables(en),tables(zh);table_errors=[]
    if len(a)!=len(b):table_errors.append('Different table count')
    cells=0
    for i,(t,u) in enumerate(zip(a,b),1):
        if len(t)!=len(u):table_errors.append(f'Table {i} has different row count')
        for j,(r,v) in enumerate(zip(t,u),1):
            if len(r)!=len(v):table_errors.append(f'Table {i} row {j} has different column count')
            for k,(c,d) in enumerate(zip(r,v),1):
                cells+=1
                # Word order may differ; signs and decimal expressions must not.
                pattern=r'[-+]?\d+(?:,\d{3})*(?:\.\d+)*(?:e[-+]?\d+)?%?'
                if Counter(re.findall(pattern,c))!=Counter(re.findall(pattern,d)):
                    table_errors.append(f'Table {i} row {j} column {k} numerical tokens differ')
    published=json.loads((HERE/'MANIFEST.json').read_text())
    old={x['path']:x['sha256'] for x in published['outputs']}
    preserved={p:sha(HERE/p)==old[p] for p in ('PROPOSAL.md','BitNet_Mobile_GPU_Proposal.pdf','BitNet_Mobile_GPU_Proposal.docx','assets/execution-flow.png','assets/execution-flow.svg')}
    checks={'section_numbers_equal':headers(en)==headers(zh)==[str(i) for i in range(1,17)],
            'code_blocks_identical':blocks(en)==blocks(zh),
            'inline_code_identical':inline(en)==inline(zh),'ordered_URLs_identical':urls(en)==urls(zh),
            'ordered_citations_identical':refs(en)==refs(zh),
            'all_digit_frequencies_equal_after_date_localization':Counter(re.findall(r'\d+',en))==Counter(re.findall(r'\d+',localized_date)),
            'table_shapes_and_numerical_cells_equal':not table_errors,
            'English_and_original_figure_preserved':all(preserved.values())}
    result={'checks':checks,'overall_pass':all(checks.values()),'table_errors':table_errors,
            'section_count':len(headers(zh)),'table_count':len(a),'table_cells_checked':cells,
            'code_blocks':len(blocks(en)),'inline_code_entries':len(inline(en)),
            'references':len(refs(en)),'URL_count':len(urls(en)),
            'English_file_checks':preserved,'English_source_sha256':sha(HERE/'PROPOSAL.md'),
            'Chinese_source_sha256':sha(HERE/'PROPOSAL_ZH.md'),
            'date_normalization':'Chinese and English evidence-cutoff dates normalize to the same day in November 2026',
            'scope':'Document translation literal/structure checks, not new numerical or model validation'}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:f.write(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 0 if result['overall_pass'] else 1

if __name__=='__main__':raise SystemExit(main())
