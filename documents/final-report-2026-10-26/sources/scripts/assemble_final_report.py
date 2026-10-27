"""Assemble the final report from immutable saved evidence; never loads model data."""
from pathlib import Path
import csv, hashlib, html, json, re, shutil
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'documents/final-report-2026-10-26'
A = ROOT / 'documents/adaptation-2026-10-24'
R = ROOT / 'documents/retention-2026-10-25'
copied = {}
def copy(src, relative):
    src = Path(src)
    dst = OUT / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    copied[relative] = {'source': str(src.relative_to(ROOT)), 'sha256': hashlib.sha256(src.read_bytes()).hexdigest()}

figures = [
 (A/'analysis/mlm-selection-audit', 'figure-1-mlm', 'MLM selection and independent audit'),
 (A/'analysis/development-duration', 'figure-2-duration', 'Viral development outcomes versus budget'),
 (A/'analysis/endpoint-effects', 'figure-3-effects', 'Endpoint effects and k-mer comparison'),
 (R/'results/family-retention', 'figure-4-families', 'Benchmark families and overall retention'),
 (R/'results/task-retention', 'figure-5-tasks', 'All 18 task effects'),
 (R/'results/benefit-retention', 'figure-6-benefit-retention', 'Viral benefit and benchmark retention'),
 (A/'analysis/pilot-resource-performance', 'figure-S1-lora', 'Exploratory LoRA resources and performance')]
for src, name, _ in figures:
    for ext in ['png','pdf']:
        copy(src.with_suffix('.'+ext), f'figures/{name}.{ext}')
for src, name in [
 (A/'analysis/paired-effects.md','adaptation-paired-effects.md'),
 (A/'analysis/paired-effects.json','adaptation-paired-effects.json'),
 (A/'analysis/readout-scores.csv','adaptation-readout-scores.csv'),
 (A/'analysis/mlm-all-seeds.md','adaptation-mlm-all-seeds.md'),
 (A/'analysis/pilot-resources.json','lora-resources.json'),
 (R/'results/paired-effects.md','retention-paired-effects.md'),
 (R/'results/paired-effects.json','retention-paired-effects.json'),
 (R/'results/scores.csv','retention-scores.csv'),
 (R/'results/task-retention.csv','retention-task-decisions.csv'),
 (R/'results/summary.json','retention-summary.json')]:
    copy(src, 'tables/'+name)
for folder, names in [(A,['protocol.md','jobs-completed.psv']), (R,['protocol.md','pre-test-freeze-verification.json','frozen-settings-before-test.json','local-verification.json','compute-cost.md','compute-cost.json','jobs.psv'])]:
    label = 'adaptation' if folder == A else 'retention'
    for name in names:
        copy(folder/name,f'evidence/{label}/{name}')
for src, rel in [
 (A/'analysis/verification.json','adaptation/analysis-verification.json'),
 (A/'results/verification.json','adaptation/server-verification.json'),
 (A/'analysis/manifest.json','adaptation/analysis-manifest.json'),
 (R/'results/verification.json','retention/server-verification.json'),
 (R/'results/manifest.json','retention/server-manifest.json'),
 (R/'results/fresh-replay.json','retention/fresh-replay.json'),
 (R/'results/dataset-audit.json','retention/dataset-audit.json'),
 (ROOT/'documents/repair-2026-10-13/README.md','audits/dataset-repair.txt'),
 (ROOT/'documents/cds-audit-2026-10-21/README.md','audits/cds-audit.txt'),
 (ROOT/'documents/experiments-2026-10-20/sampling-audit.json','audits/sampling-audit.json')]:
    copy(src, 'evidence/'+rel)
# Retain explanatory execution records as verbatim plain-text snapshots: their original
# relative references point into the full repository, rather than this portable subset.
for folder, label in [(A,'adaptation'),(R,'retention')]:
    copy(folder/'execution.md',f'evidence/{label}/execution.txt')
    copy(folder/'interpretation.md',f'evidence/{label}/interpretation.txt')
source_names = ['interpret_adaptation_duration.py','summarize_adaptation_duration.py',
 'prepare_nt_retention.py','extract_nt_retention.py','select_nt_retention.py',
 'evaluate_nt_retention.py','replay_nt_retention.py','finalize_nt_retention.py',
 'plot_nt_retention.py','verify_nt_retention_local.py','collect_nt_retention_accounting.sh',
 'summarize_nt_retention_cost.py','assemble_final_report.py']
for name in source_names:
    copy(ROOT/'scripts'/name, 'sources/scripts/'+name)
copy(ROOT/'src/baculo_adapt/retention.py','sources/src/baculo_adapt/retention.py')
copy(ROOT/'tests/test_retention.py','sources/tests/test_retention.py')
copy(ROOT/'pyproject.toml','sources/pyproject.toml')

report = OUT/'report.md'
text = report.read_text(encoding='utf-8-sig').replace('evidence/audits/dataset-repair.md','evidence/audits/dataset-repair.txt').replace('evidence/audits/cds-audit.md','evidence/audits/cds-audit.txt')
report.write_text(text,encoding='utf-8')
assets = ['# Figures and result tables','', 'The original checked figures are reproduced byte for byte. PDF files retain the original export quality; PNG files support viewing. Captions and statistical qualifications are in the final report.','']
for _,name,title in figures:
    assets.append(f'- {title}: [PDF](figures/{name}.pdf) · [PNG](figures/{name}.png)')
assets += ['', '## Complete result tables', '']
for p in sorted(p for p in (OUT/'tables').iterdir() if p.suffix != '.html'):
    assets.append(f'- [{p.name}](tables/{p.name})')
assets += ['', '[Return to final report](report.md)', '']
(OUT/'assets.md').write_text('\n'.join(assets),encoding='utf-8')
repro = '''# Reproducibility and evidence index

This is a portable reporting package assembled from completed evidence on October 26, 2026. It contains no new training, feature extraction or test prediction. Scientific source state before assembly: commit `9b67df4`; adaptation interpretation: `f5c94dd`; retention implementation/freeze: `4bf23e2` and `6a40782`.

## Verified results

- [Adaptation independent verification](evidence/adaptation/analysis-verification.json): 14 scientific runs, 59 readouts, replay tolerance and intervals.
- [Adaptation server verification](evidence/adaptation/server-verification.json) and [analysis manifest](evidence/adaptation/analysis-manifest.json).
- [Retention server verification](evidence/retention/server-verification.json), [local independent verification](evidence/retention/local-verification.json), and [fresh-process replay](evidence/retention/fresh-replay.json): 126 readouts and all statistical bounds.
- [Retention server manifest](evidence/retention/server-manifest.json), [dataset audit](evidence/retention/dataset-audit.json), [pre-test freeze](evidence/retention/pre-test-freeze-verification.json), and [frozen settings](evidence/retention/frozen-settings-before-test.json).
- [Figures and all result tables](assets.md).

## Protocol, execution and costs

- [Adaptation protocol](evidence/adaptation/protocol.md), [execution snapshot](evidence/adaptation/execution.txt), and [complete job accounting](evidence/adaptation/jobs-completed.psv).
- [Retention protocol](evidence/retention/protocol.md), [execution snapshot](evidence/retention/execution.txt), [cost breakdown](evidence/retention/compute-cost.md), and [UTC accounting](evidence/retention/jobs.psv).
- [Dataset repair](evidence/audits/dataset-repair.txt), [CDS annotation audit](evidence/audits/cds-audit.txt), and [sampling audit](evidence/audits/sampling-audit.json). These are historical audits; their test-coordinate inspection does not imply new test predictions.
- Original detailed interpretations: [adaptation](evidence/adaptation/interpretation.txt) and [retention](evidence/retention/interpretation.txt). Plain-text snapshots preserve their original references to the full repository without treating those paths as portable package links.

## Reproduction boundaries

The repository is `/home/s34yan/projects/260308_virus-genome-extrapolation` locally. Nibi's study root is `/project/6002642/shouyu/virus-genome-extrapolation`; large states and feature caches remain there under `runs/2026-10-24` and `runs/2026-10-25-retention`. Original manifests retain exact remote paths and hashes. They describe full scientific inputs, not paths guaranteed to exist within this portable reporting subset.

Saved numerical summaries are in `tables/`. Full per-window probabilities, bootstrap weights and supporting evidence remain in the repository's `documents/retention-2026-10-25/results/`; adaptation raw readout evidence remains in `documents/adaptation-2026-10-24/results/`. Large weights, features, raw datasets and those prediction arrays are intentionally omitted from this review package. This package is an evidence index and source snapshot, not a self-contained rerun environment.

From the full repository with the original saved inputs and existing analysis environment, the completed adaptation interpretation is reproduced by `scripts/interpret_adaptation_duration.py`; retention local replay uses `scripts/verify_nt_retention_local.py`. Read their input paths and existing execution records before running them. Neither is needed merely to open this report. Scripts for original preparation/extraction/evaluation are included for provenance, and are not rerun by this assembly.

Selected source snapshots:

'''
for name in source_names:
    repro += f'- [{name}](sources/scripts/{name})\n'
repro += '\n[Statistical helpers](sources/src/baculo_adapt/retention.py) · [Focused statistical tests](sources/tests/test_retention.py) · [Project dependencies](sources/pyproject.toml)\n'
repro += '\n## Package integrity\n\n[Manifest](package-manifest.json) records every copied artifact, original relative path and SHA-256. [Assembly verification](assembly-verification.json) records local link, figure and key-result checks. Report rendering uses the existing markdown-it environment; no package installation is required.\n\n[Return to final report](report.md)\n'
(OUT/'reproducibility.md').write_text(repro,encoding='utf-8')
(OUT/'README.md').write_text('# Completed study report\n\n[Read final report](report.md) · [Browser-friendly report](report.html) · [Figures and tables](assets.md) · [Reproducibility](reproducibility.md)\n\nAssembled October 26, 2026 from verified completed results. No new scientific computation.\n',encoding='utf-8')

md = MarkdownIt('commonmark', {'html':False}).enable('table')
css = '''body{font:17px/1.65 Georgia,serif;color:#202a35;background:#f3f5f7;margin:0} main{max-width:1100px;margin:auto;background:white;padding:48px 5vw} h1,h2,h3{font-family:system-ui,sans-serif;line-height:1.23;color:#173b53} h1{font-size:34px} h2{margin-top:2em;border-top:1px solid #dce4e9;padding-top:22px} a{color:#116590} img{max-width:100%;height:auto} table{width:100%;border-collapse:collapse;font:14px/1.45 system-ui,sans-serif;margin:25px 0} th,td{padding:9px;border-bottom:1px solid #dce4e9;text-align:left} th{background:#edf3f7} code{overflow-wrap:anywhere;font-size:.85em} p,li{overflow-wrap:break-word} @media print{body{background:white}main{padding:0;max-width:none}h2,h3{break-after:avoid}img,table{break-inside:avoid}a{color:inherit}}'''
# Convert all navigable package Markdown to HTML companions; originals remain editable.
for p in OUT.rglob('*.md'):
    rendered = md.render(p.read_text(encoding='utf-8-sig'))
    rendered = re.sub(r'href="([^"#]+)\.md(#[^"]*)?"',lambda m:f'href="{m[1]}.html{m[2] or ""}"',rendered)
    p.with_suffix('.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+html.escape(p.stem)+' — NT adaptation study</title><style>'+css+'</style><main>'+rendered+'</main></html>',encoding='utf-8')
# Placeholders allow the link checker to resolve the integrity records themselves.
for name in ['package-manifest.json','assembly-verification.json']:
    (OUT/name).write_text('{}\n',encoding='utf-8')
broken=[]; count=0
for p in OUT.rglob('*.html'):
    for target in re.findall(r'(?:href|src)="([^"]+)"',p.read_text()):
        if '://' in target or target.startswith(('#','mailto:')): continue
        count+=1
        if not (p.parent/target.split('#')[0]).is_file(): broken.append([str(p.relative_to(OUT)),target])
assert not broken, broken
for rel, meta in copied.items():
    assert hashlib.sha256((OUT/rel).read_bytes()).hexdigest()==meta['sha256']
from PIL import Image
for p in (OUT/'figures').glob('*.png'):
    with Image.open(p) as im: im.verify()
a=json.loads((A/'analysis/verification.json').read_text())
r=json.loads((R/'local-verification.json').read_text())
assert a['scientific_runs']==14 and a['readouts']==59
assert r['tasks']==18 and r['readouts']==126
assert r['all_paired_intervals_and_simultaneous_task_bounds_reproduced']
rows=list(csv.DictReader((R/'results/task-retention.csv').open()))
failed=[x['task'] for x in rows if float(x['simultaneous_one_sided95_lower']) <= -.05]
assert set(failed)=={'H3K27ac','H3K4me3','H3K9ac','H3K9me3'}
# Check reported numeric table entries against saved machine-readable evidence.
import numpy as np
score_rows=list(csv.DictReader((A/'analysis/readout-scores.csv').open()))
def viral_score(prefix, metric):
    return np.mean([float(x['score']) for x in score_rows if x['metric']==metric and (x['readout']==prefix or (x['readout'].startswith(prefix) and x['readout'].endswith('budget4096')))])
for label,prefix in [('100M original','100m_before'),('100M general','100m-full-general-'),('100M baculovirus','100m-full-baculo-'),('500M original','500m_before'),('500M general','500m-full-general-'),('500M baculovirus','500m-full-baculo-')]:
    row=next(x for x in text.splitlines() if x.startswith('| '+label+' |'))
    cells=row.split('|')[2:5]
    for cell,metric in zip(cells,['log_loss','mcc','roc_auc']):
        printed=float(cell.strip().split()[0])
        assert abs(printed-viral_score(prefix,metric))<.000501,(label,metric)
summary=json.loads((R/'results/summary.json').read_text())
for family in ['Promoter','Enhancer','Splice','Histone']:
    row=next(x for x in text.splitlines() if x.startswith('| '+family+' |'))
    cells=row.split('|')
    scores=summary[family]['mcc']
    for cell,condition in zip(cells[3:6],['original','baculo','general']):
        actual=scores['original'] if condition=='original' else np.mean([v for k,v in scores.items() if condition in k])
        assert abs(float(cell)-actual)<.000051,(family,condition)
verification={'date':'2026-10-26','copied_artifacts':len(copied),'local_html_links_checked':count,'broken_links':broken,'copied_hashes_match':True,'all_pngs_decode':True,'adaptation_runs':14,'adaptation_readouts':59,'retention_tasks':18,'retention_readouts':126,'simultaneous_unresolved_tasks':failed,'main_score_tables_checked_against_saved_metrics':True,'figure_overview_visually_inspected':True,'html_browser_preview':'Unavailable: browser policy disallows local file URLs; static link checks completed.','new_training':False,'new_model_inference':False,'scope':'Report assembly and packaging, using previously independently verified scientific results.'}
(OUT/'assembly-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
manifest={'date':'2026-10-26','scientific_baseline_commit':'9b67df4','copied_artifacts':copied,'files':{}}
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name!='package-manifest.json':
        manifest['files'][str(p.relative_to(OUT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(verification,indent=2))
