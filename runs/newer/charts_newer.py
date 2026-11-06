import json, sys
sys.path.insert(0, __import__("os").path.expanduser("~/.claude/skills/thick-charts"))
import thick_charts as tc

LABEL = {'qwen3_4b_slerp': 'Qwen3-4B SLERP (2 parents)', 'qwen3_4b_ties': 'Qwen3-4B TIES (base + 2)',
         'qwen3_4b_dare_ties': 'Qwen3-4B DARE-TIES (base + 3)', 'smollm3_3b_ties': 'SmolLM3-3B TIES (base + 1)'}
runs = [json.load(open(f'{n}.json')) for n in LABEL]

groups = []
for r in runs:
    merged, *parents = r['rows']
    rows = [(p['name'].split('/')[-1], round(p['ppl'], 2), 'base') for p in parents]
    lo, hi = min(p['ppl'] for p in parents), max(p['ppl'] for p in parents)
    rows.append(('merged', round(merged['ppl'], 2), 'new', f'parents {lo:.2f}-{hi:.2f}'))
    groups.append((LABEL[r['name']], rows))

tc.thick_chart(
    panels = [{'title': 'Perplexity on a fixed paragraph', 'unit': 'lower is better', 'groups': groups}],
    title = 'Newer models: Qwen3-4B (2507) and SmolLM3-3B',
    subtitle = 'Three of four land inside their parents\' range; Qwen3 TIES sits 0.16 above its worst parent.',
    meta = ['Colab L4 (24 GB), Linux', 'MergeKit 0.1.5 in its own venv', 'bf16 eval, 30-token greedy completion'],
    footnote = '2026-11-06, head 528a054. Same text and prompt as the 7B runs. Every merge has finite logits and a '
               'coherent completion. Qwen3 TIES and DARE-TIES use Qwen3-4B-Base as the base; DARE-TIES adds the '
               'hybrid Qwen3-4B as a third donor, the worst parent here (7.12).',
    out = '03_newer_perplexity', ncols = 1, fmt = lambda v: f'{v:.2f}',
)

short = {k: v.split(' (')[0] for k, v in LABEL.items()}
cost = []
for key, title, unit, f in [('merge_seconds', 'Wall time', 's', lambda v: v),
                            ('peak_host_mb', 'Peak host RAM', 'GB', lambda v: round(v / 1024, 1)),
                            ('peak_gpu_mb', 'Peak GPU memory', 'GB', lambda v: round(v / 1024, 1))]:
    cost.append({'title': title, 'unit': unit,
                 'groups': [('3-4B merges', [(short[r['name']], f(r[key]), 'new') for i, r in enumerate(runs)])]})
tc.thick_chart(
    panels = cost,
    title = 'What a 3-4B merge costs on a Colab L4',
    subtitle = 'About two minutes each with downloads; host RAM stays near 6 GB.',
    meta = ['Colab L4 (24 GB), 51 GB RAM, Linux', '--lazy-unpickle --cuda --out-shard-size 5B'],
    footnote = '2026-11-06, head 528a054. Machine-wide, sampled once a second minus the idle reading. Wall time '
               'includes the first download of each parent. Outputs: 8.06 GB (Qwen3-4B), 6.17 GB (SmolLM3-3B).',
    out = '04_newer_cost', ncols = 3,
)
