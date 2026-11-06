import json, sys
sys.path.insert(0, __import__("os").path.expanduser("~/.claude/skills/thick-charts"))
import thick_charts as tc

LABEL = {'neuralpipe_slerp': 'NeuralPipe SLERP (2 parents)', 'neuralpipe_ties': 'NeuralPipe TIES (base + 2)',
         'daredevil_dare_ties': 'Daredevil DARE-TIES (base + 3)'}
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
    title = "Labonne's MergeKit merges, rebuilt through the new merge route",
    subtitle = 'Every merge lands inside its parents\' perplexity range, with finite logits and a sane completion.',
    meta = ['Colab L4 (24 GB), Linux', 'MergeKit 0.1.5 in its own venv', 'bf16 eval, 30-token greedy completion'],
    footnote = '2026-11-06, head d0a206f7c6. Same text and prompt for every model. Parent names are the Hub repos; '
               'Marcoro14 swapped for NeuralPipe SLERP (its Marcoroni-7B-v3 parent is gone from the Hub).',
    out = '01_perplexity', ncols = 1, fmt = lambda v: f'{v:.2f}',
)

cost = []
for key, title, unit, f in [('merge_seconds', 'Wall time', 's', lambda v: v),
                            ('peak_host_mb', 'Peak host RAM', 'GB', lambda v: round(v / 1024, 1)),
                            ('peak_gpu_mb', 'Peak GPU memory', 'GB', lambda v: round(v / 1024, 1))]:
    cost.append({'title': title, 'unit': unit,
                 'groups': [('7B merges', [(LABEL[r['name']].split(' (')[0], f(r[key]), i) for i, r in enumerate(runs)])]})
tc.thick_chart(
    panels = cost,
    title = 'What a 7B merge costs on a Colab L4',
    subtitle = 'Lazy loading keeps host RAM and GPU memory under 7 GB, below the size of one 14.5 GB model.',
    meta = ['Colab L4 (24 GB), 51 GB RAM, Linux', '--lazy-unpickle --cuda --out-shard-size 5B'],
    footnote = '2026-11-06, head d0a206f7c6. Machine-wide, sampled once a second from /proc/meminfo and nvidia-smi '
               'minus the idle reading, with the CPU test suite running alongside, so RAM is an upper bound. '
               'Each output is 14.49 GB. Wall time includes the first download of each parent.',
    out = '02_cost', ncols = 3,
)
