import json, sys
sys.path.insert(0, __import__("os").path.expanduser("~/.claude/skills/thick-charts"))
import thick_charts as tc

LABEL = {'qwen3_2x4b_moe': 'Qwen3 2x4B MoE (base Instruct-2507, 2 experts)',
         'beyonder_4x7b_moe': 'Beyonder-4x7B-v2 recipe (base Marcoro14, 4 experts)'}
groups = []
for n, label in LABEL.items():
    r = json.load(open(f'{n}.json'))
    merged, *parents = r['rows']
    rows = [(p['name'].split('/')[-1], round(p['ppl'], 2), 'base') for p in parents]
    lo, hi = min(p['ppl'] for p in parents), max(p['ppl'] for p in parents)
    rows.append(('merged MoE', round(merged['ppl'], 2), 'new', f'parents {lo:.2f}-{hi:.2f}'))
    groups.append((label, rows))

tc.thick_chart(
    panels = [{'title': 'Perplexity on a fixed paragraph', 'unit': 'lower is better', 'groups': groups}],
    title = 'MoE merges through mergekit-moe',
    subtitle = 'Both land inside their parents\' range, with hidden-state gates built from each expert\'s prompts.',
    meta = ['Colab L4 (24 GB), Linux', 'MergeKit 0.1.5, gate_mode hidden, 2 experts per token', 'bf16 eval'],
    footnote = '2026-11-06, MergeKit code at 362058f; the Beyonder run also drops --lazy-unpickle, the change pushed as 118e14f. '
               'Qwen3 builds the qwen3 architecture, Beyonder mixtral. The Beyonder 24B output (48.3 GB) was evaluated in bf16 '
               'with CPU offload because an nf4 load ran out of memory on the L4. Base listed first in each group.',
    out = '05_moe_perplexity', ncols = 1, fmt = lambda v: f'{v:.2f}',
)
