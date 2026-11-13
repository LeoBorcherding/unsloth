"""Thick charts for oobabooga/unsloth run 43276227073. Every number is a median read from
../art/npu-optab/results.json (Lemonade telemetry), never typed by hand."""
import collections, json, os, statistics, sys
sys.path.insert(0, os.path.expanduser('~/.claude/skills/thick-charts'))
import thick_charts as tc
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
rows = json.load(open('../art/npu-optab/results.json'))
G = collections.defaultdict(list)
for r in rows:
    if 'lemonade_tps' in r: G[(r['model'], r['test'], r['variant'])].append(r)
def med(model, test, variant, key):
    return statistics.median(r['lemonade_' + key] for r in G[(model, test, variant)])
def tps(m, t, v): return med(m, t, v, 'tps')
def ttft(m, t, v): return med(m, t, v, 'ttft')
def F(v): return f'{v:.2f}' if v < 10 else f'{v:.1f}'
for _f in ('thick_chart',):
    _orig = tc.thick_chart
    tc.thick_chart = lambda *a, **k: _orig(*a, fmt=F, **k)
def x(a, b, slower=False): return f'{a / b:.1f}x' + (' slower' if slower else '')

D2, MTP, MOE, Q9 = 'qwen3.8-dflash2-27b-FLM', 'qwen3.8-mtp-27b-FLM', 'qwen3.6-moe-35b-a3b-FLM', 'qwen3.5-9b-FLM'
NAME = {D2: 'Qwen3.8-27B-DFlash2-NPU2', MTP: 'Qwen3.8-27B-NPU2 (MTP)', MOE: 'Qwen3.6-35B-A3B-NPU2', Q9: 'Qwen3.5-9B-NPU2'}
META = ['AMD Strix Halo NPU (DevLab, Linux)', 'FastFlowLM 1.0.12', 'Lemonade 2026.44.1', 'temperature 0']
FOOT = ('oobabooga/unsloth run 43276227073, 2026-11-12, Unsloth Studio fork phase 5 92c2fbb. Lemonade telemetry, '
        'median of 2-3 identical reps (spread under 2%), variants back to back on one runner.')

# 1. #17530: tools sent (what Unsloth Studio did before) vs none. Same prompt otherwise.
def tools_rows(m, key):
    on, off = med(m, 'tools', 'tools on', key), med(m, 'tools', 'tools off', key)
    if key == 'tps':
        return [('tools sent (before)', on, 'base'), ('no tools (#17530)', off, 'new', x(off, on))]
    return [('tools sent (before)', on, 'base'), ('no tools (#17530)', off, 'new', x(on, off) + ' sooner')]
tc.thick_chart(panels=[
  {'title': 'Reply speed', 'unit': 'tokens / s', 'groups': [(NAME[m], tools_rows(m, 'tps')) for m in (D2, MOE, Q9)]},
  {'title': 'Time to first token', 'unit': 'seconds (lower is better)', 'groups': [(NAME[m], tools_rows(m, 'ttft')) for m in (D2, MOE, Q9)]}],
  title='Tools off by default on the NPU (#17530)',
  subtitle=f'Without a tool list Qwen3.8 27B replies {x(tps(D2,"tools","tools off"), tps(D2,"tools","tools on"))} faster, '
           f'and Qwen3.6 35B starts {x(ttft(MOE,"tools","tools on"), ttft(MOE,"tools","tools off"))} sooner.',
  meta=META, footnote=FOOT + ' 256-token replies. "Tools sent" is two tool definitions (web_search, python) in the '
       'request, as Unsloth Studio sent with Search and Code on; they add 311 prompt tokens. The 27B MTP build was not run with tools.',
  out='01_tools_17530', ncols=2)

# 2. #17532: does a title request between turns cost turn 2 its prefix cache?
def title_rows(m):
    t1 = ttft(m, 'title', 'no title request: turn 1')
    a = ttft(m, 'title', 'title request between turns: turn 2'); b = ttft(m, 'title', 'no title request: turn 2')
    return [('turn 1, full prompt read', t1, 'base'), ('turn 2, title request between', a, 'rival'),
            ('turn 2, no title request (#17532)', b, 'new')]
tc.thick_chart(panels=[
  {'title': 'Time to first token', 'unit': 'seconds (lower is better)', 'groups': [(NAME[m], title_rows(m)) for m in (MOE, Q9)]}],
  title='Skipping the title request between turns (#17532)',
  subtitle="No measurable change: FastFlowLM 1.0.12 keeps the chat's prompt cache through the title request.",
  meta=META, footnote=FOOT + ' Turn 1 is a ~2,360-token prompt; turn 2 resends it plus a follow-up, so a cache hit skips '
       'most of the read. Both turn 2 variants hit it.', out='02_title_17532', ncols=1)

# 3. #17510 power mode: --pmode reached flm serve (checked in the Lemonade log).
def pm_rows(m, test, key):
    d = med(m, test, 'power default', key)
    return [('default', d, 'base'), ('turbo', med(m, test, 'power turbo', key), 0), ('power saver', med(m, test, 'power powersaver', key), 1)]
tc.thick_chart(panels=[
  {'title': 'Reply speed', 'unit': 'tokens / s', 'groups': [(NAME[m], pm_rows(m, 'settings decode', 'tps')) for m in (MOE, Q9)]},
  {'title': 'Long prompt, first token', 'unit': 'seconds (lower is better)', 'groups': [(NAME[m], pm_rows(m, 'settings prefill', 'ttft')) for m in (MOE, Q9)]}],
  title='NPU power mode (#17510)',
  subtitle='Turbo and power saver change nothing measurable on Strix Halo (all within 2%).',
  meta=META, footnote=FOOT + ' Long prompt = 1,764 tokens. Each mode is a reload with flm serve --pmode turbo / powersaver (seen in the Lemonade log).',
  out='03_power_17510', ncols=2)

# 4. #17510 prefill chunk.
def pc_rows(m):
    d = ttft(m, 'settings prefill', 'power default')
    c512 = ttft(m, 'settings prefill', 'prefill chunk 512'); c2k = ttft(m, 'settings prefill', 'prefill chunk 2048')
    return [('chunk 512', c512, 'rival', x(c512, d, True)), ('default', d, 'base'), ('chunk 2048', c2k, 'new')]
tc.thick_chart(panels=[
  {'title': 'Long prompt, first token', 'unit': 'seconds (lower is better)', 'groups': [(NAME[m], pc_rows(m)) for m in (MOE, Q9)]}],
  title='Prefill chunk length (#17510)',
  subtitle=f'The default is already the fast setting; a 512 chunk makes Qwen3.6 35B wait '
           f'{x(ttft(MOE,"settings prefill","prefill chunk 512"), ttft(MOE,"settings prefill","power default"))} longer.',
  meta=META, footnote=FOOT + ' Long prompt = 1,764 tokens. Each setting is a reload with flm serve --prefill-chunk-len. Reply speed does not change '
       'with chunk length.', out='04_prefill_chunk_17510', ncols=1)

# 5. Per model, plain chat at the stack's defaults.
def b(m): return tps(m, 'baseline', 'default')
tc.thick_chart(panels=[
  {'title': 'Reply speed', 'unit': 'tokens / s', 'groups': [
    ('Small', [('LFM3-1.7B-NPU2', b('lfm3-1.7b-FLM'), 'new'), ('Qwen3-VL-4B-NPU2', b('qwen3vl-it-4b-FLM'), 0), ('Gemma4-E4B-IT-NPU2', b('gemma4-it-e4b-FLM'), 1)]),
    ('Mid', [('Qwen3.5-9B-NPU2', b(Q9), 0), ('Gemma4-12B-IT-NPU2', b('gemma4-it-12b-FLM'), 1)]),
    ('Large', [('Qwen3.6-35B-A3B-NPU2 (MoE)', b(MOE), 'new'), (NAME[D2], b(D2), 'base'), (NAME[MTP], b(MTP), 'base')])]}],
  title='Reply speed on the AMD NPU, by model',
  subtitle=f'The 35B MoE runs {x(b(MOE), b(D2))} faster than the dense 27B: it reads 3B weights per token, not 27B.',
  meta=META, footnote=FOOT + ' Plain chat, no tools, 256-token replies (Qwen3-VL stopped at 234).', out='05_models_toks', ncols=1)

# 6. The 27B: what speculation buys. "Off" = tools sent, which turns speculation off in FLM 1.0.12.
off = tps(D2, 'tools', 'tools on')
spec = [('speculation off (tools sent)', off, 'base'),
        ('MTP, 256-token reply', b(MTP), 0, x(b(MTP), off)), ('MTP, 512-token reply', tps(MTP, '27b speculation', '512-token answer'), 0, x(tps(MTP, '27b speculation', '512-token answer'), off)),
        ('DFlash2, 256-token reply', b(D2), 'new', x(b(D2), off)), ('DFlash2, 512-token reply', tps(D2, '27b speculation', '512-token answer'), 'new', x(tps(D2, '27b speculation', '512-token answer'), off))]
tc.thick_chart(panels=[{'title': 'Reply speed', 'unit': 'tokens / s', 'groups': [('Qwen3.8 27B', spec)]}],
  title='Why Qwen3.8 27B runs at 2-3 tokens per second',
  subtitle='Speculation is working and buys 7-10x; 2-3 tok/s is the ceiling for a dense 27B on FastFlowLM 1.0.12.',
  meta=META, footnote=FOOT + ' "Speculation off" is the DFlash2 build with tools sent (flm turns speculation off under a '
       'tool grammar), 0.28 tok/s, matching the 3.45 s/token FastFlowLM reports for plain decode in ROCm/FastFlowLM#1029.',
  out='06_27b_speculation', ncols=1)

# 7. The whole stack, both directly measured: before = tools sent at default settings; after = the phase 5 defaults.
def stack(m, key):
    before = med(m, 'tools', 'tools on', key); after = med(m, 'baseline', 'default', key)
    note = x(after, before) if key == 'tps' else x(before, after) + ' sooner'
    return [('before the stack (tools sent)', before, 'base'), ('phase 5 stack, defaults', after, 'new', note)]
tc.thick_chart(panels=[
  {'title': 'Reply speed', 'unit': 'tokens / s', 'groups': [(NAME[m], stack(m, 'tps')) for m in (D2, MOE, Q9)]},
  {'title': 'Time to first token', 'unit': 'seconds (lower is better)', 'groups': [(NAME[m], stack(m, 'ttft')) for m in (D2, MOE, Q9)]}],
  title='All NPU speed PRs together (#17530, #17532, #17510)',
  subtitle='The whole gain comes from #17530; #17532 and the #17510 defaults add nothing on top on FastFlowLM 1.0.12.',
  meta=META, footnote=FOOT + ' Both bars measured directly, not added up: "before" sends the tool list at default settings; '
       '"stack" is plain chat with no tools, no title request and default power and prefill chunk.',
  out='07_stack_all', ncols=2)
print('ok')
