# Issue 5054 Repro and Test Plan

Issue: https://github.com/unslothai/unsloth/issues/5054

## Goal

Verify that GGUF export:
1. Reproduces the sudo password prompt on baseline main.
2. Does not prompt for sudo on the fix branch.
3. Still successfully exports GGUF artifacts.

## Branches and Commits

- Baseline branch: `main` (synced from `leoborcherding/main`)
- Investigation branch: `investigate/gguf-sudo-prompt-5054`
- Proposed fix branch: use `investigate/gguf-sudo-prompt-5054` after fix commits

Record exact SHAs before each run:

```bash
git rev-parse --short HEAD
git status -sb
```

## Environment Requirements

- Linux/HPC environment where sudo is either unavailable or requires password prompt.
- Python 3.10+.
- CUDA optional for this specific repro (GGUF conversion path itself does not require GPU for the subprocess investigation).

Recommended clean test env per branch run:

```bash
python -m venv .venv-5054
source .venv-5054/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install -U unsloth_zoo
python -c "import unsloth, unsloth_zoo; print('unsloth ok')"
```

## Repro Script

Use the committed repro script:

- `tests/studio/repro_issue_5054.py`

If you need to recreate it manually, this is the content:

```python
import os
from unsloth import FastLanguageModel

os.environ['UNSLOTH_ENABLE_LOGGING'] = '1'

# Keep model small to reduce setup time
MODEL_NAME = 'unsloth/gemma-3-1b-it-bnb-4bit'
OUT_DIR = 'tmp/issue_5054_gguf'

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# This is the call reported in issue #5054
model.save_pretrained_gguf(
    OUT_DIR,
    tokenizer,
    quantization_method='q8_0',
)

print('DONE')
```

## One-Command Compare Run (Recommended)

Use the committed runner script:

```bash
chmod +x tests/studio/run_issue_5054_compare.sh
tests/studio/run_issue_5054_compare.sh
```

Optional custom branches:

```bash
tests/studio/run_issue_5054_compare.sh main investigate/gguf-sudo-prompt-5054
```

This generates:

- `tmp/main_issue_5054.log`
- `tmp/fix_issue_5054.log`
- `tmp/issue_5054_compare_summary.txt`

## Baseline Run (Main, Without Fix)

```bash
git checkout main
git pull --ff-only leoborcherding main
python tests/studio/repro_issue_5054.py 2>&1 | tee tmp/main_issue_5054.log
```

Collect evidence:

```bash
grep -inE "sudo|password|apt-get|permission denied|installing llama\.cpp" tmp/main_issue_5054.log || true
ls -lah tmp/issue_5054_gguf || true
```

Expected baseline result (current bug):
- A sudo/password prompt appears during GGUF export setup, OR command output indicates attempted privileged install path.

## Fix-Branch Run (Compare)

```bash
git checkout investigate/gguf-sudo-prompt-5054
# Apply fix commits here, then:
python tests/studio/repro_issue_5054.py 2>&1 | tee tmp/fix_issue_5054.log
```

Collect evidence:

```bash
grep -inE "sudo|password|apt-get|permission denied|installing llama\.cpp" tmp/fix_issue_5054.log || true
ls -lah tmp/issue_5054_gguf || true
```

Expected fix result:
- No sudo/password prompt.
- GGUF output is generated successfully.

## Comparison Checklist (Main vs Fix)

- [ ] Same machine and user account used for both runs.
- [ ] Same Python version and dependency install method.
- [ ] Same model, quantization method, and output path pattern.
- [ ] `main` log captured (`tmp/main_issue_5054.log`).
- [ ] `fix` log captured (`tmp/fix_issue_5054.log`).
- [ ] No sudo/password prompt in fix log.
- [ ] GGUF files exist in both runs (or baseline fails only due to sudo path).
- [ ] Export duration/regression noted (optional).

## Optional Deep Subprocess Investigation

If prompt source is unclear, run with syscall/process tracing:

```bash
strace -ff -o tmp/main_strace.log python tests/studio/repro_issue_5054.py
```

Then inspect for privilege escalation attempts:

```bash
grep -RinE "execve\(.*sudo|apt-get|dnf|yum" tmp/main_strace.log*
```

## PR Evidence to Attach

- Main SHA and fix SHA.
- `tmp/main_issue_5054.log` excerpt showing sudo prompt.
- `tmp/fix_issue_5054.log` excerpt showing no sudo prompt and successful export.
- Short root-cause summary of where subprocess install path was triggered.
- Any dependency pin/version detail if fix depends on updated `unsloth_zoo` behavior.
