#!/usr/bin/env bash
# From a source checkout (install.sh --local, editable update) REPO_ROOT is the clone, and
# .venv there is the developer's own uv venv. Only Unsloth-named leftovers may be removed.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
SETUP_SH="$ROOT/studio/setup.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

grep -E '^\[ -d "\$REPO_ROOT/\.venv[^"]*" \] && rm -rf ' "$SETUP_SH" > "$WORK/cleanup.sh"
grep -q 'REPO_ROOT/.venv_overlay' "$WORK/cleanup.sh" || {
    echo "FAIL: legacy venv cleanup lines not found"
    exit 1
}

REPO_ROOT="$WORK/checkout"
mkdir -p "$REPO_ROOT/studio" "$REPO_ROOT/.venv/bin" "$REPO_ROOT/.venv_overlay" "$REPO_ROOT/.venv_t5"
: > "$REPO_ROOT/.venv/bin/python"
# shellcheck disable=SC1090
. "$WORK/cleanup.sh" || true

[ -f "$REPO_ROOT/.venv/bin/python" ] || {
    echo "FAIL: setup.sh deleted the checkout's own .venv"
    exit 1
}
for d in .venv_overlay .venv_t5; do
    [ ! -e "$REPO_ROOT/$d" ] || {
        echo "FAIL: legacy $d was not cleaned up"
        exit 1
    }
done

echo "All checkout venv checks passed"
