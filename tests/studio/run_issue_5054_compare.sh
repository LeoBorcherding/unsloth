#!/usr/bin/env bash
set -euo pipefail

BASE_BRANCH="${1:-main}"
FIX_BRANCH="${2:-investigate/gguf-sudo-prompt-5054}"
FORK_REMOTE="${FORK_REMOTE:-leoborcherding}"
VENV_DIR=".venv-5054"
SUMMARY_FILE="tmp/issue_5054_compare_summary.txt"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Working tree is not clean. Commit or stash changes before running comparison."
    exit 1
fi

ORIGINAL_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
mkdir -p tmp

cleanup() {
    git checkout "$ORIGINAL_BRANCH" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if git remote get-url "$FORK_REMOTE" >/dev/null 2>&1; then
    git fetch "$FORK_REMOTE" "$BASE_BRANCH"
else
    echo "Warning: remote '$FORK_REMOTE' not found. Baseline pull step will be skipped." >&2
    FORK_REMOTE=""
fi

setup_env() {
    python -m venv "$VENV_DIR"
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
    python -m pip install --upgrade pip
    pip install -e .
    pip install -U unsloth_zoo
}

run_repro() {
    local label="$1"
    local branch="$2"
    local log_file="tmp/${label}_issue_5054.log"

    git checkout "$branch"

    if [[ "$label" == "main" ]] && [[ -n "$FORK_REMOTE" ]]; then
        git pull --ff-only "$FORK_REMOTE" "$BASE_BRANCH"
    fi

    setup_env

    set +e
    python tests/studio/repro_issue_5054.py 2>&1 | tee "$log_file"
    local exit_code=$?
    set -e

    deactivate || true

    {
        echo "===== ${label^^} ====="
        echo "branch: $branch"
        echo "sha: $(git rev-parse --short HEAD)"
        echo "exit_code: $exit_code"
        echo "sudo-related lines:"
        grep -inE "sudo|password|apt-get|permission denied|installing llama\\.cpp" "$log_file" || true
        echo
    } >> "$SUMMARY_FILE"
}

: > "$SUMMARY_FILE"

echo "Issue 5054 compare run" >> "$SUMMARY_FILE"
echo "date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> "$SUMMARY_FILE"
echo "repo: $REPO_ROOT" >> "$SUMMARY_FILE"
echo >> "$SUMMARY_FILE"

run_repro "main" "$BASE_BRANCH"
run_repro "fix" "$FIX_BRANCH"

echo "GGUF output files:" >> "$SUMMARY_FILE"
ls -lah tmp/issue_5054_gguf >> "$SUMMARY_FILE" 2>&1 || true

echo
echo "Done. Summary: $SUMMARY_FILE"
echo "Main log: tmp/main_issue_5054.log"
echo "Fix log:  tmp/fix_issue_5054.log"
