#!/bin/bash
# On the Colab VM: three real merges (GPU) and the backend suite at head and base (CPU), in parallel.
R=/content/sbx/mk_results; mkdir -p $R
PY=$(ls -d /root/.unsloth/studio/unsloth_studio/bin/python 2>/dev/null || command -v python3)
echo "$PY" > $R/python.txt
cd /content/unsloth/studio/backend
cp /content/sbx/labonne_runs.py tests/manual/labonne_runs.py
( $PY tests/manual/labonne_runs.py neuralpipe_slerp neuralpipe_ties daredevil_dare_ties > $R/merges.log 2>&1; echo "exit $?" >> $R/merges.log; touch $R/merges.done ) &
(
  timeout 5400 $PY -m pytest -q -p no:cacheprovider tests --ignore=tests/manual > $R/suite_head.txt 2>&1
  echo "exit $?" >> $R/suite_head.txt
  git -C /content/unsloth worktree add -q /content/base 070e482c76ccc2bb1147fd0da471cdfa4ec7a47f
  cd /content/base/studio/backend
  timeout 5400 $PY -m pytest -q -p no:cacheprovider tests --ignore=tests/manual > $R/suite_base.txt 2>&1
  echo "exit $?" >> $R/suite_base.txt
  touch $R/suite.done
) &
wait
