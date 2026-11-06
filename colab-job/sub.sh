#!/bin/bash
R=/content/sbx/mk_results
cd /content/unsloth && git fetch -q origin night-shift-tester/mergekit-merge
[ -d /content/h29 ] || git worktree add -q /content/h29 FETCH_HEAD
[ -d /content/base ] || git worktree add -q /content/base 070e482c76ccc2bb1147fd0da471cdfa4ec7a47f
git -C /content/h29 log -1 --format=%H > $R/sub_head_sha.txt
cd /content/h29/studio/backend && python3 -m pytest -q -p no:cacheprovider tests -k 'export or merge' --ignore=tests/manual > $R/sub_head.txt 2>&1
cd /content/base/studio/backend && python3 -m pytest -q -p no:cacheprovider tests -k 'export or merge' > $R/sub_base.txt 2>&1
touch $R/sub.done
