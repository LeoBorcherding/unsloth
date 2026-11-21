#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Tests the AMD ROCm 10 (whl-next) pin in install.sh: exact torch set and the device extra. Lifted, so drift fails.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
. "$SCRIPT_DIR/_harness.sh"

lift_fn() {
    awk -v fn="$1" '$0 ~ "^"fn"\\(\\) \\{" {p=1} p {print} p && /^\}/ {exit}' "$INSTALL_SH"
}
# The constraint case that owns the whl-next arm, from its `case` line to its `esac`.
lift_constraint_case() {
    awk '/^case "\$_torch_index_leaf" in$/ {buf=$0; p=1; next}
         p {buf=buf "\n" $0} p && /^esac$/ {if (buf ~ /whl-next\)/) {print buf; exit} p=0}' "$INSTALL_SH"
}

FN_EXTRA=$(lift_fn _amd_whl_next_device_extra)
CONSTS=$(grep -E '^_AMD_WHL_NEXT_[A-Z]+=' "$INSTALL_SH" || true)
CASE_BLOCK=$(lift_constraint_case)
[ -n "$FN_EXTRA" ] || { echo "FAIL: _amd_whl_next_device_extra not found in install.sh"; exit 1; }
[ -n "$CASE_BLOCK" ] || { echo "FAIL: no whl-next arm in install.sh's constraint case"; exit 1; }

echo "=== _amd_whl_next_device_extra ==="
run_extra() { bash -c "$FN_EXTRA"'
    _amd_whl_next_device_extra "$1"' _ "$1"; }
assert_eq "RX 9060 XT"                    "device-gfx1200"               "$(run_extra gfx1200)"
assert_eq "rocminfo names each agent twice" "device-gfx1201"             "$(run_extra "$(printf 'gfx1201\ngfx1201\n')")"
assert_eq "ISA suffix stripped"           "device-gfx90a"                "$(run_extra 'gfx90a:sramecc+:xnack-')"
assert_eq "upper case"                    "device-gfx1151"               "$(run_extra GFX1151)"
assert_eq "mixed host keeps both"         "device-gfx1100,device-gfx1201" "$(run_extra "$(printf 'gfx1100\ngfx1201\n')")"
# gfx906 has no whl-next build: an unknown arch must not drop to a partial extra.
assert_eq "unknown arch -> device-all"    "device-all"                   "$(run_extra "$(printf 'gfx1201\ngfx906\n')")"
assert_eq "nothing read -> device-all"    "device-all"                   "$(run_extra '')"

echo ""
echo "=== constraint arm ==="
run_case() {
    # $1 leaf, $2 preset extra, $3 probe output, $4 KFD output; prints constraints and extra
    bash -c "$CONSTS
        _probe_amd_gfx_arch() { printf '%s' \"\$P\"; }
        _kfd_gfx_targets() { printf '%s' \"\$K\"; }
        $FN_EXTRA
        TORCH_CONSTRAINT=default; TORCHVISION_CONSTRAINT=default; TORCHAUDIO_CONSTRAINT=default
        _torch_index_leaf=\"\$1\"; _TORCH_EXTRA=\"\$2\"
        $CASE_BLOCK
        echo \"\$TORCH_CONSTRAINT|\$TORCHVISION_CONSTRAINT|\$TORCHAUDIO_CONSTRAINT|\$_TORCH_EXTRA\"" _ "$1" "$2" 2>/dev/null
}
SET='torch==2.14.0+rocm10.1.0|torchvision==0.29.0a0+rocm10.1.0|torchaudio==2.11.0.3+rocm10.1.0'
assert_eq "whl-next pins the 10.1 set, extra from rocminfo" "$SET|device-gfx1200" "$(P=gfx1200 K= run_case whl-next '')"
assert_eq "no rocminfo: extra from KFD"                     "$SET|device-gfx1201" "$(P= K=gfx1201 run_case whl-next '')"
assert_eq "a set UNSLOTH_TORCH_EXTRA wins"                  "$SET|device-gfx1100" "$(P=gfx1200 K= run_case whl-next device-gfx1100)"
assert_eq "nothing readable -> device-all"                  "$SET|device-all"     "$(P= K= run_case whl-next '')"
assert_eq "rocm7.2 leaf unchanged" \
    'torch>=2.11.0,<2.12.0|torchvision>=0.26.0,<0.27.0|torchaudio>=2.11.0,<2.12.0|' "$(P=gfx1200 K= run_case rocm7.2 '')"
assert_eq "cpu leaf untouched"     'default|default|default|'                        "$(P=gfx1200 K= run_case cpu '')"

summary
