#!/usr/bin/env bash
# evo gate: locked-path integrity. Exits non-zero if probe.py / benchmarking/ /
# labless/submit_to_labless.py differ from the base 'main' branch — i.e. an
# experiment tried to game the score by editing the evaluation itself. This is
# the anti-Goodhart guard; it mirrors the Labless locked-path rule.
set -u
WT="${1:?usage: evo_gate_locked.sh <worktree>}"
LOCKED=(probe.py benchmarking labless/submit_to_labless.py)
git -C "$WT" diff --quiet main -- "${LOCKED[@]}" || { echo "LOCKED=dirty(committed-vs-main)"; exit 1; }
git -C "$WT" diff --quiet --       "${LOCKED[@]}" || { echo "LOCKED=dirty(uncommitted)"; exit 1; }
echo "LOCKED=clean"
