#!/usr/bin/env python3
# reset_trigger.py — decides whether the nanopath-overfitX RAW training track should be
# RESET/CONSOLIDATED into a fresh evo workspace. Small, dependency-light (stdlib only).
#
# It reads overfitX/MANIFEST.json and (best-effort) the train workspace's `evo` state, then
# prints exactly two lines for the orchestrator to parse:
#     TRIGGER=yes|no
#     REASON=count|stall|none
#
# Trigger logic (OR):
#   count : tracks.train.committed_since_reset >= trigger.count           (default 50)
#   stall : the best RAW score has not improved by >= trigger.stall_eps    (default 0.006)
#           over the last trigger.stall_window commits                     (default 8)
#
# The manifest carries a precomputed stall counter (tracks.train.stall) as the simple, always-
# available implementation. If `evo` is on PATH and the train workspace is known, we ALSO
# recompute the stall from `evo frontier`/`evo status` and trigger if EITHER says stalled —
# the live recompute catches drift if the manifest's counter is stale.
#
# Usage:
#   python3 reset_trigger.py [--manifest overfitX/MANIFEST.json] [--evo-workspace <dir>]

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def load_manifest(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"reset_trigger: manifest not found at {path}")
    try:
        return json.loads(path.read_text())
    except Exception as e:
        sys.exit(f"reset_trigger: could not parse {path}: {e}")


def _get(d: dict, dotted: str, default=None):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def evo_recompute_stall(workspace: str, window: int, eps: float):
    """Best-effort: ask `evo` for the score history and recompute whether the best raw score
    improved by >= eps over the last `window` commits. Returns True/False, or None if `evo`
    is unavailable / output can't be parsed (caller then relies on the manifest counter)."""
    if not workspace:
        return None
    evo = None
    for cand in ("evo",):
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=False, timeout=10)
            evo = cand
            break
        except Exception:
            continue
    if evo is None:
        return None

    # Try a few read-only subcommands that expose committed scores; tolerate any failing.
    scores = []
    for sub in (["frontier", "--json"], ["status", "--json"], ["report", "--json"]):
        try:
            p = subprocess.run([evo, *sub], cwd=workspace, capture_output=True,
                               text=True, timeout=30)
        except Exception:
            continue
        if p.returncode != 0 or not p.stdout.strip():
            continue
        try:
            data = json.loads(p.stdout)
        except Exception:
            continue
        # Walk the JSON for committed experiment scores in order.
        found = _extract_scores(data)
        if len(found) >= 2:
            scores = found
            break
    if len(scores) < window + 1:
        return None  # not enough history to judge a stall; defer to manifest

    # Best-so-far at the start of the window vs. best-so-far at the end.
    head = scores[:-window]
    best_before = max(head) if head else float("-inf")
    best_now = max(scores)
    return (best_now - best_before) < eps


def _extract_scores(data):
    """Pull an ordered list of committed scores out of an `evo` JSON blob, defensively."""
    out = []
    if isinstance(data, dict):
        for key in ("experiments", "nodes", "history", "commits"):
            seq = data.get(key)
            if isinstance(seq, list):
                for item in seq:
                    if isinstance(item, dict) and isinstance(item.get("score"), (int, float)):
                        out.append(float(item["score"]))
                if out:
                    return out
        if isinstance(data.get("score"), (int, float)):
            out.append(float(data["score"]))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("score"), (int, float)):
                out.append(float(item["score"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(SCRIPT_DIR / "overfitX" / "MANIFEST.json"))
    ap.add_argument("--evo-workspace", default=os.environ.get("OVERFITX_TRAIN_WORKSPACE", ""))
    args = ap.parse_args()

    man = load_manifest(Path(args.manifest))

    committed = int(_get(man, "tracks.train.committed_since_reset", 0))
    stall_counter = int(_get(man, "tracks.train.stall", 0))
    count_thresh = int(_get(man, "trigger.count", 50))
    stall_window = int(_get(man, "trigger.stall_window", 8))
    stall_eps = float(_get(man, "trigger.stall_eps", 0.006))

    reason = "none"
    trigger = False

    # 1) Hard commit budget.
    if committed >= count_thresh:
        trigger, reason = True, "count"

    # 2) Stall — manifest's precomputed counter (always available).
    manifest_stalled = stall_counter >= stall_window

    # 2b) Stall — live recompute from `evo`, if available (catches a stale manifest counter).
    live_stalled = evo_recompute_stall(args.evo_workspace, stall_window, stall_eps)

    if not trigger and (manifest_stalled or live_stalled is True):
        trigger, reason = True, "stall"

    print(f"TRIGGER={'yes' if trigger else 'no'}")
    print(f"REASON={reason}")


if __name__ == "__main__":
    main()
