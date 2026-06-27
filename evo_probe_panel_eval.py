#!/usr/bin/env python3
# evo_probe_panel_eval.py — PRE-PROBE panel benchmark for the nanopath-overfitX system.
#
# WHAT THIS IS
# ------------
# The PRE-PROBE evo track does NOT train. It evaluates a *fixed* panel of already-trained
# checkpoints (2 random survivors + 1 consolidated baseline, see overfitX/panel.example.json)
# under the locked downstream probe suite, and reports the panel MEAN of `final_probe_score`
# as the single evo score. Because the panel is held fixed for the whole cycle and probe/eval
# seeds never change, a higher score can only come from a genuinely better probe/eval *recipe*
# (the thing an evo subagent edits) — not from cherry-picking an easier checkpoint. That is the
# anti-overfit property the two-track loop relies on.
#
# HOW IT SCORES A CHECKPOINT WITHOUT TRAINING
# -------------------------------------------
# train.py supports `train.resume=<ckpt>`. On resume it restores the checkpoint's
# examples_seen / train_flops (train.py:314-317). The training loop guard
#     while examples_seen + batch_size <= max_train_samples and train_flops < max_train_flops:
# (train.py:628) therefore runs ZERO iterations when the checkpoint's budget is already met,
# after which train.py runs the final probe and writes summary.json (train.py:833 / 868).
# So a "probe-only" run == resume from a budget-complete checkpoint. We set train.resume on a
# COPY of configs/main.yaml inside the worktree, point output_dir at a per-checkpoint scratch
# dir, invoke train.py, and read that run's summary.json.
#
# BELOW-BUDGET GUARD (anti misconfiguration)
# ------------------------------------------
# If a panel path points at a checkpoint whose examples_seen/train_flops are BELOW the config
# budget, resuming it would actually TRAIN (the loop guard would let iterations through). That
# would (a) burn GPU and (b) silently change the panel. We refuse: every checkpoint is opened
# and its counters checked against the config budget BEFORE any run; a below-budget checkpoint
# is a hard ERROR (we do not score it). This guards against wrong panel paths.
#
# OUTPUT CONTRACT
# ---------------
# Mirrors evo_eval.py EXACTLY so evo scores this identically:
#   - PROBES map imported verbatim from evo_eval.py (8 canonical probes).
#   - One trace per probe written to $EVO_TRACES_DIR (task_<name>.json), where the per-probe
#     score is the MEAN of that probe across the panel checkpoints (not a single run).
#   - The experiment `score` is the panel mean of `final_probe_score`, published atomically to
#     $EVO_RESULT_PATH (claim-then-rename), else printed.
#
# USAGE (registered as the preprobe workspace's --benchmark command):
#   python3 {worktree}/evo_probe_panel_eval.py {worktree}/run_summary.json
# The positional arg is accepted for symmetry with evo_eval.py but is NOT a real summary here
# (this benchmark generates its own per-checkpoint summaries). The panel itself is read from
# overfitX/panel.json (next to this script's repo root), overridable via $OVERFITX_PANEL.

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Reuse evo_eval.py's PROBES dict verbatim. We import it if importable, else
# fall back to an inline copy that MUST stay byte-identical to evo_eval.py.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROBES = None
try:
    # evo_eval.py executes top-level code on import (it reads sys.argv[1]); guard against that
    # by only importing the symbol via a restricted exec of its PROBES literal if direct import
    # is unsafe. Simplest robust path: locate an evo_eval.py and exec just its PROBES assignment.
    import importlib.util

    _candidates = [
        SCRIPT_DIR / "evo_eval.py",
        *sorted(SCRIPT_DIR.glob(".evo/run_*/worktrees/exp_*/evo_eval.py")),
    ]
    _src = None
    for _c in _candidates:
        if _c.exists():
            _src = _c.read_text()
            break
    if _src is not None:
        m = re.search(r"PROBES\s*=\s*\{.*?\n\}", _src, re.S)
        if m:
            _ns = {}
            exec(compile(m.group(0), "<evo_eval.PROBES>", "exec"), _ns)
            PROBES = _ns.get("PROBES")
except Exception:
    PROBES = None

if PROBES is None:
    # Verbatim copy of evo_eval.py's PROBES (keep in sync if evo_eval.py changes).
    PROBES = {
        "linear":       ("final_probe_linear_mean_f1",       "_linear_val"),
        "knn":          ("final_probe_knn_mean_f1",          "_knn_val"),
        "16shot":       ("final_probe_fewshot_mean_f1",      "_fewshot_val"),
        "segmentation": ("final_probe_seg_mean_jaccard",     "_seg_val"),
        "progression":  ("final_probe_slide_mean_auc",       "ucla_lung"),
        "mutation":     ("final_probe_auc_mean",             "surgen"),
        "survival":     ("final_probe_survival_mean_cindex", "_cindex"),
        "robustness":   ("final_probe_robustness_mean",      "pathorob"),
    }


def _die(msg: str) -> "None":
    sys.exit(f"[evo_probe_panel_eval] ERROR: {msg}")


# ---------------------------------------------------------------------------
# Locate the worktree (this script's dir), the repo root, and the panel file.
# ---------------------------------------------------------------------------
WORKTREE = SCRIPT_DIR
# The worktree carries its own copy of configs/main.yaml; if missing, fall back to repo root.
CONFIG_IN_WORKTREE = WORKTREE / "configs" / "main.yaml"
if not CONFIG_IN_WORKTREE.exists():
    _die(f"no configs/main.yaml under worktree {WORKTREE}; cannot set train.resume per panel ckpt")

# The live panel is whichever overfitX/batchN/panel.json the orchestrator pins for this cycle:
# it sets $OVERFITX_PANEL to that path. The <script_dir>/overfitX/panel.json default is
# EXAMPLE-ONLY (a worktree may not carry a real panel), present so the script is runnable
# standalone and so the error message points at a concrete location.
_EXAMPLE_PANEL = SCRIPT_DIR / "overfitX" / "panel.json"
PANEL_PATH = Path(os.environ.get("OVERFITX_PANEL", str(_EXAMPLE_PANEL)))
if not PANEL_PATH.exists():
    _die(f"panel file not found at {PANEL_PATH} (set $OVERFITX_PANEL to this cycle's "
         f"overfitX/batchN/panel.json). Expected a list of >=3 checkpoint entries "
         f"(template: {_EXAMPLE_PANEL.parent / 'panel.example.json'}).")

panel_raw = json.loads(PANEL_PATH.read_text())
# Allow either a bare list or {"panel": [...]} / {"_comment":..., "panel":[...]}.
if isinstance(panel_raw, dict):
    panel = panel_raw.get("panel") or panel_raw.get("checkpoints") or []
else:
    panel = panel_raw
panel = [e for e in panel if isinstance(e, dict) and e.get("path")]
if len(panel) < 3:
    _die(f"panel must have >=3 checkpoint entries with a 'path'; got {len(panel)} from {PANEL_PATH}")


# ---------------------------------------------------------------------------
# Read the config budget once (max_train_samples / max_train_flops) so we can
# (a) detect below-budget checkpoints and (b) confirm a resumed run does 0 steps.
# ---------------------------------------------------------------------------
import yaml  # noqa: E402  (deferred so the import error message is contextual)

cfg = yaml.safe_load(os.path.expandvars(CONFIG_IN_WORKTREE.read_text()))
MAX_SAMPLES = int(cfg["train"]["max_train_samples"])
MAX_FLOPS = int(cfg["train"]["max_train_flops"])
BATCH = int(cfg["train"]["batch_size"])


def _ckpt_counters(ckpt_path: Path):
    """Return (examples_seen, train_flops) from a checkpoint without loading model weights onto GPU."""
    try:
        import torch
    except Exception as e:  # pragma: no cover
        _die(f"torch import failed while inspecting {ckpt_path}: {e}")
    try:
        # weights_only=False because train.py saves python ints / dicts alongside tensors.
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception as e:
        _die(f"could not load checkpoint {ckpt_path}: {e}")
    missing = [k for k in ("examples_seen", "train_flops") if k not in ck]
    if missing:
        _die(f"checkpoint {ckpt_path} is missing {missing}; not a train.py pretraining checkpoint?")
    return int(ck["examples_seen"]), int(ck["train_flops"])


def _budget_met(examples_seen: int, train_flops: int) -> bool:
    """True iff resuming would run ZERO training iterations (mirrors train.py:628 guard)."""
    return not (examples_seen + BATCH <= MAX_SAMPLES and train_flops < MAX_FLOPS)


def _write_resume_config(ckpt_path: Path, output_dir: Path) -> Path:
    """Write a per-checkpoint config YAML (copy of main.yaml) with train.resume + output_dir set."""
    sub = yaml.safe_load(os.path.expandvars(CONFIG_IN_WORKTREE.read_text()))
    sub["train"]["resume"] = str(ckpt_path)
    sub["project"]["output_dir"] = str(output_dir)
    dst = output_dir.parent / f"{output_dir.name}.resume.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(sub, sort_keys=False))
    return dst


def _run_probe_only(entry: dict) -> dict:
    """Resume from a budget-complete checkpoint -> 0 training steps -> final probe -> summary.json."""
    ckpt = Path(os.path.expandvars(entry["path"]))
    if not ckpt.exists():
        _die(f"panel checkpoint does not exist: {ckpt} (role={entry.get('role')}, seed={entry.get('seed')})")

    examples_seen, train_flops = _ckpt_counters(ckpt)
    if not _budget_met(examples_seen, train_flops):
        # HARD STOP: this would actually train. Almost always a wrong panel path.
        _die(
            f"BELOW-BUDGET checkpoint refused: {ckpt}\n"
            f"  examples_seen={examples_seen} (cap {MAX_SAMPLES}), train_flops={train_flops} (cap {MAX_FLOPS}).\n"
            f"  Resuming this would run training steps and silently mutate the fixed panel. "
            f"Point this panel entry at a budget-COMPLETE checkpoint."
        )

    # Per-checkpoint scratch output dir inside the worktree (kept out of the repo tree root).
    # Namespace by EVO_EXPERIMENT_ID so concurrent readout evals do NOT share/rmtree the same
    # per-checkpoint scratch dir (prior shared-dir collisions raced rmtree+write -> FileNotFound /
    # "Directory not empty" crashes). Each readout job now owns its own subtree.
    exp_ns = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.environ.get("EVO_EXPERIMENT_ID", "default"))
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{entry.get('role','ckpt')}_{entry.get('seed','x')}")
    out_dir = WORKTREE / "overfitX" / "_preprobe_runs" / exp_ns / tag
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    resume_cfg = _write_resume_config(ckpt, out_dir)

    env = dict(os.environ)
    # Don't let the resumed run try to re-attach to the source run's wandb identity churn evo logs.
    env.setdefault("WANDB_MODE", "disabled")
    proc = subprocess.run(
        [sys.executable, str(WORKTREE / "train.py"), str(resume_cfg)],
        cwd=str(WORKTREE), env=env, capture_output=True, text=True,
    )
    summary_path = out_dir / "summary.json"
    if proc.returncode != 0 or not summary_path.exists():
        tail = (proc.stderr or proc.stdout or "")[-3000:]
        _die(f"probe-only run failed for {ckpt} (rc={proc.returncode}). summary={summary_path.exists()}.\n"
             f"--- train.py tail ---\n{tail}")

    summary = json.loads(summary_path.read_text())
    # The BELOW-BUDGET guard above already PROVED zero iterations will run (examples_seen+BATCH
    # exceeds the cap before any training), so we do NOT check steps_completed here: a budget-met
    # resume legitimately restores `step` from the checkpoint (~7812), it does not reset to 0.
    # Belt-and-suspenders post-check: confirm this run actually resumed from a checkpoint.
    if not summary.get("resumed_from"):
        _die(f"summary.json for {ckpt} has no resumed_from; expected a probe-only resume run. "
             f"train.py did not record a resume source — wrong config or stale summary.")
    if "final_probe_score" not in summary:
        _die(f"summary.json for {ckpt} has no final_probe_score (probe failed?)")
    return summary


# ---------------------------------------------------------------------------
# Run the panel, aggregate, and emit traces in evo_eval.py's exact format.
# ---------------------------------------------------------------------------
def main():
    now0 = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summaries = []
    for entry in panel:
        summaries.append((entry, _run_probe_only(entry)))

    # Per-probe = MEAN across panel checkpoints. final score = MEAN of final_probe_score.
    n = len(summaries)

    def _ckpt_label(i, e):
        return e.get("role", f"ckpt{i}") + f"_{e.get('seed','')}"

    scores = {}
    per_probe_datasets = {}   # flat {metric_key: panel mean} — mirrors evo_eval.py's `datasets`.
    per_probe_breakdown = {}  # per-checkpoint values, kept OUT of `datasets`.
    for name, (agg_key, token) in PROBES.items():
        vals = [float(s[agg_key]) for _, s in summaries]
        scores[name] = sum(vals) / n
        # `datasets` must be a FLAT {metric_key: value} dict exactly like evo_eval.py. Here each
        # value is the panel MEAN of that per-dataset metric across the checkpoints that report it.
        dataset_keys = {k for _, s in summaries for k in s if token in k and k != agg_key}
        per_probe_datasets[name] = {}
        for k in dataset_keys:
            kvals = [float(s[k]) for _, s in summaries if k in s]
            if kvals:
                per_probe_datasets[name][k] = sum(kvals) / len(kvals)
        # The per-checkpoint breakdown (agg + per-dataset) lives in a SEPARATE top-level trace
        # key so the ideator/verifier can still see which checkpoint moved.
        per_probe_breakdown[name] = {
            "agg": {_ckpt_label(i, e): float(s[agg_key]) for i, (e, s) in enumerate(summaries)},
            "datasets": {
                k: {_ckpt_label(i, e): float(s[k]) for i, (e, s) in enumerate(summaries) if k in s}
                for k in dataset_keys
            },
        }

    final_scores = [float(s["final_probe_score"]) for _, s in summaries]
    panel_mean = round(sum(final_scores) / n, 4)
    # tile_presentations: panel sum, for dashboard parity with evo_eval.py.
    panel_tile_presentations = sum(int(s.get("tile_presentations", 0)) for _, s in summaries)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    traces_dir = os.environ.get("EVO_TRACES_DIR")
    exp_id = os.environ.get("EVO_EXPERIMENT_ID", "unknown")
    if traces_dir:
        os.makedirs(traces_dir, exist_ok=True)
    for name in PROBES:
        val = scores[name]
        if traces_dir:
            trace = {"experiment_id": exp_id, "task_id": name, "score": val,
                     "status": "passed" if val >= 0.5 else "failed",
                     "ended_at": now,
                     "datasets": per_probe_datasets[name],
                     "per_checkpoint": per_probe_breakdown[name]}
            (Path(traces_dir) / f"task_{name}.json").write_text(json.dumps(trace, indent=2))

    result = {
        "score": panel_mean,
        "tasks": scores,
        "started_at": now0,
        "ended_at": now,
        "panel_size": n,
        "panel_final_probe_scores": {
            _ckpt_label(i, e): round(float(s["final_probe_score"]), 4)
            for i, (e, s) in enumerate(summaries)
        },
        "tile_presentations": panel_tile_presentations,
        "stop_reason": "preprobe_panel",
    }
    payload = json.dumps(result, indent=2)

    # Atomic claim-then-rename publish per the evo contract (identical to evo_eval.py).
    rp = os.environ.get("EVO_RESULT_PATH")
    if rp:
        os.close(os.open(rp, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        open(rp + ".tmp", "w").write(payload)
        os.replace(rp + ".tmp", rp)
    else:
        print(payload)


if __name__ == "__main__":
    main()
