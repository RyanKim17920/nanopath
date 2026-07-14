# evo_eval.py — eval-only inline instrumentation for the nanopath probe benchmark.
# Parses a COMPLETED run's summary.json (produced by train.py + the locked probe.py)
# and emits one evo trace per downstream probe + the mean (final_probe_score) as the
# experiment score. It never retrains and never touches probe.py / benchmarking/.
# Usage (registered as the evo --benchmark command): python3 evo_eval.py <summary.json>
import json, os, re, sys
from datetime import datetime, timezone

summary = json.load(open(sys.argv[1]))

# STALE-SUMMARY GUARD (anti false-commit). A new worktree inherits its parent's run_summary.json
# at creation time. If a subagent calls `evo run` BEFORE writing THIS experiment's own summary,
# we would score the parent's stale copy verbatim (byte-identical scores, +0.0000 delta) and
# commit a fake node — the "score found before the run / became a starting node" bug
# (exp_0135/0139/0145). Detect it: the summary's config_path/output_dir must reference THIS
# experiment id, not a different exp_NNNN. Crash loudly rather than commit a phantom score.
_eid = os.environ.get("EVO_EXPERIMENT_ID", "")
if re.fullmatch(r"exp_\d{4}", _eid or ""):
    _ref = f"{summary.get('config_path','')} {summary.get('output_dir','')} {summary.get('wandb','')}"
    _others = set(re.findall(r"exp_\d{4}", _ref)) - {_eid}
    if _others and _eid not in _ref:
        sys.exit(f"STALE SUMMARY: run_summary.json belongs to {sorted(_others)}, not {_eid}. You "
                 f"called `evo run` before writing {_eid}'s OWN summary.json (it is still the parent's "
                 f"inherited copy). Produce {_eid}'s real run_summary.json first, then re-run.")

# The 8 canonical probes -> their summary.json aggregate key, plus a substring that
# selects that probe's constituent per-dataset metrics (attached as trace extras so
# the ideator/verifier can see WHICH dataset inside a probe moved).
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

traces_dir = os.environ.get("EVO_TRACES_DIR")
exp_id = os.environ.get("EVO_EXPERIMENT_ID", "unknown")
now = datetime.now(timezone.utc).isoformat(timespec="seconds")
if traces_dir:
    os.makedirs(traces_dir, exist_ok=True)

scores = {}
for name, (agg_key, token) in PROBES.items():
    val = float(summary[agg_key])
    scores[name] = val
    if traces_dir:
        detail = {k: summary[k] for k in summary if token in k and k != agg_key}
        trace = {"experiment_id": exp_id, "task_id": name, "score": val,
                 "status": "passed" if val >= 0.5 else "failed",
                 "ended_at": now, "datasets": detail}
        (open(os.path.join(traces_dir, f"task_{name}.json"), "w")
         .write(json.dumps(trace, indent=2)))

score = round(float(summary["final_probe_score"]), 4)
result = {"score": score, "tasks": scores,
          "started_at": now, "ended_at": now,
          "stop_reason": summary.get("stop_reason"),
          "tile_presentations": summary.get("tile_presentations")}
payload = json.dumps(result, indent=2)

# Atomic claim-then-rename publish per the evo contract.
rp = os.environ.get("EVO_RESULT_PATH")
if rp:
    os.close(os.open(rp, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    open(rp + ".tmp", "w").write(payload)
    os.replace(rp + ".tmp", rp)
else:
    print(payload)
