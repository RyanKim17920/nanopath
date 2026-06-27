# evo_eval.py — eval-only inline instrumentation for the nanopath probe benchmark.
# Parses a COMPLETED run's summary.json (produced by train.py + the locked probe.py)
# and emits one evo trace per downstream probe + the mean (final_probe_score) as the
# experiment score. It never retrains and never touches probe.py / benchmarking/.
# Usage (registered as the evo --benchmark command): python3 evo_eval.py <summary.json>
import json, os, sys
from datetime import datetime, timezone

summary = json.load(open(sys.argv[1]))

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
