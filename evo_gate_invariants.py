# evo gate: Labless-eligibility invariants. Exits non-zero unless the run trained
# exactly the leaderboard budget — stops an experiment "winning" by training longer
# or on more tiles than the frozen baseline. Reads the same summary.json the score
# comes from.
import json, sys
s = json.load(open(sys.argv[1]))
ok = (s["max_train_samples"] == 1_000_000
      and s["tile_presentations"] <= 1_000_000
      and s["max_train_flops"] == 1_000_000_000_000_000_000)
print("INVARIANTS=ok" if ok else
      f"INVARIANTS=violated samples={s.get('max_train_samples')} "
      f"tiles={s.get('tile_presentations')} flops={s.get('max_train_flops')}")
sys.exit(0 if ok else 1)
