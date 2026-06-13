# exp_0176 — land the eval-only JBU seg densifier (port of exp_0167, stale-summary fixed)

## What this is
Eval-only image-guided Joint Bilateral Upsampler (JBU / FeatUp-style) on the FROZEN seg readout.
`model.encode_image` (probe-only entrypoint) upsamples the last-4-block fused 16x16 patch grid ->
32x32 using bilateral weights = spatial Gaussian x range Gaussian on the input image's grayscale
structure. Parameter-free, no backward, no trunk retrain. CLS path (`probe_features` ->
`x_norm_clstoken`) is byte-identical -> all 7 CLS probes unchanged; only seg (worst @0.289) moves.

Parent: exp_0145 (best-path; encode_image readout untouched). Source of fusion: exp_0114.
exp_0167 probed THIS EXACT densifier at 0.658352 (seg 0.3273) but was STRANDED by the stale-summary
bug (its run_summary.json still pointed at exp_0114, score 0.6577). This node fixes only that bug.

## Files changed vs exp_0145
- model.py: `UPSAMPLE_GRID=32`, `JBU_SIGMA_RANGE=0.1`, `_jbu_upsample()`, rewired `encode_image`
  to reshape fused patches -> [B,C,h,w], JBU-upsample to 32x32, return [regs || dense patches].
- train.py: (1) reprobe_only fresh-wandb-identity guard (skip resuming parent's wandb run);
  (2) probe co-location OOM guard (CPU-evict trainer state before each probe subprocess, restore
  after; tolerate CalledProcessError). NOT a killpg/pkill guard (harness-bug-1).
- configs/jf_i_hed07_jbu.yaml: NEW. output_dir=/data/.../exp_0176 (own id), resume=exp_0114/latest.pt,
  reprobe_only=true, posthoc_ema=false. examples_seen==999936 so train loop never enters.

## RUN RECIPE (do NOT run until verifier passes; orchestrator launches `evo run`)
1. [DONE] FIRST: wrote run_summary.json with THIS exp's identity (experiment_id/config_path/
   output_dir/wandb_name all exp_0176, final_probe_score=null placeholder). grep-verified: only
   exp_0176 appears -> evo_eval stale-guard passes.
2. Launch the reprobe SLURM job with configs/jf_i_hed07_jbu.yaml (slurm/train_1gpu.sbatch). It
   resumes exp_0114/latest.pt, skips the train loop, runs the final probe with the JBU densifier,
   and writes /data/ryan.kim/nanopath/evo/exp_0176/summary.json.
3. LAST: after the probe finishes, read the real mean_probe_score from the slurm *.out
   ('mean_probe_score=...') and copy the produced summary.json -> worktree/run_summary.json,
   confirming final_probe_score is the REAL post-probe value (NOT 0.6577, NOT null). Re-grep
   identity is still exp_0176. THEN `evo run exp_0176`.
4. If `evo run` reports STALE SUMMARY, the identity write did not take -> fix before declaring done.

## Expected: seg ~0.3273, score ~0.658352 (>0.6577 best). CLS probes byte-identical.
