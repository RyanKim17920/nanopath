# BoehmK Survival

## Role In Nanopath

`boehmk_pfs` is an ovarian slide-level survival probe. Upstream PathoBench calls the endpoint PFS, short for progression-free survival. It contributes Harrell's validation c-index as one of the two datasets averaged into the README survival column.

## Source

- Labels: `MahmoodLab/Patho-Bench`, file `boehmk_/PFS/k=all.tsv`
- Upstream metadata: `task_type: survival`, `metrics: cindex`, with `PFS_event` and `PFS_days`
- Raw WSIs: BOEHMK Synapse project at `https://www.synapse.org/Synapse:syn25946117/wiki/611576`
- Portable setup mirror: `medarc/nanopath`, under `probes/boehmk_pfs/`

## Split And Patches

Nanopath vendors `boehmk_pfs.json`, derived from PathoBench BOEHMK survival/PFS fold_0. PathoBench fold_0 test remains held out; Nanopath uses deterministic 3-fold event-stratified validation over the fold_0 train pool.

| split | cases/slides | event labels | cached patches |
|---|---:|---|---:|
| train pool | 146 | 96 event / 50 censored | 271,467 cached 20x/512 tissue tiles |
| per-fold train | 97-98 | reused | reused |
| per-fold val | 48-49 | reused | reused |
| held-out PathoBench test | 37 | 24 event / 13 censored | not read |

## Implementation

`prepare.py` normally downloads the pre-extracted `medarc/nanopath` parquet cache: `patches.parquet`, `labels.tsv`, and `tiling_version.txt`. `fetch_boehmk_pfs_from_synapse()` is the regeneration helper for rebuilding that mirror after the user has accepted the BOEHMK Synapse access terms. It downloads the Synapse `data.tar.gz`, extracts a deterministic 20x, 512 px, 0-overlap tissue grid, and writes one combined `patches.parquet`.

`probe.py` streams a deterministic raster-spaced sub-bag of up to 768 cached patches per slide with a no-crop square resize, mean-pools patch embeddings by slide, pools by case unit, z-scores features with train-fold statistics, and fits `sksurv.linear_model.CoxPHSurvivalAnalysis(alpha=2.0)` on the full standardized feature matrix. It reports the mean validation Harrell's c-index across the three event-stratified folds. The head has no dimensionality reduction, elastic-net sparsity, or alpha sweep.

## Null Distribution Audit

![BoehmK PFS null distributions](null_plots/boehmk_pfs_null_distributions.png)

The orange null uses randomized-weight DINOv2-small evaluations through the same probe path: mean 0.5518, std 0.0085, max 0.5683. An iid random risk score still centers at chance (0.5004 in a 200-draw Harrell check), and exact CoxPH on iid Gaussian 384-d features centers near chance (0.492 over 30 draws), so the high randomized-DINO null is not a c-index or CoxPH-analysis offset.

This makes BoehmK PFS a cautionary survival benchmark. DINOv2 never sees `case_id`, but nuisance controls show that the folds expose non-pathology shortcuts: numeric `case_id` alone scores 0.558 on Nanopath's 3-fold split and 0.562 over the five official PathoBench folds; JPEG byte-length stats score 0.526; 128-tile thumbnail summaries score 0.538 from RGB mean, 0.575 from contrast, and 0.598 from a darkness/tissue proxy. These controls suggest randomized DINOv2 is acting as a low-level stain/scanner/tissue-density feature extractor, not discovering survival biology. Treat raw BoehmK c-index as shortcut-sensitive unless folds or scoring make these nuisance baselines return to ~0.50.

## Difference From Original Usage

PathoBench's BOEHMK survival task reports Harrell's c-index. PathoBench is designed for standardized task evaluation across folds and pools Trident patch embeddings. Nanopath keeps the same 20x/512 patch-grid cache, uses a deterministic up-to-768-tile sub-bag for final-probe runtime, and uses repeated train-derived internal validation for fast iteration without scoring the PathoBench test fold. The survival head is intentionally simple and fixed: train-fold z-scored pooled features into CoxPH alpha 2.0. The ridge penalty preserves all embedding dimensions and avoids turning the probe into a sparse feature-selection benchmark, while standardization keeps the penalty scale-comparable. The tissue mask is a lightweight deterministic thumbnail mask rather than Trident HEST segmentation.
