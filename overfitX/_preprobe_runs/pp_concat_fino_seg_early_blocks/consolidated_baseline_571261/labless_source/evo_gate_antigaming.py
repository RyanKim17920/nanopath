# evo gate (pre-phase): ANTI-EVAL-GAMING. Fails the clear, unambiguous cheat: special-casing the
# EVAL/probe path with a transform fit on the eval batch or keyed to a specific probe (the pathorob
# #3 / batch-PCA pattern). Conservative by design — comments are stripped and only eval-ACTIVE
# patterns are flagged, so it does NOT false-positive on:
#   * multi-layer readout concat (#1 CLS-concat, #2 dense-fuse) — legitimate;
#   * TRAINING-TIME transforms/losses, incl. ones gated `if not self.training: return x` (OFF at
#     eval) — that is the GOOD direction (bake the idea into training).
# Subtler gray cases (e.g. eval-time whitening fit on the probe batch) are left to the semantic
# verifier + the project.md rule, not this static gate.
import re, sys, os

# gate is invoked as `python3 {worktree}/evo_gate_antigaming.py {worktree}` — read source from the
# worktree dir (argv[1]) if given, else cwd.
WT = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else "."

def strip_comments(code):
    # drop triple-quoted blocks and #-comments so dataset names in prose don't trigger
    code = re.sub(r'"""(?:.|\n)*?"""', "", code)
    code = re.sub(r"'''(?:.|\n)*?'''", "", code)
    out = []
    for ln in code.splitlines():
        out.append(re.sub(r"#.*$", "", ln))
    return "\n".join(out)

src = ""
for f in (os.path.join(WT,"model.py"), os.path.join(WT,"train.py")):
    if os.path.exists(f):
        src += "\n" + strip_comments(open(f, encoding="utf-8", errors="ignore").read())

viol = []

# (0) ROBUST READOUT-DIFF: the probe-facing readout methods `probe_features` (CLS probes) and
# `encode_image` (seg) MUST be byte-identical to the canonical baseline. Legit TRAINING changes
# (objective/aug/curriculum/architecture) never touch these — only eval-only readout engineering
# (#1 multi-layer/CLS-concat, #2 dense-fusion) does. This exact diff catches ALL such variants with
# no regex brittleness and no false positives. Reference: ~/nanopath/baseline_readout.txt.
def _readout(model_path):
    import ast
    try:
        s = open(model_path, encoding="utf-8", errors="ignore").read()
        t = ast.parse(s)
        return "\n".join(ast.get_source_segment(s, n) for n in ast.walk(t)
                         if isinstance(n, ast.FunctionDef) and n.name in ("probe_features", "encode_image"))
    except Exception:
        return None
_REF = "/admin/home/ryan.kim/nanopath/baseline_readout.txt"
_mp = os.path.join(WT, "model.py")
if os.path.exists(_mp) and os.path.exists(_REF):
    cur = _readout(_mp)
    ref = open(_REF, encoding="utf-8", errors="ignore").read().strip()
    if cur is not None and cur.strip() != ref:
        viol.append("probe_features/encode_image MODIFIED vs baseline (probe-only readout "
                    "engineering — #1 concat / #2 dense-fuse). Probes must read the RAW native feature.")

# (1) The pathorob #3 code, verbatim (function / module constant — not a bare word in prose).
if re.search(r"_suppress_top_pcs|ROBUST_NORM\b", src):
    viol.append("pathorob-style per-probe nuisance suppression (_suppress_top_pcs / ROBUST_NORM)")

# (2) Per-probe special-casing: a benchmark DATASET NAME appears in model.py CODE (comments stripped).
#     The trunk must be probe-agnostic; if forward()/encode_image branches on a benchmark name, it is
#     behaving differently per-benchmark.
modsrc = strip_comments(open(os.path.join(WT,"model.py"), encoding="utf-8", errors="ignore").read()) if os.path.exists(os.path.join(WT,"model.py")) else ""
if re.search(r"\b(pathorob|pannuke|monusac|consep|leopard_bcr|cptac_pda|surgen|break_his)\b", modsrc):
    viol.append("model.py CODE references a specific benchmark dataset (per-probe special-casing)")

# (3) Eval-ACTIVE batch-statistic fit: a flag that ENABLES a transform when NOT training (the
#     pathorob `apply = ... and not self.training` shape), as opposed to `if not self.training: return`
#     (training-only, legal). Require the enable-shape AND a batch-stat op nearby.
if re.search(r"(and|or)\s+not\s+self\.training", src) and \
   re.search(r"torch\.linalg\.(svd|eigh)|pca_lowrank|_suppress_top_pcs", src):
    # exclude the pure early-return guard form
    if not re.search(r"not\s+self\.training\s*:\s*return", src):
        viol.append("eval-ACTIVE batch-statistic transform (enabled only when not training)")

if viol:
    print("ANTIGAMING=violated :: " + " | ".join(dict.fromkeys(viol)))
    print("Bake the idea into TRAINING (scanner/stain-invariance or decorrelation LOSS), not an "
          "eval-time transform fit on the benchmark batch.")
    sys.exit(1)
print("ANTIGAMING=ok")
sys.exit(0)
