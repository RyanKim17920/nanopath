# evo gate (pre-phase): ANTI-EVAL-GAMING (revised 2026-06-15 per admin ruling).
# RULE: the probe input may be ANYTHING (CLS concat, patch stacking, multi-layer readout) AS LONG
# AS it is (a) the SAME for every probe (the model must not vary per-probe), and (b) NOT fit on /
# derived from the probe's own data ("looking at the pathorob/survival data is not allowed").
# So this gate fails ONLY:
#   (1) per-probe special-casing  — _suppress_top_pcs/ROBUST_NORM, or a benchmark dataset name
#       referenced in model.py code (the trunk behaves differently per benchmark);
#   (2) fitting on the PROBE/EVAL batch — PCA/SVD/eigh/whitening/covariance computed on the incoming
#       features in any EVAL-REACHABLE code path (probe_features, encode_image, or any part of
#       forward that runs when self.training is False — incl. plain `if not self.training:` blocks,
#       the `else` of `if self.training:`, and UNGUARDED ops that run at eval too).
# It does NOT fail uniform readout engineering (concat/stacking with no data fitting), nor
# TRAINING-TIME losses/transforms guarded by `if self.training:` (decorrelation/whitening computed
# during training are encouraged).
import re, sys, os, ast

WT = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else "."

def strip_comments(code):
    code = re.sub(r'"""(?:.|\n)*?"""', "", code)
    code = re.sub(r"'''(?:.|\n)*?'''", "", code)
    return "\n".join(re.sub(r"#.*$", "", ln) for ln in code.splitlines())

mp = os.path.join(WT, "model.py")
model_raw = open(mp, encoding="utf-8", errors="ignore").read() if os.path.exists(mp) else ""
model_src = strip_comments(model_raw)
viol = []

# Only ACTUAL fit/decomposition COMPUTATIONS count — not bare names like `whiten`/`zca`, which also
# match the legal APPLY of a frozen training buffer (e.g. a method called `cls_whiten` that just does a
# matmul with an EMA-accumulated matrix). Matching those names was what made the gate false-positive on
# legitimate training-time whitening. The real giveaway of "fitting on the batch" is a decomposition call.
FIT = (r"(torch\.linalg\.(svd|eigh|qr|cholesky|lstsq)|pca_lowrank|\.cov\(|\.symeig\(|"
       r"torch\.svd\(|torch\.eig\(|torch\.symeig\()")
FITRE = re.compile(FIT)

# (1) per-probe special-casing -----------------------------------------------------------------
if re.search(r"_suppress_top_pcs|ROBUST_NORM\b", model_src):
    viol.append("per-probe nuisance suppression (_suppress_top_pcs / ROBUST_NORM)")
if re.search(r"\b(pathorob|pannuke|monusac|consep|leopard_bcr|cptac_pda|surgen|break_his)\b", model_src):
    viol.append("model.py CODE references a specific benchmark dataset (per-probe special-casing)")

# (2) fitting on the eval/probe batch — AST guard analysis -------------------------------------
# A FIT op is ILLEGAL iff it can execute when self.training is False. We determine that from the
# enclosing `if self.training:` / `if not self.training:` guards (the ONLY way to make code
# training-only is to guard it with `if self.training:`). Everything else is eval-reachable.
def _training_test(test):
    # 'train' if test is `self.training`, 'eval' if `not self.training`, else None
    if isinstance(test, ast.Attribute) and test.attr == "training" \
       and isinstance(test.value, ast.Name) and test.value.id == "self":
        return "train"
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) \
       and _training_test(test.operand) == "train":
        return "eval"
    return None

def _contains(node, target):
    return any(c is target for c in ast.walk(node))

def _eval_reachable(node, parents):
    # default reachable at eval; a single training-only guard above it makes it legal
    cur = node
    while cur in parents:
        p = parents[cur]
        if isinstance(p, ast.If):
            t = _training_test(p.test)
            in_body = any(_contains(b, cur) or b is cur for b in p.body)
            if t == "train" and in_body:
                return False                     # inside `if self.training:` body -> training only
            if t == "eval" and not in_body:
                return False                     # inside `else` of `if not self.training:` -> training
        cur = p
    return True

# We start from the methods the probe harness actually calls at eval (probe_features, encode_image,
# forward) and follow self.<method>() / self.<submodule>() calls transitively, but ONLY along
# eval-reachable code paths (skipping `if self.training:` bodies). A fit COMPUTATION found anywhere on
# that closure = looking at the probe/eval batch -> illegal. A fit that lives in a method NOT reachable
# from eval (e.g. `update_stats`, called only from the train loop) is never visited -> legal.
EVAL_ENTRIES = ("probe_features", "encode_image", "forward")
TRAIN_ONLY_NAMES = ("update_stats", "_update_stats", "_update_stats_fp32")

eval_fit_hit = None
if model_raw.strip():
    try:
        tree = ast.parse(model_raw)
        parents = {}
        for p in ast.walk(tree):
            for c in ast.iter_child_nodes(p):
                parents[c] = p
        funcs = {}                       # method name -> FunctionDef (first def wins)
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef):
                funcs.setdefault(n.name, n)
        class_methods = {}               # class name -> {method: FunctionDef}
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                class_methods[n.name] = {m.name: m for m in n.body if isinstance(m, ast.FunctionDef)}
        attr_class = {}                  # self.<attr> -> ClassName, from `self.x = Class(...)` in __init__
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                f = n.value.func
                cname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
                for tgt in n.targets:
                    if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                        attr_class[tgt.attr] = cname

        def scan(fn):
            """In fn's EVAL-reachable code: report a FIT computation, else yield self.<target> call names."""
            targets = []
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call) or not _eval_reachable(node, parents):
                    continue
                seg = strip_comments(ast.get_source_segment(model_raw, node) or "")
                if FITRE.search(seg.split(")")[0] + ")"):
                    return ("FIT", seg.splitlines()[0].strip()[:80])
                f = node.func           # follow self.<attr>(...)
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self":
                    targets.append(f.attr)
            return ("CALLS", targets)

        queue = [funcs[e] for e in EVAL_ENTRIES if e in funcs]
        seen = set()
        while queue and not eval_fit_hit:
            fn = queue.pop()
            if id(fn) in seen:
                continue
            seen.add(id(fn))
            kind, payload = scan(fn)
            if kind == "FIT":
                eval_fit_hit = payload
                break
            for a in payload:
                if a in TRAIN_ONLY_NAMES:
                    continue                                   # train-loop-only updater, not an eval path
                if a in funcs:
                    queue.append(funcs[a])
                cls = attr_class.get(a)                        # submodule: follow its forward/__call__
                if cls and cls in class_methods:
                    for m in ("forward", "__call__"):
                        if m in class_methods[cls]:
                            queue.append(class_methods[cls][m])
    except SyntaxError:
        # if model.py won't parse, fall back to the conservative text heuristic
        if FITRE.search(model_src):
            eval_fit_hit = "FIT op present (model.py unparseable — flagged conservatively)"

if eval_fit_hit:
    viol.append("data-fit transform (PCA/SVD/eigh/whitening/cov) reachable at EVAL — looking at "
                "probe/eval data :: " + eval_fit_hit)

if viol:
    print("ANTIGAMING=violated :: " + " | ".join(dict.fromkeys(viol)))
    print("Allowed: UNIFORM readout (same for all probes, no data fitting) + TRAINING-ONLY transforms "
          "guarded by `if self.training:`. Banned: per-probe variation + fitting on the probe/eval "
          "batch. Bake data-derived ideas into TRAINING instead.")
    sys.exit(1)
print("ANTIGAMING=ok")
sys.exit(0)
