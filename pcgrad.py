# PCGrad (gradient surgery, Yu et al. 2020) for the multi-component SSL loss.
#
# The pretraining objective is a SUM of per-component losses (DINO, JEPA/I-JEPA,
# KDE uniformity, FINO metadata). When two components disagree on a parameter's
# direction their gradients CONFLICT (negative cosine); naively summing lets the
# larger term stomp the smaller. PCGrad resolves this by projecting a component's
# gradient onto the NORMAL PLANE of any other component it conflicts with, so the
# retained update no longer opposes that other component.
#
# This module is intentionally dependency-light (torch only) and free of any
# module-level side effects so it is importable + unit-testable in isolation
# (train.py itself reads sys.argv/config at import-adjacent scope). train.py
# imports `pcgrad_backward` and calls it in place of `total_loss.backward()`
# ONLY when `pcgrad.enabled` is true in the config; when disabled train.py takes
# the untouched `total_loss.backward()` path so the baseline is byte-identical.

import torch


def _flatten(grads):
    # Flatten a per-parameter grad list (with None for untouched params) into one
    # 1-D vector, substituting zeros for None so every component shares a layout.
    return torch.cat([g.reshape(-1) for g in grads])


def pcgrad_combine(grads, main_loss_idx=0, conflict_threshold=0.0, mode="cosine_veto"):
    """Combine a list of flattened per-component gradient vectors via PCGrad.

    grads: list of 1-D tensors, all the same length (one per loss component).
    main_loss_idx: the "primary" objective (default DINO=0). Its gradient is
        preserved unmodified; every other component is projected off the
        conflicting components (main first) so the primary objective keeps
        priority in the combined update.
    conflict_threshold: with mode="cosine_veto", a pair conflicts (and is
        projected) only when cos(g_i, g_j) < conflict_threshold. The paper's
        rule is dot < 0 (threshold 0); the overfitX sweep uses -0.1 so only
        clearly-opposed pairs are deconflicted.
    mode: "cosine_veto" -> project when cosine < conflict_threshold.
          anything else  -> classic PCGrad (project when dot < 0).

    Returns the combined 1-D gradient vector. With no conflicting pairs this is
    exactly sum(grads) (i.e. identical to a naive backward).
    """
    n = len(grads)
    pc = [g.clone() for g in grads]
    # Deterministic order: for each non-main component, deconflict against the
    # main loss first, then the remaining components in index order.
    others_order = {
        i: ([main_loss_idx] + [k for k in range(n) if k != i and k != main_loss_idx])
        for i in range(n)
    }
    for i in range(n):
        if i == main_loss_idx:
            continue  # primary objective's gradient is never projected away
        for j in others_order[i]:
            gj = grads[j]
            gj_sq = torch.dot(gj, gj)
            if float(gj_sq) <= 0.0:
                continue  # zero reference gradient -> nothing to project onto
            gi = pc[i]
            dot = torch.dot(gi, gj)
            if mode == "cosine_veto":
                denom = gi.norm() * gj.norm() + 1e-12
                conflict = float(dot / denom) < conflict_threshold
            else:
                conflict = float(dot) < 0.0
            if conflict:
                pc[i] = gi - (dot / gj_sq) * gj
    return torch.stack(pc, dim=0).sum(dim=0)


def pcgrad_backward(losses, params, main_loss_idx=0, conflict_threshold=0.0,
                    mode="cosine_veto", ramp_weight=1.0):
    """Compute per-component gradients, apply PCGrad surgery, and write the
    result into `p.grad` for every parameter in `params` (replacing what a
    single `sum(losses).backward()` would have produced).

    losses: list of scalar loss tensors, one per component, in a FIXED order
        so main_loss_idx stays meaningful. Components that are detached
        constants (requires_grad False / no grad_fn) contribute a zero gradient
        and keep their slot.
    params: the parameters that the optimizer will step (grads are taken w.r.t.
        exactly these, so the surgical grad lands on the stepped tensors).
    ramp_weight: blend factor in [0,1] between the naive sum (0) and the full
        PCGrad-combined gradient (1): final = naive + w*(pcgrad - naive). Lets
        the caller ramp surgery in over early training where grads are noisy.

    Returns a dict of diagnostics (per-component grad norms, conflict count,
    cosine-with-main) for logging; its side effect is setting p.grad.
    """
    params = [p for p in params if p is not None]
    numels = [p.numel() for p in params]
    shapes = [p.shape for p in params]

    # Per-component gradient extraction via per-component .backward() into p.grad
    # (grads zeroed between components), NOT a nested
    # `torch.autograd.grad(loss, params, ...)`. The autograd.grad path re-enters
    # the engine with an `inputs=params` restriction; on the live GPU graph — which
    # under FlopCounterMode on the first step carries multi-grad / accumulate-grad
    # hooks — that restriction trips the C++ engine:
    #   RuntimeError: A leaf node was passed to _will_engine_execute_node but we
    #   are currently running autograd.grad(). This is currently not supported.
    # Plain .backward() takes the ordinary full-graph path those hooks already
    # support (it is exactly what the disabled-default baseline runs), so per
    # component we zero grads, backward (retaining the graph for all but the last),
    # then read p.grad for the params the optimizer steps. Detached/constant
    # components (requires_grad False / no grad_fn) contribute a zero vector.
    saved_grads = [p.grad for p in params]
    for p in params:
        p.grad = None
    active_idx = [
        i for i, loss in enumerate(losses)
        if loss is not None and getattr(loss, "requires_grad", False) and loss.grad_fn is not None
    ]
    comp_by_idx = {}
    for pos, i in enumerate(active_idx):
        for p in params:
            p.grad = None
        losses[i].backward(retain_graph=(pos < len(active_idx) - 1))
        comp_by_idx[i] = _flatten([
            p.grad.detach() if p.grad is not None
            else torch.zeros(n, device=p.device, dtype=p.dtype)
            for p, n in zip(params, numels)
        ])
    # Discard the per-component grads left in p.grad; the surgical result is
    # written back explicitly below.
    for p in params:
        p.grad = None
    comp_vecs = [comp_by_idx.get(i) for i in range(len(losses))]
    active = [i in comp_by_idx for i in range(len(losses))]

    total = sum(numels)
    ref = next((v for v in comp_vecs if v is not None), None)
    if ref is None:
        for p, g in zip(params, saved_grads):
            p.grad = g  # restore prior grads; nothing to deconflict
        return {"pcgrad_active_components": 0}
    zero = torch.zeros(total, device=ref.device, dtype=ref.dtype)
    comp_vecs = [v if v is not None else zero for v in comp_vecs]

    naive = torch.stack(comp_vecs, dim=0).sum(dim=0)
    combined = pcgrad_combine(
        comp_vecs, main_loss_idx=main_loss_idx,
        conflict_threshold=conflict_threshold, mode=mode,
    )
    w = float(ramp_weight)
    final = naive if w <= 0.0 else (combined if w >= 1.0 else naive + w * (combined - naive))

    # Scatter the combined vector back into per-parameter .grad tensors.
    offset = 0
    for p, n, shp in zip(params, numels, shapes):
        p.grad = final[offset:offset + n].view(shp).clone()
        offset += n

    # Diagnostics (cheap): cosine of each active component with the main loss.
    diag = {"pcgrad_active_components": int(sum(active))}
    gm = comp_vecs[main_loss_idx]
    gm_norm = gm.norm()
    n_conflict = 0
    for i, v in enumerate(comp_vecs):
        if i == main_loss_idx or not active[i]:
            continue
        cos = float(torch.dot(v, gm) / (v.norm() * gm_norm + 1e-12))
        diag[f"pcgrad_cos_main_{i}"] = cos
        if cos < conflict_threshold:
            n_conflict += 1
    diag["pcgrad_conflicts_with_main"] = n_conflict
    return diag
