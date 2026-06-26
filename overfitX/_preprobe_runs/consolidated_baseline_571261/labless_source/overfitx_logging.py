"""Canonical per-step logging for nanopath-overfitX worktrees.

ONE source of truth so every worktree (and every future wave) logs the same
core metrics, to metrics.jsonl AND to the same WandB panel with the same prefix.

Design: the per-step ``train_log`` dict ALREADY carries correct values for
dino/jepa/kde/total/grad_norm/lr (built by each branch's own reduced-dict).
This shim therefore READS those from train_log -- it never re-passes
branch-specific loss variables (doing so previously clobbered train_log["dino"]
with the FINO value). The only externally-supplied value is ``fino`` (the FINO /
meta_loss term), which is guaranteed present and pushed into train_log.

Idempotent guard marker used by the injector:  # ovfx_canon_log
"""

CANON_KEYS = ("dino", "jepa", "kde", "fino", "total", "grad_norm", "lr")
PANEL_PREFIX = "main-train-results/"


def _f(v):
    try:
        return float(v.detach()) if hasattr(v, "detach") else float(v)
    except Exception:
        return None


def ovfx_canon_step(train_log, *, fino=None, wandb_run=None, wandb_step=None):
    """Guarantee the canonical core metrics in *train_log* and mirror them to the
    'main-train-results/' WandB panel.

    Parameters
    ----------
    train_log : dict   per-step log dict (mutated in-place). Already holds correct
                       dino/jepa/kde/total/grad_norm/lr; this only normalizes them
                       to float and guarantees the 'fino' key.
    fino       : tensor or float, optional   the FINO / meta_loss term to record.
    wandb_run  : WandB run or None
    wandb_step : int or None

    Returns the same (mutated) train_log.
    """
    # Guarantee the one key that isn't always present from the branch's own dict.
    if fino is not None:
        fv = _f(fino)
        if fv is not None:
            train_log["fino"] = fv
    # Normalize existing canonical keys to float IN PLACE. Never fabricate a
    # missing key (no key invented => no mislabeling), never overwrite with a
    # foreign variable.
    for k in CANON_KEYS:
        if k in train_log and train_log[k] is not None:
            fv = _f(train_log[k])
            if fv is not None:
                train_log[k] = fv
    # Mirror the canonical core to the single shared panel.
    if wandb_run is not None:
        core = {k: train_log[k] for k in CANON_KEYS if k in train_log}
        try:
            wandb_run.log({PANEL_PREFIX + k: v for k, v in core.items()},
                          step=wandb_step)
        except Exception:
            pass
    return train_log
