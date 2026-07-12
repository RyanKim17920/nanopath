# Efficacy tests for pcgrad.py: prove gradient surgery is LIVE.
#   1. combined gradient DIFFERS from the naive sum when components conflict,
#   2. combined gradient is IDENTICAL to the naive sum when they do NOT conflict
#      (== the disabled/baseline behaviour, since disabled => plain backward()),
#   3. the real pcgrad_backward() writes correct .grad onto the stepped params.
#
# Run: /admin/home/ryan.kim/nanopath/.venv/bin/python3 tests/test_pcgrad.py
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcgrad import pcgrad_combine, pcgrad_backward


def test_combine_conflict_differs_from_sum():
    # g0 (main, DINO) and g1 (aux) point in opposing directions: cos = -0.707 < -0.1.
    g0 = torch.tensor([1.0, 0.0])
    g1 = torch.tensor([-1.0, 1.0])
    naive = g0 + g1  # [0, 1]
    combined = pcgrad_combine([g0, g1], main_loss_idx=0, conflict_threshold=-0.1, mode="cosine_veto")
    # Expected: main g0 preserved; g1 projected off g0 -> g1' = g1 - (g1.g0/||g0||^2) g0 = [0,1]; sum = [1,1].
    assert torch.allclose(combined, torch.tensor([1.0, 1.0])), combined
    assert not torch.allclose(combined, naive), "combined must differ from naive sum under conflict"
    print(f"  [conflict] naive={naive.tolist()} pcgrad={combined.tolist()} -> DIFFERS  OK")


def test_combine_no_conflict_equals_sum():
    # cos(g1,g0) = +0.707 > -0.1 -> no projection -> combined == naive sum.
    g0 = torch.tensor([1.0, 0.0])
    g1 = torch.tensor([1.0, 1.0])
    naive = g0 + g1
    combined = pcgrad_combine([g0, g1], main_loss_idx=0, conflict_threshold=-0.1, mode="cosine_veto")
    assert torch.allclose(combined, naive), (combined, naive)
    print(f"  [aligned]  naive={naive.tolist()} pcgrad={combined.tolist()} -> IDENTICAL  OK")


def test_combine_threshold_gates_mild_conflict():
    # cos = -0.05: classic PCGrad (dot<0) WOULD project, but cosine_veto thr=-0.1 must NOT.
    import math
    theta = math.acos(-0.05)
    g0 = torch.tensor([1.0, 0.0])
    g1 = torch.tensor([math.cos(theta), math.sin(theta)])
    combined = pcgrad_combine([g0, g1], main_loss_idx=0, conflict_threshold=-0.1, mode="cosine_veto")
    assert torch.allclose(combined, g0 + g1, atol=1e-6), "mild conflict (cos=-0.05) must be vetoed by thr=-0.1"
    # classic mode (dot<0) SHOULD project it, changing the result.
    classic = pcgrad_combine([g0, g1], main_loss_idx=0, mode="classic")
    assert not torch.allclose(classic, g0 + g1), "classic PCGrad should project any dot<0"
    print("  [threshold] cosine_veto vetoes cos=-0.05, classic projects it  OK")


def test_backward_wiring_conflict():
    # Real path: two component losses sharing one Linear's weights, with conflicting grads.
    # Linear objectives l0 = <y,a>, l1 = <y,-a> give grad1 = -grad0 exactly (cos = -1),
    # a guaranteed conflict that exercises autograd.grad + surgery + scatter deterministically.
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 3, bias=False)
    x = torch.randn(8, 4)
    a = torch.randn(8, 3)
    params = list(lin.parameters())

    def losses():
        y = lin(x)
        l0 = (y * a).sum()                # main
        l1 = (y * (-a)).sum()             # aux: grad is exactly -grad(l0)  (cos = -1)
        detached_zero = torch.zeros(())   # a detached component keeps its slot, contributes 0
        return [l0, l1, detached_zero]

    # Baseline: naive sum backward.
    lin.zero_grad(set_to_none=True)
    ls = losses()
    (ls[0] + ls[1]).backward()
    naive_grad = params[0].grad.clone()

    # PCGrad path.
    lin.zero_grad(set_to_none=True)
    diag = pcgrad_backward(losses(), params, main_loss_idx=0, conflict_threshold=-0.1, mode="cosine_veto")
    pcgrad_grad = params[0].grad.clone()

    assert diag["pcgrad_active_components"] == 2, diag
    assert diag["pcgrad_conflicts_with_main"] >= 1, diag
    assert not torch.allclose(pcgrad_grad, naive_grad), "pcgrad .grad must differ from naive under conflict"
    print(f"  [wiring/conflict] active={diag['pcgrad_active_components']} "
          f"conflicts={diag['pcgrad_conflicts_with_main']} cos_main_1={diag.get('pcgrad_cos_main_1'):.3f} "
          f"grad_delta={ (pcgrad_grad-naive_grad).abs().max():.4f} -> DIFFERS  OK")


def test_backward_wiring_no_conflict_matches_baseline():
    # When components are aligned, pcgrad_backward must reproduce the naive backward exactly.
    torch.manual_seed(1)
    lin = torch.nn.Linear(4, 3, bias=False)
    x = torch.randn(8, 4)
    params = list(lin.parameters())

    def losses():
        y = lin(x)
        l0 = (y - 1.0).pow(2).mean()
        l1 = 0.5 * (y - 1.0).pow(2).mean()  # same direction as l0 -> cos = +1, never conflicts
        return [l0, l1]

    lin.zero_grad(set_to_none=True)
    ls = losses()
    (ls[0] + ls[1]).backward()
    naive_grad = params[0].grad.clone()

    lin.zero_grad(set_to_none=True)
    diag = pcgrad_backward(losses(), params, main_loss_idx=0, conflict_threshold=-0.1, mode="cosine_veto")
    pcgrad_grad = params[0].grad.clone()

    assert diag["pcgrad_conflicts_with_main"] == 0, diag
    assert torch.allclose(pcgrad_grad, naive_grad, atol=1e-6), "aligned pcgrad must equal naive backward"
    print("  [wiring/aligned] pcgrad .grad == naive backward .grad  OK")


def test_ramp_weight_blend():
    # ramp_weight=0 must reproduce the naive sum even under conflict.
    torch.manual_seed(2)
    lin = torch.nn.Linear(4, 3, bias=False)
    x = torch.randn(8, 4)
    a = torch.randn(8, 3)
    params = list(lin.parameters())
    y = lin(x)
    ls = [(y * a).sum(), (y * (-a)).sum()]
    lin.zero_grad(set_to_none=True)
    (ls[0] + ls[1]).backward()
    naive = params[0].grad.clone()
    y2 = lin(x)
    ls2 = [(y2 * a).sum(), (y2 * (-a)).sum()]
    lin.zero_grad(set_to_none=True)
    pcgrad_backward(ls2, params, main_loss_idx=0, conflict_threshold=-0.1, ramp_weight=0.0)
    assert torch.allclose(params[0].grad, naive, atol=1e-6), "ramp_weight=0 must equal naive sum"
    print("  [ramp] ramp_weight=0 reproduces naive sum  OK")


if __name__ == "__main__":
    for fn in [
        test_combine_conflict_differs_from_sum,
        test_combine_no_conflict_equals_sum,
        test_combine_threshold_gates_mild_conflict,
        test_backward_wiring_conflict,
        test_backward_wiring_no_conflict_matches_baseline,
        test_ramp_weight_blend,
    ]:
        print(f"{fn.__name__}:")
        fn()
    print("\nALL PCGRAD TESTS PASSED")
