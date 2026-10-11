import numpy as np
import torch

from ba3active.metrics import ranking, softmax
from ba3active.rag_moe import RAGMoE


def test_rag_moe_has_compact_parameter_budget():
    model = RAGMoE(width=24)
    assert sum(parameter.numel() for parameter in model.parameters()) == 682


def test_structural_influence_is_bounded_by_gate_and_reliability():
    torch.manual_seed(4)
    model = RAGMoE(width=24).eval()
    x = torch.randn(11, 9)
    structural = torch.linspace(-1.5, 1.5, 11)
    rho = 0.42
    global_features = torch.tensor([[rho, 0.5, 0.2, 0.35]] * len(x))

    with torch.no_grad():
        temporal, alpha, fused = model(x, structural, global_features)

    assert torch.all((0.0 <= alpha) & (alpha <= 1.0))
    assert torch.all(torch.abs(fused - temporal) <= rho * torch.abs(structural) + 1e-7)


def test_zero_reliability_recovers_temporal_expert():
    torch.manual_seed(8)
    model = RAGMoE(width=24).eval()
    x = torch.randn(7, 9)
    structural = torch.randn(7)
    global_features = torch.tensor([[0.0, 0.5, 0.1, 0.4]] * len(x))

    with torch.no_grad():
        temporal, _, fused = model(x, structural, global_features)

    assert torch.allclose(fused, temporal)


def test_ranking_receives_a_valid_probability_distribution():
    score = np.asarray([-2.0, 0.4, 1.1, -0.2])
    probability = softmax(score)
    metrics = ranking(probability, truth_index=2)

    assert np.isclose(probability.sum(), 1.0)
    assert np.all(probability >= 0.0)
    assert metrics["rank"] == 1
    assert metrics["nll"] >= 0.0
    assert metrics["entropy"] >= 0.0
