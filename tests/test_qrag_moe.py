import numpy as np
import torch

from ba3active.lanl_real_qrag import RealEpisode, load_episode_cache, save_episode_cache
from ba3active.qrag_moe import (
    ClassicalReliabilityRouter,
    EntangledQuantumRouter,
    QuantumReservoirRouter,
    QuantumReliabilityRouter,
    _train_small_router,
)
from ba3active.rag_moe import EpisodeData
from ba3active.interdependent_qrag import RouterRecords, regret_supervision
from ba3active.interdependent_checkpoint import audit_checkpoint


def test_quantum_router_has_audited_circuit_budget():
    router = QuantumReliabilityRouter()
    assert sum(parameter.numel() for parameter in router.parameters()) == 16
    assert router.n_qubits == 3
    assert router.depth == 2


def test_quantum_state_is_normalized_and_admission_is_bounded():
    torch.manual_seed(3)
    router = QuantumReliabilityRouter()
    alpha, state = router.circuit(torch.randn(6))
    assert torch.isclose(state.abs().square().sum(), torch.tensor(1.0), atol=1e-5)
    assert 0.0 <= float(alpha.detach()) <= 1.0


def test_quantum_router_is_differentiable():
    torch.manual_seed(5)
    router = QuantumReliabilityRouter()
    loss = router(torch.randn(4, 6)).mean()
    loss.backward()
    assert router.theta.grad is not None
    assert torch.isfinite(router.theta.grad).all()
    assert float(router.theta.grad.abs().sum()) > 0.0


def test_finite_shot_router_returns_valid_frequency():
    router = QuantumReliabilityRouter()
    value = router.finite_shot(torch.zeros(6), 256, np.random.default_rng(9))
    assert 0.0 <= value <= 1.0


def test_classical_comparator_is_small_but_nonlinear():
    router = ClassicalReliabilityRouter()
    assert sum(parameter.numel() for parameter in router.parameters()) == 17
    output = router(torch.randn(5, 6))
    assert output.shape == (5,)
    assert torch.all((0.0 <= output) & (output <= 1.0))


def test_configurable_circuit_families_are_normalized():
    for qubits, depth, entanglement in ((2, 1, "none"), (3, 2, "linear"), (4, 3, "ring")):
        router = QuantumReliabilityRouter(qubits, depth, entanglement)
        alpha, state = router.circuit(torch.randn(6))
        assert torch.isclose(state.abs().square().sum(), torch.tensor(1.0), atol=1e-5)
        assert 0.0 <= float(alpha.detach()) <= 1.0


def test_density_noise_path_is_physical_and_bounded():
    router = QuantumReliabilityRouter()
    rng = np.random.default_rng(11)
    value = router.noisy_finite_shot(torch.zeros(6), 256, 0.02, 0.03, rng)
    assert 0.0 <= value <= 1.0


def test_router_training_seed_controls_initialization_and_solution():
    features = torch.tensor(np.random.default_rng(2).normal(size=(10, 6)), dtype=torch.float32)
    labels = np.linspace(0.0, 1.0, 10)
    first = _train_small_router(QuantumReliabilityRouter(), features, labels, seed=17, epochs=3)
    second = _train_small_router(QuantumReliabilityRouter(), features, labels, seed=17, epochs=3)
    with torch.no_grad():
        assert torch.allclose(first(features), second(features), atol=1e-7)


def test_real_episode_cache_is_module_independent(tmp_path):
    data = EpisodeData(
        identifier=9,
        x=np.ones((3, 5), dtype=np.float32),
        structural=np.asarray((-1.0, 0.0, 1.0), dtype=np.float32),
        degree=np.asarray((1.0, 2.0, 3.0), dtype=np.float32),
        truth=2,
        rho=0.4,
        se=0.2,
        gini=0.3,
    )
    episode = RealEpisode(data, 600, "C1", "C3", 1.1, 25.0, 3)
    path = tmp_path / "episodes.pkl"
    save_episode_cache(path, [episode], {"episodes": 1})
    loaded, audit = load_episode_cache(path)
    assert audit == {"episodes": 1}
    assert loaded[0].source == "C1"
    assert loaded[0].target == "C3"
    assert np.array_equal(loaded[0].data.x, data.x)


def test_entangled_router_batches_states_and_parity_observables():
    router = EntangledQuantumRouter(n_qubits=4, depth=2, entanglement="ring", parity=True)
    features = torch.randn(11, 8)
    observables = router.observables(features)
    admission = router(features)
    assert observables.shape == (11, 8)
    assert admission.shape == (11,)
    assert torch.all((observables >= -1.00001) & (observables <= 1.00001))
    assert torch.all((admission >= 0.0) & (admission <= 1.0))


def test_entangled_router_is_differentiable_and_shot_bounded():
    router = EntangledQuantumRouter()
    features = torch.randn(7, 8)
    router(features).mean().backward()
    assert router.theta.grad is not None
    assert float(router.theta.grad.abs().sum()) > 0.0
    value = router.finite_shot(features[0], 64, np.random.default_rng(21), 0.01, 0.02)
    assert 0.0 <= value <= 1.0


def test_quantum_reservoir_trains_only_readout_and_supports_shots():
    router = QuantumReservoirRouter(reservoirs=2, n_qubits=4, depth=2, parity=True)
    features = torch.randn(6, 8)
    output = router(features)
    output.mean().backward()
    assert output.shape == (6,)
    assert router.readout.weight.grad is not None
    assert all(not parameter.requires_grad for circuit in router.quantum_reservoirs
               for parameter in circuit.parameters())
    value = router.finite_shot(features[0], 64, np.random.default_rng(12), 0.01, 0.02)
    assert 0.0 <= value <= 1.0


def test_regret_supervision_routes_to_the_better_ranked_expert():
    records = RouterRecords(
        features=np.zeros((3, 8), dtype=np.float32),
        temporal_truth=np.ones(3, dtype=np.float32),
        graph_truth=np.ones(3, dtype=np.float32),
        temporal_square=np.ones(3, dtype=np.float32),
        graph_square=np.ones(3, dtype=np.float32),
        cross_product=np.ones(3, dtype=np.float32),
        identifiers=np.arange(3),
        condition_ids=np.arange(3),
        temporal_rank=np.asarray((1, 10, 4), dtype=np.float32),
        graph_rank=np.asarray((10, 1, 4), dtype=np.float32),
    )
    target, weight = regret_supervision(records)
    assert target[0] < 0.5 < target[1]
    assert np.isclose(target[2], 0.5)
    assert weight[2] < weight[0]


def test_checkpoint_rejects_a_collapsed_router():
    rows = []
    for condition in ("complete", "random50", "hub50", "auth_only", "dns_only"):
        for model, mrr, alpha in (
            ("quantum_moe_selected", 0.40, 0.92),
            ("temporal", 0.30, 0.0),
            ("graph_only", 0.40, 1.0),
            ("classical_moe", 0.40, 0.90),
        ):
            rows.append({"condition": condition, "model": model, "mrr": mrr,
                         "alpha": alpha, "episode": 1})
    assert audit_checkpoint(__import__("pandas").DataFrame(rows))["decision"] == "STOP"
