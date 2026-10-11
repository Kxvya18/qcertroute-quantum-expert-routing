"""Hybrid quantum-classical reliability routing for RAG-MoE.

The quantum component is deliberately episode-level. A three-qubit,
two-layer data-reuploading circuit predicts one structural-admission value per
episode and telemetry condition. Endpoint scoring remains classical, so circuit
cost does not scale with the candidate-set size.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .episodes import rolling_folds
from .metrics import ranking, softmax
from .rag_moe import (
    CONDITIONS,
    build_episode,
    load_records,
    mask_for,
    scaler_fit,
    synthetic_population,
    tensors,
    train_temporal,
)


QUANTUM_SEEDS = (7, 19, 42)


def _ry(angle: torch.Tensor) -> torch.Tensor:
    half = angle / 2.0
    c, s = torch.cos(half), torch.sin(half)
    return torch.stack((torch.stack((c, -s)), torch.stack((s, c)))).to(torch.complex64)


def _rz(angle: torch.Tensor) -> torch.Tensor:
    half = angle / 2.0
    zero = torch.zeros((), dtype=torch.complex64, device=angle.device)
    return torch.stack((
        torch.stack((torch.exp(-1j * half), zero)),
        torch.stack((zero, torch.exp(1j * half))),
    ))


def _single_qubit(state: torch.Tensor, gate: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    order = [qubit] + [axis for axis in range(n_qubits) if axis != qubit]
    inverse = np.argsort(order).tolist()
    shaped = state.reshape([2] * n_qubits).permute(order).reshape(2, -1)
    return (gate @ shaped).reshape([2] * n_qubits).permute(inverse).reshape(-1)


def _cnot_inverse(n_qubits: int, control: int, target: int, device) -> torch.Tensor:
    forward = []
    for source in range(2**n_qubits):
        control_bit = (source >> (n_qubits - 1 - control)) & 1
        destination = source
        if control_bit:
            destination ^= 1 << (n_qubits - 1 - target)
        forward.append(destination)
    inverse = np.argsort(forward)
    return torch.tensor(inverse, dtype=torch.long, device=device)


def _expectation_z(state: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    probabilities = state.abs().square()
    signs = torch.tensor(
        [1.0 if ((index >> (n_qubits - 1 - qubit)) & 1) == 0 else -1.0 for index in range(2**n_qubits)],
        dtype=probabilities.dtype,
        device=state.device,
    )
    return torch.sum(probabilities * signs)


class QuantumReliabilityRouter(nn.Module):
    """Data-reuploading router with configurable shallow entanglement."""

    def __init__(self, n_qubits: int = 3, depth: int = 2, entanglement: str = "ring"):
        super().__init__()
        if n_qubits < 2 or depth < 1:
            raise ValueError("The router requires at least two qubits and one layer")
        if entanglement not in {"none", "linear", "ring"}:
            raise ValueError("entanglement must be none, linear, or ring")
        self.n_qubits = n_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.theta = nn.Parameter(torch.empty(depth, n_qubits, 2))
        self.readout = nn.Linear(n_qubits, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.theta, -0.15, 0.15)
        nn.init.constant_(self.readout.weight, 0.20)
        nn.init.constant_(self.readout.bias, -2.0)

    def circuit(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = torch.pi * torch.tanh(features)
        state = torch.zeros(2**self.n_qubits, dtype=torch.complex64, device=features.device)
        state[0] = 1.0 + 0.0j
        for layer in range(self.depth):
            for qubit in range(self.n_qubits):
                first = angles[(2 * qubit + layer) % len(angles)]
                second = angles[(2 * qubit + 1 + layer) % len(angles)]
                state = _single_qubit(state, _ry(first + self.theta[layer, qubit, 0]), qubit, self.n_qubits)
                state = _single_qubit(state, _rz(second + self.theta[layer, qubit, 1]), qubit, self.n_qubits)
            controls = range(self.n_qubits - 1) if self.entanglement == "linear" else range(self.n_qubits)
            if self.entanglement == "none":
                controls = ()
            for control in controls:
                target = (control + 1) % self.n_qubits
                state = state[_cnot_inverse(self.n_qubits, control, target, state.device)]
        local = torch.stack([_expectation_z(state, qubit, self.n_qubits) for qubit in range(self.n_qubits)])
        alpha = torch.clamp(torch.sigmoid(self.readout(local).squeeze()), 1e-6, 1.0 - 1e-6)
        return alpha, state

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            return self.circuit(features)[0]
        return torch.stack([self.circuit(row)[0] for row in features])

    @torch.no_grad()
    def finite_shot(self, features: torch.Tensor, shots: int, rng: np.random.Generator) -> float:
        _, state = self.circuit(features)
        local = []
        for qubit in range(self.n_qubits):
            expectation = float(_expectation_z(state, qubit, self.n_qubits).item())
            plus = rng.binomial(shots, np.clip((1.0 + expectation) / 2.0, 0.0, 1.0))
            local.append(2.0 * plus / shots - 1.0)
        measured = torch.tensor(local, dtype=torch.float32)
        return float(torch.sigmoid(self.readout(measured).squeeze()).item())

    @staticmethod
    def _expand_single(gate: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
        identity = torch.eye(2, dtype=torch.complex64, device=gate.device)
        operator = None
        for index in range(n_qubits):
            factor = gate if index == qubit else identity
            operator = factor if operator is None else torch.kron(operator, factor)
        return operator

    @staticmethod
    def _cnot_matrix(n_qubits: int, control: int, target: int, device) -> torch.Tensor:
        size = 2**n_qubits
        operator = torch.zeros((size, size), dtype=torch.complex64, device=device)
        for source in range(size):
            destination = source
            if (source >> (n_qubits - 1 - control)) & 1:
                destination ^= 1 << (n_qubits - 1 - target)
            operator[destination, source] = 1.0 + 0.0j
        return operator

    @staticmethod
    def _apply_density_gate(rho: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
        return operator @ rho @ operator.conj().transpose(-2, -1)

    @torch.no_grad()
    def noisy_finite_shot(
        self,
        features: torch.Tensor,
        shots: int,
        depolarizing: float,
        readout_error: float,
        rng: np.random.Generator,
    ) -> float:
        """Density-matrix inference with local depolarizing and readout noise."""
        if not 0.0 <= depolarizing <= 1.0 or not 0.0 <= readout_error <= 0.5:
            raise ValueError("invalid noise probability")
        angles = torch.pi * torch.tanh(features)
        size = 2**self.n_qubits
        rho = torch.zeros((size, size), dtype=torch.complex64, device=features.device)
        rho[0, 0] = 1.0 + 0.0j
        paulis = (
            torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=features.device),
            torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=features.device),
            torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=features.device),
        )
        for layer in range(self.depth):
            for qubit in range(self.n_qubits):
                first = angles[(2 * qubit + layer) % len(angles)]
                second = angles[(2 * qubit + 1 + layer) % len(angles)]
                for gate in (_ry(first + self.theta[layer, qubit, 0]), _rz(second + self.theta[layer, qubit, 1])):
                    operator = self._expand_single(gate, qubit, self.n_qubits)
                    rho = self._apply_density_gate(rho, operator)
            controls = range(self.n_qubits - 1) if self.entanglement == "linear" else range(self.n_qubits)
            if self.entanglement == "none":
                controls = ()
            for control in controls:
                operator = self._cnot_matrix(self.n_qubits, control, (control + 1) % self.n_qubits, features.device)
                rho = self._apply_density_gate(rho, operator)
            if depolarizing > 0.0:
                for qubit in range(self.n_qubits):
                    mixed = torch.zeros_like(rho)
                    for pauli in paulis:
                        operator = self._expand_single(pauli, qubit, self.n_qubits)
                        mixed += self._apply_density_gate(rho, operator)
                    rho = (1.0 - depolarizing) * rho + depolarizing * mixed / 3.0
        local = []
        for qubit in range(self.n_qubits):
            z = self._expand_single(paulis[2], qubit, self.n_qubits)
            expectation = float(torch.real(torch.trace(rho @ z)).item()) * (1.0 - 2.0 * readout_error)
            plus = rng.binomial(shots, np.clip((1.0 + expectation) / 2.0, 0.0, 1.0))
            local.append(2.0 * plus / shots - 1.0)
        measured = torch.tensor(local, dtype=torch.float32, device=self.readout.weight.device)
        return float(torch.sigmoid(self.readout(measured).squeeze()).item())


class ClassicalReliabilityRouter(nn.Module):
    """Small nonlinear comparator with a similar parameter budget."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(6, 2), nn.Tanh(), nn.Linear(2, 1), nn.Sigmoid())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class EntangledQuantumRouter(nn.Module):
    """Batched real-amplitude router with local and parity observables.

    Unlike the original episode-wise simulator, this implementation evolves a
    complete batch of statevectors in parallel.  Its RY-CNOT circuit remains a
    valid variational quantum circuit while avoiding complex-valued operations,
    which allows the router to train on Apple's MPS backend.
    """

    def __init__(
        self,
        feature_count: int = 8,
        n_qubits: int = 4,
        depth: int = 2,
        entanglement: str = "ring",
        parity: bool = True,
    ):
        super().__init__()
        if n_qubits < 2 or depth < 1:
            raise ValueError("at least two qubits and one layer are required")
        if entanglement not in {"none", "linear", "ring"}:
            raise ValueError("entanglement must be none, linear, or ring")
        self.feature_count = feature_count
        self.n_qubits = n_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.parity = parity
        self.theta = nn.Parameter(torch.empty(depth, n_qubits, 2))

        signs = []
        for qubit in range(n_qubits):
            signs.append([
                1.0 if ((index >> (n_qubits - 1 - qubit)) & 1) == 0 else -1.0
                for index in range(2**n_qubits)
            ])
        pairs = []
        if parity:
            pair_count = n_qubits if entanglement == "ring" else max(n_qubits - 1, 1)
            for control in range(pair_count):
                target = (control + 1) % n_qubits
                pairs.append(np.asarray(signs[control]) * np.asarray(signs[target]))
        observables = np.asarray([*signs, *pairs], dtype=np.float32)
        self.register_buffer("observable_signs", torch.tensor(observables))
        self.readout = nn.Linear(len(observables), 1)

        controls = range(n_qubits - 1) if entanglement == "linear" else range(n_qubits)
        if entanglement == "none":
            controls = ()
        self._cnot_names = []
        for control in controls:
            target = (control + 1) % n_qubits
            name = f"cnot_{control}_{target}"
            self.register_buffer(name, _cnot_inverse(n_qubits, control, target, "cpu"))
            self._cnot_names.append(name)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.theta, -0.20, 0.20)
        nn.init.normal_(self.readout.weight, mean=0.0, std=0.12)
        nn.init.constant_(self.readout.bias, 0.0)

    def _batched_ry(self, state: torch.Tensor, angle: torch.Tensor, qubit: int) -> torch.Tensor:
        batch = state.shape[0]
        shaped = state.reshape(batch, *([2] * self.n_qubits)).movedim(qubit + 1, -1)
        first, second = shaped[..., 0], shaped[..., 1]
        view = (batch,) + (1,) * (first.ndim - 1)
        cosine = torch.cos(angle / 2.0).reshape(view)
        sine = torch.sin(angle / 2.0).reshape(view)
        rotated = torch.stack((cosine * first - sine * second, sine * first + cosine * second), dim=-1)
        return rotated.movedim(-1, qubit + 1).reshape(batch, -1)

    def state_probabilities(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        batch = features.shape[0]
        state = torch.zeros((batch, 2**self.n_qubits), dtype=features.dtype, device=features.device)
        state[:, 0] = 1.0
        angles = torch.pi * torch.tanh(features)
        for layer in range(self.depth):
            for qubit in range(self.n_qubits):
                offset = 2 * layer * self.n_qubits
                first = angles[:, (offset + qubit) % self.feature_count]
                state = self._batched_ry(state, first + self.theta[layer, qubit, 0], qubit)
            for name in self._cnot_names:
                state = state[:, getattr(self, name)]
            for qubit in range(self.n_qubits):
                second = angles[:, (offset + self.n_qubits + qubit) % self.feature_count]
                state = self._batched_ry(state, second + self.theta[layer, qubit, 1], qubit)
            for name in reversed(self._cnot_names):
                state = state[:, getattr(self, name)]
        return state.square()

    def observables(self, features: torch.Tensor) -> torch.Tensor:
        return self.state_probabilities(features) @ self.observable_signs.transpose(0, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        single = features.ndim == 1
        values = torch.sigmoid(self.readout(self.observables(features)).squeeze(-1))
        return values[0] if single else values

    @torch.no_grad()
    def finite_shot(
        self,
        features: torch.Tensor,
        shots: int,
        rng: np.random.Generator,
        depolarizing: float = 0.0,
        readout_error: float = 0.0,
    ) -> float:
        if shots < 1:
            raise ValueError("shots must be positive")
        exact = self.observables(features).squeeze(0).detach().cpu().numpy()
        local_count = self.n_qubits
        attenuation = np.ones_like(exact)
        local_factor = (1.0 - 4.0 * depolarizing / 3.0) ** self.depth
        attenuation[:local_count] *= local_factor * (1.0 - 2.0 * readout_error)
        attenuation[local_count:] *= local_factor**2 * (1.0 - 2.0 * readout_error) ** 2
        expectation = np.clip(exact * attenuation, -1.0, 1.0)
        plus = rng.binomial(shots, (1.0 + expectation) / 2.0)
        measured = torch.tensor(2.0 * plus / shots - 1.0, dtype=self.readout.weight.dtype,
                                device=self.readout.weight.device)
        return float(torch.sigmoid(self.readout(measured).squeeze()).item())


class QuantumReservoirRouter(nn.Module):
    """Quantum extreme-learning router with fixed data-reuploading reservoirs.

    Each reservoir maps the eight routing statistics to local Z and adjacent
    ZZ observables. Only the final linear measurement readout is trained. This
    avoids variational landscape instability while retaining a quantum feature
    map whose correlation observables can be ablated directly.
    """

    def __init__(
        self,
        feature_count: int = 8,
        reservoirs: int = 4,
        n_qubits: int = 4,
        depth: int = 2,
        parity: bool = True,
        reservoir_seed: int = 1729,
    ):
        super().__init__()
        if reservoirs < 1:
            raise ValueError("at least one quantum reservoir is required")
        self.n_qubits = n_qubits
        self.depth = depth
        self.parity = parity
        self.quantum_reservoirs = nn.ModuleList()
        generator = torch.Generator(device="cpu").manual_seed(reservoir_seed)
        for _ in range(reservoirs):
            circuit = EntangledQuantumRouter(
                feature_count=feature_count,
                n_qubits=n_qubits,
                depth=depth,
                entanglement="ring",
                parity=parity,
            )
            with torch.no_grad():
                circuit.theta.copy_(0.85 * torch.randn(circuit.theta.shape, generator=generator))
            for parameter in circuit.parameters():
                parameter.requires_grad_(False)
            self.quantum_reservoirs.append(circuit)
        observable_count = n_qubits * (2 if parity else 1)
        self.readout = nn.Linear(reservoirs * observable_count, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.readout.weight, mean=0.0, std=0.08)
        nn.init.constant_(self.readout.bias, 0.0)

    def quantum_features(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([circuit.observables(features) for circuit in self.quantum_reservoirs], dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        single = features.ndim == 1
        values = torch.sigmoid(self.readout(self.quantum_features(features)).squeeze(-1))
        return values[0] if single else values

    @torch.no_grad()
    def finite_shot(
        self,
        features: torch.Tensor,
        shots: int,
        rng: np.random.Generator,
        depolarizing: float = 0.0,
        readout_error: float = 0.0,
    ) -> float:
        if shots < 1:
            raise ValueError("shots must be positive")
        sampled = []
        local_factor = (1.0 - 4.0 * depolarizing / 3.0) ** self.depth
        for circuit in self.quantum_reservoirs:
            exact = circuit.observables(features).squeeze(0).detach().cpu().numpy()
            attenuation = np.full_like(exact, local_factor * (1.0 - 2.0 * readout_error))
            if self.parity:
                attenuation[self.n_qubits:] = local_factor**2 * (1.0 - 2.0 * readout_error) ** 2
            expectation = np.clip(exact * attenuation, -1.0, 1.0)
            plus = rng.binomial(shots, (1.0 + expectation) / 2.0)
            sampled.extend(2.0 * plus / shots - 1.0)
        measured = torch.tensor(sampled, dtype=self.readout.weight.dtype, device=self.readout.weight.device)
        return float(torch.sigmoid(self.readout(measured).squeeze()).item())


class InteractionClassicalRouter(nn.Module):
    """Parameter-matched nonlinear comparator for the entangled router."""

    def __init__(self, feature_count: int = 8, width: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, width),
            nn.Tanh(),
            nn.Linear(width, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass
class RouterBundle:
    temporal: nn.Module
    quantum: QuantumReliabilityRouter
    classical: ClassicalReliabilityRouter
    feature_mean: torch.Tensor
    feature_std: torch.Tensor


def router_features(temporal: np.ndarray, structural: np.ndarray, data, observed_fraction: float) -> np.ndarray:
    probability = softmax(temporal)
    entropy = -float(np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0))))
    entropy /= max(float(np.log(max(len(temporal), 2))), 1e-12)
    temporal_rank = np.argsort(np.argsort(temporal))
    structural_rank = np.argsort(np.argsort(structural))
    disagreement = float(np.mean(np.abs(temporal_rank - structural_rank)) / max(len(temporal) - 1, 1))
    return np.asarray((
        data.rho,
        entropy,
        disagreement,
        1.0 - observed_fraction,
        np.log1p(data.se),
        data.gini,
    ), dtype=np.float32)


def _episode_feature(model, data, condition, mean, std, seed):
    mask = mask_for(data, condition, np.random.default_rng(seed + data.identifier))
    x, structural, _, _ = tensors(data, mask, mean, std)
    model.eval()
    with torch.no_grad():
        temporal = model.temporal(x).squeeze(-1).numpy()
    return temporal, router_features(temporal, data.structural, data, float(mask.mean()))


def _train_small_router(model, features, targets, seed, epochs=40):
    torch.manual_seed(seed)
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    else:
        for module in model.modules():
            if module is not model and hasattr(module, "reset_parameters"):
                module.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025, weight_decay=1e-4)
    x = features.detach().clone() if isinstance(features, torch.Tensor) else torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        prediction = model(x)
        loss = F.binary_cross_entropy(prediction, y) + 0.02 * prediction.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
    return model


def fast_gate_targets(train, mean, std, seed, temporal_device="cpu"):
    """Cross-fitted targets with a compact conference-scale training budget."""
    labels = {}
    grid = np.asarray((0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0))
    ordered = list(train)
    halves = (ordered[::2], ordered[1::2])
    for target, source in ((halves[0], halves[1]), (halves[1], halves[0])):
        if not source:
            source = target
        model = train_temporal(source, mean, std, seed + len(labels) + 101, epochs=8, device=temporal_device)
        model.eval()
        for data in target:
            for offset, condition in enumerate(CONDITIONS):
                mask = mask_for(data, condition, np.random.default_rng(seed + data.identifier + 1009 * offset))
                x, _, _, _ = tensors(data, mask, mean, std)
                with torch.no_grad():
                    temporal = model.temporal(x).squeeze(-1).numpy()
                utility = np.asarray([
                    ranking(softmax(temporal + admission * data.rho * data.structural), data.truth)["mrr"]
                    for admission in grid
                ])
                labels[(data.identifier, condition)] = float(grid[int(np.argmax(utility))])
    return labels


def prepare_router_training(train, mean, std, seed, temporal_device="cpu"):
    """Fit the temporal expert and return leakage-controlled router data."""
    temporal = train_temporal(train, mean, std, seed, epochs=12, device=temporal_device)
    targets = fast_gate_targets(train, mean, std, seed, temporal_device=temporal_device)
    raw_features, labels = [], []
    for data in train:
        for offset, condition in enumerate(CONDITIONS):
            _, feature = _episode_feature(temporal, data, condition, mean, std, seed + 1009 * offset)
            raw_features.append(feature)
            labels.append(targets[(data.identifier, condition)])
    raw_features = np.asarray(raw_features, dtype=np.float32)
    feature_mean = torch.tensor(raw_features.mean(axis=0), dtype=torch.float32)
    feature_std = torch.tensor(raw_features.std(axis=0).clip(1e-5), dtype=torch.float32)
    scaled = (torch.tensor(raw_features) - feature_mean) / feature_std
    return temporal, scaled, labels, feature_mean, feature_std


def fit_routers(train, mean, std, seed, temporal_device="cpu", quantum_kwargs=None) -> RouterBundle:
    temporal, scaled, labels, feature_mean, feature_std = prepare_router_training(
        train, mean, std, seed, temporal_device=temporal_device
    )
    quantum = _train_small_router(QuantumReliabilityRouter(**(quantum_kwargs or {})), scaled, labels, seed + 301)
    classical = _train_small_router(ClassicalReliabilityRouter(), scaled, labels, seed + 601)
    return RouterBundle(temporal, quantum, classical, feature_mean, feature_std)


def evaluate_bundle(bundle, data, condition, mean, std, seed, shots=256):
    temporal, feature = _episode_feature(bundle.temporal, data, condition, mean, std, seed)
    feature_tensor = (torch.tensor(feature) - bundle.feature_mean) / bundle.feature_std
    bundle.quantum.eval()
    bundle.classical.eval()
    with torch.no_grad():
        quantum_alpha = float(bundle.quantum(feature_tensor).item())
        classical_alpha = float(bundle.classical(feature_tensor).item())
    shot_alpha = bundle.quantum.finite_shot(feature_tensor, shots, np.random.default_rng(seed + data.identifier + 777))
    scores = {
        "temporal": temporal,
        "fixed_curvature": temporal + data.rho * data.structural,
        "classical_router": temporal + classical_alpha * data.rho * data.structural,
        "quantum_router": temporal + quantum_alpha * data.rho * data.structural,
        "quantum_256shot": temporal + shot_alpha * data.rho * data.structural,
    }
    alphas = {
        "temporal": 0.0,
        "fixed_curvature": 1.0,
        "classical_router": classical_alpha,
        "quantum_router": quantum_alpha,
        "quantum_256shot": shot_alpha,
    }
    return scores, alphas, feature


def _append_rows(rows, scores, alphas, data, condition, seed, extra):
    for model, score in scores.items():
        row = {
            **extra,
            "seed": seed,
            "episode": data.identifier,
            "condition": condition,
            "model": model,
            "alpha": alphas[model],
            "rho": data.rho,
        }
        row.update(ranking(softmax(score), data.truth))
        rows.append(row)


def evaluate_controlled(output: Path):
    rows = []
    identifier = 0
    for beta in (0.0, 1.0, 2.0):
        population = synthetic_population(beta, 2, identifier, 30)
        identifier += 30
        train, test = population[:20], population[20:]
        mean, std = scaler_fit(train)
        for seed in QUANTUM_SEEDS:
            bundle = fit_routers(train, mean, std, seed)
            for data in test:
                for condition in CONDITIONS:
                    scores, alphas, _ = evaluate_bundle(bundle, data, condition, mean, std, seed)
                    _append_rows(rows, scores, alphas, data, condition, seed, {"beta": beta, "m": 2})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "qrag_controlled_results.csv", index=False)
    return frame


def evaluate_lanl(records, output: Path, cap=20000):
    rows = []
    episodes = [record.episode for record in records]
    for fold, train_ep, test_ep in rolling_folds(episodes):
        train_ids = {episode.identifier for episode in train_ep}
        raw = [build_episode(record, cap, 1.0) for record in records if record.episode.identifier in train_ids]
        raw = [item for item in raw if item is not None]
        scale = float(np.median([1 / max(item.se * item.se, 1e-12) for item in raw])) if raw else 1.0
        by_id = {record.episode.identifier: build_episode(record, cap, scale) for record in records}
        train = [by_id[episode.identifier] for episode in train_ep if by_id[episode.identifier] is not None]
        test = [by_id[episode.identifier] for episode in test_ep if by_id[episode.identifier] is not None]
        if len(train) < 3 or not test:
            continue
        mean, std = scaler_fit(train)
        for seed in QUANTUM_SEEDS:
            bundle = fit_routers(train, mean, std, seed)
            for data in test:
                for condition in CONDITIONS:
                    scores, alphas, _ = evaluate_bundle(bundle, data, condition, mean, std, seed)
                    _append_rows(rows, scores, alphas, data, condition, seed, {"fold": fold})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "qrag_lanl_results.csv", index=False)
    return frame


def summarize(frame, keys):
    return frame.groupby([*keys, "condition", "model"]).agg(
        episodes=("episode", "nunique"),
        seeds=("seed", "nunique"),
        mrr=("mrr", "mean"),
        hit5=("hit5", "mean"),
        hit10=("hit10", "mean"),
        nll=("nll", "mean"),
        alpha=("alpha", "mean"),
    ).reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-controlled", action="store_true")
    parser.add_argument("--skip-lanl", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    controlled = pd.DataFrame() if args.skip_controlled else evaluate_controlled(args.output)
    lanl = pd.DataFrame() if args.skip_lanl else evaluate_lanl(load_records(args.input), args.output)
    if not controlled.empty:
        summarize(controlled, ["beta", "m"]).to_csv(args.output / "qrag_controlled_summary.csv", index=False)
    if not lanl.empty:
        summarize(lanl, []).to_csv(args.output / "qrag_lanl_summary.csv", index=False)
    print(json.dumps({
        "controlled_rows": len(controlled),
        "lanl_rows": len(lanl),
        "quantum_parameters": 16,
        "classical_router_parameters": 17,
        "statevector_simulation": True,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
