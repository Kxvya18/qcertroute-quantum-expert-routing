"""Chronological, similar-capacity router comparison on real email episodes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .email_qrouter import CONDITIONS, load_episodes
from .qrag_moe import EntangledQuantumRouter, InteractionClassicalRouter


def _metrics(temporal: np.ndarray, graph: np.ndarray, truth: np.ndarray,
             weight: np.ndarray) -> dict[str, float]:
    n = len(truth)
    fused = (1.0 - weight[:, None]) * temporal + weight[:, None] * graph
    actual = fused[np.arange(n), truth]
    rank = 1 + np.sum(fused > actual[:, None], axis=1)
    return {
        "mrr": float(np.mean(1.0 / rank)),
        "hit10": float(np.mean(rank <= 10)),
        "nll": float(-np.mean(np.log(np.maximum(actual, 1e-12)))),
        "mean_gate": float(np.mean(weight)),
    }


def _fit(model: torch.nn.Module, features: np.ndarray, temporal_true: np.ndarray,
         graph_true: np.ndarray, train: np.ndarray, validation: np.ndarray,
         epochs: int, seed: int) -> tuple[torch.nn.Module, int, float]:
    torch.manual_seed(seed)
    model.reset_parameters() if hasattr(model, "reset_parameters") else None
    optimizer = torch.optim.Adam(model.parameters(), lr=0.012, weight_decay=1e-4)
    x = torch.tensor(features, dtype=torch.float32)
    t = torch.tensor(temporal_true, dtype=torch.float32)
    g = torch.tensor(graph_true, dtype=torch.float32)
    best_state, best_epoch, best_loss, stale = None, 0, float("inf"), 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train)
        for indices in np.array_split(order, max(1, int(np.ceil(len(order) / 512)))):
            gate = model(x[indices])
            likelihood = (1.0 - gate) * t[indices] + gate * g[indices]
            loss = -torch.log(torch.clamp(likelihood, min=1e-10)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            gate = model(x[validation])
            likelihood = (1.0 - gate) * t[validation] + gate * g[validation]
            current = float(-torch.log(torch.clamp(likelihood, min=1e-10)).mean())
        if current < best_loss - 1e-5:
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            best_loss, best_epoch, stale = current, epoch, 0
        else:
            stale += 1
        if stale >= 12:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("outputs/email_qrouter_v1/episodes.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/email_qrouter_v1/train"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = load_episodes(args.episodes)
    n, conditions, nodes = data.temporal.shape
    days = np.unique(data.day)
    first_cut = days[int(len(days) * 0.6)]
    second_cut = days[int(len(days) * 0.8)]
    episode_split = np.where(data.day < first_cut, "train",
                             np.where(data.day < second_cut, "validation", "test"))
    split = np.repeat(episode_split, conditions)
    x = data.features.reshape(n * conditions, -1)
    train = np.flatnonzero(split == "train")
    validation = np.flatnonzero(split == "validation")
    test = np.flatnonzero(split == "test")
    mean, scale = x[train].mean(axis=0), x[train].std(axis=0)
    scale = np.maximum(scale, 0.1)
    x = np.clip((x - mean) / scale, -5.0, 5.0).astype(np.float32)
    temporal = data.temporal.reshape(n * conditions, nodes)
    graph = data.graph.reshape(n * conditions, nodes)
    truth = np.repeat(data.truth, conditions)
    true_t = temporal[np.arange(len(truth)), truth]
    true_g = graph[np.arange(len(truth)), truth]
    args.output.mkdir(parents=True, exist_ok=True)
    validation_grid = []
    for weight in np.linspace(0.0, 1.0, 21):
        score = _metrics(temporal[validation], graph[validation], truth[validation],
                         np.full(len(validation), weight))
        validation_grid.append((score["nll"], float(weight)))
    fixed_weight = min(validation_grid)[1]
    weights = {"temporal": np.zeros(len(truth)), "graph": np.ones(len(truth)),
               "fixed_mix": np.full(len(truth), fixed_weight)}
    fit_summary = []
    for seed in range(args.seeds):
        for kind in ("quantum", "classical"):
            torch.manual_seed(733 + seed)
            model = (EntangledQuantumRouter(feature_count=x.shape[1], n_qubits=4, depth=2)
                     if kind == "quantum" else
                     InteractionClassicalRouter(feature_count=x.shape[1], width=2))
            model, selected_epoch, validation_loss = _fit(
                model, x, true_t, true_g, train, validation, args.epochs, 733 + seed)
            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(x, dtype=torch.float32)).numpy()
            weights[f"{kind}_seed{seed}"] = pred
            torch.save(model.state_dict(), args.output / f"{kind}_seed{seed}.pt")
            fit_summary.append({"model": kind, "seed": seed, "epoch": selected_epoch,
                                "validation_nll": validation_loss,
                                "parameters": sum(p.numel() for p in model.parameters())})
    metric_rows = []
    for name, gate in weights.items():
        for condition_id, condition in enumerate(CONDITIONS):
            indices = test[test % conditions == condition_id]
            score = _metrics(temporal[indices], graph[indices], truth[indices], gate[indices])
            metric_rows.append({"model": name, "condition": condition,
                                "episodes": len(indices), **score})
    with (args.output / "test_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with (args.output / "training.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fit_summary[0]))
        writer.writeheader()
        writer.writerows(fit_summary)
    np.savez_compressed(args.output / "predictions.npz", weights=np.stack(list(weights.values())),
                        weight_names=np.asarray(list(weights)), day=data.day, truth=data.truth,
                        split=episode_split, feature_mean=mean, feature_scale=scale)
    protocol = {"dataset": data.audit["dataset"], "episodes": n,
                "split_days": [int(first_cut), int(second_cut)],
                "split_episodes": {key: int(np.sum(episode_split == key)) for key in ("train", "validation", "test")},
                "fixed_mix_weight": fixed_weight, "seeds": args.seeds,
                "epochs_cap": args.epochs, "batch_size": 512,
                "optimizer": "Adam", "learning_rate": 0.012,
                "validation_criterion": "minimum negative log likelihood with patience 12"}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2))
    print("\nTest MRR by condition")
    for row in metric_rows:
        print(row["model"], row["condition"], round(row["mrr"], 4),
              "NLL", round(row["nll"], 4))


if __name__ == "__main__":
    main()
