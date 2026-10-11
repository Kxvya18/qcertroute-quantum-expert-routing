from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Episode:
    identifier: int
    start: int
    end: int
    origin: str
    user: str
    lateral_source: str
    later_destinations: tuple[str, ...]
    count: int


def load_redteam(path: str | Path):
    seen = set()
    result = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            item = (int(row[0]), row[1], row[2], row[3])
            if item not in seen:
                result.append(item)
                seen.add(item)
    return sorted(result)


def sessionize(path: str | Path, gap_seconds: int = 900, minimum_events: int = 2):
    rows = load_redteam(path)
    groups = []
    for row in rows:
        if not groups or row[0] - groups[-1][-1][0] > gap_seconds:
            groups.append([row])
        else:
            groups[-1].append(row)
    episodes = []
    for group in groups:
        destinations = tuple(dict.fromkeys(event[3] for event in group))
        if len(group) < minimum_events or len(destinations) < 2:
            continue
        first = group[0]
        episodes.append(
            Episode(
                identifier=len(episodes),
                start=first[0],
                end=group[-1][0],
                origin=first[3],
                user=first[1],
                lateral_source=first[2],
                later_destinations=destinations[1:],
                count=len(group),
            )
        )
    return episodes


def rolling_folds(episodes, train_sizes=(20, 25, 30, 35), test_size=5):
    folds = []
    for fold_id, train_end in enumerate(train_sizes):
        test = episodes[train_end : train_end + test_size]
        if len(test) == test_size:
            folds.append((fold_id, episodes[:train_end], test))
    return folds
