from __future__ import annotations

import argparse
import gzip
import json
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from .episodes import Episode, sessionize


@dataclass
class ExtractedEpisode:
    episode: Episode
    calibration_degree: Counter = field(default_factory=Counter)
    fit_choices: Counter = field(default_factory=Counter)
    pre_activity: Counter = field(default_factory=Counter)
    post_activity: dict[str, Counter] = field(default_factory=lambda: {"auth": Counter(), "dns": Counter()})
    post_contact: dict[str, Counter] = field(default_factory=lambda: {"auth": Counter(), "dns": Counter()})


def _reader(path, modality):
    if modality == "auth":
        names = ["time", "source_user", "user", "source", "destination", "auth_type", "logon_type", "orientation", "status"]
        include = ["time", "source", "destination"]
    else:
        names = ["time", "source", "destination"]
        include = names
    return pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(column_names=names, block_size=64 * 1024 * 1024, use_threads=True),
        convert_options=pacsv.ConvertOptions(
            include_columns=include,
            column_types={"time": pa.int64(), "source": pa.string(), "destination": pa.string()},
        ),
    )


def _values(array, mask):
    return pc.filter(array, pa.array(mask)).to_pylist()


def extract_modality(records, path, modality, calibration_seconds, fit_seconds, post_seconds):
    max_time = max(record.episode.start + post_seconds for record in records)
    scanned = 0
    matched = 0
    for batch in _reader(path, modality):
        time = batch.column("time").to_numpy(zero_copy_only=False)
        if len(time) == 0:
            continue
        if int(time[0]) > max_time:
            break
        scanned += len(time)
        first, last = int(time[0]), int(time[-1])
        source_column = batch.column("source")
        destination_column = batch.column("destination")
        for record in records:
            origin = record.episode.start
            lower = origin - calibration_seconds - fit_seconds
            upper = origin + post_seconds
            if upper < first or lower > last:
                continue
            calibration = (time >= origin - calibration_seconds - fit_seconds) & (time < origin - fit_seconds)
            fit = (time >= origin - fit_seconds) & (time < origin)
            post = (time >= origin) & (time <= upper)
            if modality == "auth":
                if np.any(calibration):
                    source = _values(source_column, calibration)
                    destination = _values(destination_column, calibration)
                    record.calibration_degree.update(item for item in source if item and item.startswith("C"))
                    record.calibration_degree.update(item for item in destination if item and item.startswith("C"))
                    record.pre_activity.update(item for item in source if item and item.startswith("C"))
                    record.pre_activity.update(item for item in destination if item and item.startswith("C"))
                if np.any(fit):
                    source = _values(source_column, fit)
                    destination = _values(destination_column, fit)
                    record.fit_choices.update(item for item in destination if item and item.startswith("C"))
                    record.pre_activity.update(item for item in source if item and item.startswith("C"))
                    record.pre_activity.update(item for item in destination if item and item.startswith("C"))
            if np.any(post):
                source = _values(source_column, post)
                destination = _values(destination_column, post)
                if modality == "auth":
                    record.post_activity[modality].update(item for item in source if item and item.startswith("C"))
                    record.post_activity[modality].update(item for item in destination if item and item.startswith("C"))
                    later = set(record.episode.later_destinations)
                    record.post_contact[modality].update(
                        s for s, d in zip(source, destination) if s and s.startswith("C") and d in later
                    )
                    record.post_contact[modality].update(
                        d for s, d in zip(source, destination) if d and d.startswith("C") and s in later
                    )
                else:
                    record.post_activity[modality].update(item for item in source if item and item.startswith("C"))
                matched += int(np.count_nonzero(post))
    return {"rows_scanned": scanned, "post_window_events": matched}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redteam", type=Path, required=True)
    parser.add_argument("--auth", type=Path, required=True)
    parser.add_argument("--dns", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gap", type=int, default=900)
    parser.add_argument("--calibration", type=int, default=3600)
    parser.add_argument("--fit", type=int, default=3600)
    parser.add_argument("--post", type=int, default=7200)
    args = parser.parse_args()
    episodes = sessionize(args.redteam, args.gap)
    records = [ExtractedEpisode(item) for item in episodes]
    audit = {
        "auth": extract_modality(records, args.auth, "auth", args.calibration, args.fit, args.post),
        "dns": extract_modality(records, args.dns, "dns", args.calibration, args.fit, args.post),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "gap_seconds": args.gap,
        "calibration_seconds": args.calibration,
        "fit_seconds": args.fit,
        "post_seconds": args.post,
        "episodes": len(records),
        "unique_origins": len({item.episode.origin for item in records}),
        "audit": audit,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
