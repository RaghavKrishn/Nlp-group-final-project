from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable


DIALECTS = ("natural", "latex", "prefix", "postfix")

STRUCTURE_TARGETS_BY_FAMILY = {
    "Heat": {
        "time_order": 1,
        "has_first_spatial": 0,
        "has_second_spatial": 1,
        "nonlinear": 0,
        "spatial_vars": 1,
    },
    "Wave": {
        "time_order": 2,
        "has_first_spatial": 0,
        "has_second_spatial": 1,
        "nonlinear": 0,
        "spatial_vars": 1,
    },
    "Burgers": {
        "time_order": 1,
        "has_first_spatial": 1,
        "has_second_spatial": 1,
        "nonlinear": 1,
        "spatial_vars": 1,
    },
    "Laplace": {
        "time_order": 0,
        "has_first_spatial": 0,
        "has_second_spatial": 1,
        "nonlinear": 0,
        "spatial_vars": 2,
    },
    "Advection": {
        "time_order": 1,
        "has_first_spatial": 1,
        "has_second_spatial": 0,
        "nonlinear": 0,
        "spatial_vars": 1,
    },
    "KleinGordon": {
        "time_order": 2,
        "has_first_spatial": 0,
        "has_second_spatial": 1,
        "nonlinear": 0,
        "spatial_vars": 1,
    },
    "ReactionDiffusion": {
        "time_order": 1,
        "has_first_spatial": 0,
        "has_second_spatial": 1,
        "nonlinear": 1,
        "spatial_vars": 1,
    },
    "Beam": {
        "time_order": 2,
        "has_first_spatial": 0,
        "has_second_spatial": 0,
        "nonlinear": 0,
        "spatial_vars": 1,
    },
}

REASONING_KEYWORDS = {
    "Heat": ["first-order time", "second-order spatial", "diffus"],
    "Wave": ["second-order time", "wave", "speed"],
    "Burgers": ["nonlinear", "convect", "viscosit", "u*u_x"],
    "Laplace": ["steady", "no time", "equilibrium", "u_xx"],
    "Advection": ["transport", "advect", "first-order"],
}


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_target(instance: dict) -> str:
    ops = ", ".join(instance["labels"]["operators"])
    reasoning = instance["labels"]["reasoning"]
    family = instance["labels"]["behavioral"]
    return f"family: {family} | operators: {ops} | reasoning: {reasoning}"


def parse_prediction(prediction: str) -> dict:
    parsed = {"family": None, "operators": [], "reasoning": ""}
    for part in prediction.split("|"):
        clean = part.strip()
        lower = clean.lower()
        if lower.startswith("family:"):
            parsed["family"] = clean.split(":", 1)[1].strip()
        elif lower.startswith("operators:"):
            ops = clean.split(":", 1)[1].strip()
            parsed["operators"] = [item.strip() for item in ops.split(",") if item.strip()]
        elif lower.startswith("reasoning:"):
            parsed["reasoning"] = clean.split(":", 1)[1].strip()
    return parsed


def is_reasoning_correct(family: str, reasoning_text: str) -> bool:
    keywords = REASONING_KEYWORDS.get(family, [])
    lowered = reasoning_text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def equation_signature(instance: dict) -> str:
    payload = {
        "family": instance["family"],
        "coefficients": instance.get("coefficients", {}),
    }
    return json.dumps(payload, sort_keys=True)


def build_operator_vocab(data: list[dict]) -> list[str]:
    operators = sorted(
        {
            operator.lower()
            for instance in data
            for operator in instance["labels"].get("operators", [])
        }
    )
    return operators


def derive_structure_targets(instance: dict) -> dict[str, int]:
    family = instance["labels"]["behavioral"]
    if family not in STRUCTURE_TARGETS_BY_FAMILY:
        raise KeyError(
            f"No structural-target mapping found for family '{family}'. "
            "Update STRUCTURE_TARGETS_BY_FAMILY before using multitask training."
        )
    return dict(STRUCTURE_TARGETS_BY_FAMILY[family])


def split_counts(total: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    if total <= 0:
        return (0, 0, 0)

    if not math.isclose(sum(fractions), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Split fractions must sum to 1.0, got {fractions}.")

    raw = [total * fraction for fraction in fractions]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total - sum(counts)

    order = sorted(range(len(raw)), key=lambda idx: (raw[idx] - counts[idx]), reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1

    active = [idx for idx, fraction in enumerate(fractions) if fraction > 0]
    if total < len(active):
        raise ValueError(
            f"Cannot allocate {len(active)} non-empty splits from only {total} items."
        )

    for idx in active:
        if counts[idx] > 0:
            continue
        donor = max(
            (candidate for candidate in active if counts[candidate] > 1),
            key=lambda candidate: counts[candidate],
            default=None,
        )
        if donor is None:
            raise ValueError(
                f"Unable to guarantee a non-empty split for fractions={fractions} and total={total}."
            )
        counts[donor] -= 1
        counts[idx] += 1

    return tuple(counts)


def stratified_instance_split(
    data: list[dict],
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for instance in data:
        by_family[instance["family"]].append(instance)

    rng = random.Random(seed)
    train_data: list[dict] = []
    val_data: list[dict] = []
    test_data: list[dict] = []

    for family in sorted(by_family):
        items = list(by_family[family])
        rng.shuffle(items)
        n_train, n_val, n_test = split_counts(
            len(items), (train_fraction, val_fraction, test_fraction)
        )
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train : n_train + n_val])
        test_data.extend(items[n_train + n_val : n_train + n_val + n_test])

    return train_data, val_data, test_data


def stratified_group_split(
    data: list[dict],
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    signature_fn: Callable[[dict], str],
) -> tuple[list[dict], list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    group_family: dict[str, str] = {}

    for instance in data:
        signature = signature_fn(instance)
        groups[signature].append(instance)
        group_family.setdefault(signature, instance["family"])

    family_groups: dict[str, list[str]] = defaultdict(list)
    for signature, family in group_family.items():
        family_groups[family].append(signature)

    required_splits = sum(
        1 for fraction in (train_fraction, val_fraction, test_fraction) if fraction > 0
    )
    too_small = {
        family: len(signatures)
        for family, signatures in family_groups.items()
        if len(signatures) < required_splits
    }
    if too_small:
        details = ", ".join(
            f"{family}={count} groups" for family, count in sorted(too_small.items())
        )
        raise ValueError(
            "Equation-level split is not possible for every family in this dataset: "
            f"{details}. This usually means the dataset is too repetitive for a leakage-free split."
        )

    rng = random.Random(seed)
    train_data: list[dict] = []
    val_data: list[dict] = []
    test_data: list[dict] = []

    for family in sorted(family_groups):
        signatures = list(family_groups[family])
        rng.shuffle(signatures)
        n_train, n_val, n_test = split_counts(
            len(signatures), (train_fraction, val_fraction, test_fraction)
        )

        train_keys = signatures[:n_train]
        val_keys = signatures[n_train : n_train + n_val]
        test_keys = signatures[n_train + n_val : n_train + n_val + n_test]

        for key in train_keys:
            train_data.extend(groups[key])
        for key in val_keys:
            val_data.extend(groups[key])
        for key in test_keys:
            test_data.extend(groups[key])

    return train_data, val_data, test_data


def family_counts(data: list[dict]) -> dict[str, int]:
    return dict(Counter(instance["family"] for instance in data))


def exact_input_overlap(train_data: list[dict], test_data: list[dict], dialect: str) -> dict:
    train_inputs = {instance["dialects"][dialect] for instance in train_data}
    test_inputs = [instance["dialects"][dialect] for instance in test_data]
    overlapping = sum(1 for value in test_inputs if value in train_inputs)
    overlapping_unique = len({value for value in test_inputs if value in train_inputs})
    return {
        "train_unique": len(train_inputs),
        "test_unique": len(set(test_inputs)),
        "overlap_unique": overlapping_unique,
        "overlap_examples": overlapping,
        "overlap_rate": overlapping / len(test_inputs) if test_inputs else 0.0,
    }


def equation_overlap(train_data: list[dict], test_data: list[dict]) -> dict:
    train_signatures = {equation_signature(instance) for instance in train_data}
    test_signatures = [equation_signature(instance) for instance in test_data]
    overlapping = sum(1 for value in test_signatures if value in train_signatures)
    overlapping_unique = len({value for value in test_signatures if value in train_signatures})
    return {
        "train_unique": len(train_signatures),
        "test_unique": len(set(test_signatures)),
        "overlap_unique": overlapping_unique,
        "overlap_examples": overlapping,
        "overlap_rate": overlapping / len(test_signatures) if test_signatures else 0.0,
    }


def load_accuracy_matrix(path: str | Path) -> dict[str, dict[str, float]]:
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {path}.")

    header = rows[0][1:]
    matrix: dict[str, dict[str, float]] = {}
    for row in rows[1:]:
        if not row:
            continue
        matrix[row[0]] = {column: float(value) for column, value in zip(header, row[1:])}
    return matrix


def summarize_matrix(matrix: dict[str, dict[str, float]]) -> dict:
    dialects = list(matrix)
    diagonal = [matrix[dialect][dialect] for dialect in dialects]
    off_diagonal = [
        matrix[row][column]
        for row in dialects
        for column in dialects
        if row != column
    ]
    off_diagonal_means = {
        row: sum(matrix[row][column] for column in dialects if column != row)
        / max(len(dialects) - 1, 1)
        for row in dialects
    }
    best = max(off_diagonal_means, key=off_diagonal_means.get)
    worst = min(off_diagonal_means, key=off_diagonal_means.get)
    return {
        "dialects": dialects,
        "diagonal_mean": sum(diagonal) / len(diagonal) if diagonal else 0.0,
        "off_diagonal_mean": sum(off_diagonal) / len(off_diagonal) if off_diagonal else 0.0,
        "off_diagonal_means": off_diagonal_means,
        "best_generalizer": best,
        "worst_generalizer": worst,
    }
