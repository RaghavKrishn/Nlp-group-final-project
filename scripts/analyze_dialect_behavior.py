from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from dialect_utils import (
    DIALECTS,
    equation_overlap,
    equation_signature,
    exact_input_overlap,
    family_counts,
    load_accuracy_matrix,
    load_jsonl,
    stratified_instance_split,
    summarize_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize four-dialect experiment behavior and quantify train/test leakage."
    )
    parser.add_argument(
        "--dataset",
        default="varied_data_generation/dataset.jsonl",
        help="Path to the JSONL dataset used for the experiments.",
    )
    parser.add_argument(
        "--cross-accuracy",
        default="cross_dialect_accuracy.csv",
        help="CSV produced by the cross-dialect notebook or training script.",
    )
    parser.add_argument(
        "--in-dialect-results",
        default="dialect_results.json",
        help="JSON file with per-dialect in-dialect metrics.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def print_dataset_summary(data: list[dict]) -> None:
    print("Dataset summary:")
    print(f"  instances={len(data)}")
    print(f"  families={family_counts(data)}")

    for dialect in DIALECTS:
        unique_count = len({instance["dialects"][dialect] for instance in data})
        print(f"  unique_{dialect}_inputs={unique_count}")

    equation_groups = Counter(equation_signature(instance) for instance in data)
    by_family_groups: dict[str, set[str]] = defaultdict(set)
    for instance in data:
        by_family_groups[instance["family"]].add(equation_signature(instance))

    print("Unique equation groups per family:")
    for family in sorted(by_family_groups):
        print(f"  {family}: {len(by_family_groups[family])}")

    impossible = {family: len(groups) for family, groups in by_family_groups.items() if len(groups) < 3}
    if impossible:
        details = ", ".join(f"{family}={count}" for family, count in sorted(impossible.items()))
        print(
            "  warning: equation-level train/val/test splitting is not possible for every family "
            f"with the current dataset ({details})."
        )

    print("Most repeated equations:")
    for signature, count in equation_groups.most_common(5):
        payload = json.loads(signature)
        print(f"  {payload['family']} {payload['coefficients']} -> {count} copies")


def print_overlap_summary(data: list[dict], seed: int) -> None:
    train_data, _, test_data = stratified_instance_split(
        data,
        seed=seed,
        train_fraction=0.8,
        val_fraction=0.1,
        test_fraction=0.1,
    )

    print("Approximate notebook-style split leakage (instance-level split):")
    for dialect in DIALECTS:
        overlap = exact_input_overlap(train_data, test_data, dialect)
        print(
            "  "
            f"{dialect:7s} exact-train/test overlap={overlap['overlap_examples']}/{len(test_data)} "
            f"({overlap['overlap_rate']:.1%})"
        )

    eq_overlap = equation_overlap(train_data, test_data)
    print(
        "  "
        f"equation train/test overlap={eq_overlap['overlap_examples']}/{len(test_data)} "
        f"({eq_overlap['overlap_rate']:.1%})"
    )


def print_result_summary(in_dialect_path: str, cross_accuracy_path: str) -> None:
    print("Saved metric summary:")

    with open(in_dialect_path) as handle:
        in_dialect = json.load(handle)

    in_family = {row["dialect"]: row["family_accuracy"] * 100 for row in in_dialect}
    print(f"  in_dialect_family_accuracy={in_family}")

    matrix = load_accuracy_matrix(cross_accuracy_path)
    summary = summarize_matrix(matrix)
    print(f"  mean_diagonal_accuracy={summary['diagonal_mean']:.1f}%")
    print(f"  mean_off_diagonal_accuracy={summary['off_diagonal_mean']:.1f}%")
    print(
        "  "
        f"best_cross_generalizer={summary['best_generalizer']} "
        f"({summary['off_diagonal_means'][summary['best_generalizer']]:.1f}%)"
    )
    print(
        "  "
        f"worst_cross_generalizer={summary['worst_generalizer']} "
        f"({summary['off_diagonal_means'][summary['worst_generalizer']]:.1f}%)"
    )

    print("  row_means_off_diagonal={")
    for dialect in summary["dialects"]:
        print(f"    {dialect}: {summary['off_diagonal_means'][dialect]:.1f}%")
    print("  }")

    print("Largest asymmetries:")
    asymmetries = []
    for row in summary["dialects"]:
        for column in summary["dialects"]:
            if row >= column:
                continue
            left = matrix[row][column]
            right = matrix[column][row]
            asymmetries.append((abs(left - right), row, column, left, right))

    for _, left_name, right_name, left_score, right_score in sorted(asymmetries, reverse=True)[:5]:
        print(
            "  "
            f"{left_name}->{right_name}={left_score:.1f}%  "
            f"{right_name}->{left_name}={right_score:.1f}%"
        )


def main() -> None:
    args = parse_args()
    data = load_jsonl(args.dataset)
    print_dataset_summary(data)
    print()
    print_overlap_summary(data, args.seed)
    print()
    print_result_summary(args.in_dialect_results, args.cross_accuracy)


if __name__ == "__main__":
    main()
