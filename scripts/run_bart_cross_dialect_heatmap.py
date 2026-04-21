from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate the fine-tuned BART cross-dialect classifier and save a heatmap "
            "for either the validation or test split."
        )
    )
    parser.add_argument(
        "--dataset",
        default="varied_data_generation/dataset.jsonl",
        help="Path to the dataset JSONL, relative to the repo root unless absolute.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/bart_dialects_cross_val_80_10_10",
        help="Directory where checkpoints, CSV metrics, and the heatmap image will be written.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("instance", "equation"),
        default="instance",
        help="Train/validation/test splitting mode.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--matrix-split",
        choices=("val", "test"),
        default="val",
        help="Which split to use for the cross-dialect matrix.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Execution device forwarded to the BART classification script.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even if the matrix CSV already exists.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only plot an existing matrix CSV; do not run training.",
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def matrix_filename(matrix_split: str) -> str:
    return "cross_dialect_accuracy.csv" if matrix_split == "test" else f"cross_dialect_accuracy_{matrix_split}.csv"


def ensure_matrix(args: argparse.Namespace, matrix_path: Path) -> None:
    if args.plot_only:
        if not matrix_path.exists():
            raise FileNotFoundError(f"{matrix_path} does not exist and --plot-only was requested.")
        return

    if matrix_path.exists() and not args.force_retrain:
        print(f"Using existing matrix: {matrix_path}")
        return

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_bart_dialect_classification.py"),
        "--dataset",
        str(resolve_repo_path(args.dataset)),
        "--output-root",
        str(resolve_repo_path(args.output_root)),
        "--split-mode",
        args.split_mode,
        "--train-fraction",
        str(args.train_fraction),
        "--val-fraction",
        str(args.val_fraction),
        "--test-fraction",
        str(args.test_fraction),
        "--batch-size",
        str(args.batch_size),
        "--num-epochs",
        str(args.num_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--max-input-length",
        str(args.max_input_length),
        "--cross-eval",
        "--cross-eval-split",
        args.matrix_split,
        "--device",
        args.device,
    ]
    if args.force_retrain:
        cmd.append("--force-retrain")

    print("Running BART cross-dialect experiment:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def load_matrix(matrix_path: Path) -> pd.DataFrame:
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"{matrix_path} does not exist. Run this script without --plot-only, "
            "or rerun with --force-retrain if training failed previously."
        )
    return pd.read_csv(matrix_path, index_col=0)


def plot_heatmap(df: pd.DataFrame, figure_path: Path, matrix_split: str) -> None:
    import matplotlib.pyplot as plt

    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(df.values, cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_xlabel(f"Evaluation Dialect ({matrix_split} split)")
    ax.set_ylabel("Fine-Tuned On Dialect")
    ax.set_title(f"Fine-Tuned BART Cross-Dialect {matrix_split.capitalize()} Accuracy")

    for row_idx, row_name in enumerate(df.index):
        for col_idx, col_name in enumerate(df.columns):
            value = df.loc[row_name, col_name]
            ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", color="black", fontsize=10)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Family accuracy (%)")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = resolve_repo_path(args.output_root)
    split_output_dir = output_root / args.split_mode
    matrix_path = split_output_dir / matrix_filename(args.matrix_split)
    figure_path = split_output_dir / f"{matrix_path.stem}_heatmap.png"

    ensure_matrix(args, matrix_path)
    df = load_matrix(matrix_path)
    plot_heatmap(df, figure_path, args.matrix_split)

    print(f"Saved matrix CSV to {matrix_path}")
    print(f"Saved heatmap image to {figure_path}")
    print("Matrix preview:")
    print(df.to_string())


if __name__ == "__main__":
    main()
