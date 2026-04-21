from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
from pathlib import Path

from dialect_utils import (
    DIALECTS,
    equation_overlap,
    equation_signature,
    exact_input_overlap,
    family_counts,
    load_jsonl,
    stratified_group_split,
    stratified_instance_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one BART classifier per PDE dialect and optionally run cross-dialect evaluation."
    )
    parser.add_argument(
        "--dataset",
        default="varied_data_generation/dataset.jsonl",
        help="Path to the training dataset JSONL.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/bart_dialects",
        help="Directory where checkpoints and metrics will be written.",
    )
    parser.add_argument(
        "--model-name",
        default="facebook/bart-base",
        help="Hugging Face model name or local checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--dialects",
        nargs="+",
        default=list(DIALECTS),
        choices=list(DIALECTS),
        help="Dialects to train/evaluate.",
    )
    parser.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="Optional subset of PDE families to keep.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("instance", "equation"),
        default="instance",
        help=(
            "instance reproduces the current notebook behavior; equation keeps the same family+coefficients "
            "out of multiple splits, which is stricter but may fail on highly repetitive families."
        ),
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument(
        "--cross-eval",
        action="store_true",
        help="After in-dialect evaluation, run the full train-dialect x test-dialect matrix.",
    )
    parser.add_argument(
        "--cross-eval-split",
        choices=("val", "test"),
        default="test",
        help="Which split to use when building the cross-dialect accuracy matrix.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only print split/leakage diagnostics; do not import transformers or train models.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore any existing checkpoints under output-root and retrain from scratch.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Execution device. Use cpu to avoid Apple MPS issues during evaluation.",
    )
    return parser.parse_args()


def choose_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(args: argparse.Namespace, torch_module) -> str:
    if args.device != "auto":
        return args.device
    return choose_device(torch_module)


def set_random_seed(seed: int, torch_module) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def split_dataset(data: list[dict], args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict]]:
    if args.split_mode == "instance":
        return stratified_instance_split(
            data,
            seed=args.seed,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
        )

    return stratified_group_split(
        data,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        signature_fn=equation_signature,
    )


def print_split_report(train_data: list[dict], val_data: list[dict], test_data: list[dict]) -> None:
    print("Split sizes:")
    print(f"  train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")
    print("Family counts:")
    print(f"  train={family_counts(train_data)}")
    print(f"  val={family_counts(val_data)}")
    print(f"  test={family_counts(test_data)}")

    print("Train/Test overlap diagnostics:")
    for dialect in DIALECTS:
        overlap = exact_input_overlap(train_data, test_data, dialect)
        print(
            "  "
            f"{dialect:7s} exact-overlap={overlap['overlap_examples']}/{len(test_data)} "
            f"({overlap['overlap_rate']:.1%}), "
            f"unique-overlap={overlap['overlap_unique']}"
        )

    equation = equation_overlap(train_data, test_data)
    print(
        "  "
        f"equation exact-overlap={equation['overlap_examples']}/{len(test_data)} "
        f"({equation['overlap_rate']:.1%}), "
        f"unique-overlap={equation['overlap_unique']}"
    )


def save_matrix(path: Path, matrix: dict[str, dict[str, float]], dialects: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + dialects)
        for row in dialects:
            writer.writerow([row] + [matrix[row][column] for column in dialects])


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    checkpoints = sorted(
        (path for path in output_dir.iterdir() if path.name.startswith("checkpoint-")),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def import_training_stack():
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from transformers import (
            BartForSequenceClassification,
            BartTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install at least transformers, datasets, numpy, and torch "
            "in the environment that will run this script."
        ) from exc

    return {
        "np": np,
        "torch": torch,
        "Dataset": Dataset,
        "BartForSequenceClassification": BartForSequenceClassification,
        "BartTokenizer": BartTokenizer,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def make_training_arguments(args: argparse.Namespace, output_dir: Path, stack: dict, device: str):
    TrainingArguments = stack["TrainingArguments"]
    signature = inspect.signature(TrainingArguments.__init__).parameters

    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "save_total_limit": 1,
        "seed": args.seed,
    }

    if "use_cpu" in signature:
        kwargs["use_cpu"] = device == "cpu"

    if "use_mps_device" in signature:
        kwargs["use_mps_device"] = device == "mps"

    if "evaluation_strategy" in signature:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in signature:
        kwargs["eval_strategy"] = "epoch"

    if "save_strategy" in signature:
        kwargs["save_strategy"] = "epoch"

    if "weight_decay" in signature:
        kwargs["weight_decay"] = 0.01

    if "fp16" in signature:
        kwargs["fp16"] = device == "cuda"

    if "dataloader_pin_memory" in signature:
        kwargs["dataloader_pin_memory"] = device == "cuda"

    return TrainingArguments(**kwargs)


def make_eval_arguments(output_dir: Path, stack: dict, device: str, batch_size: int):
    TrainingArguments = stack["TrainingArguments"]
    signature = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": str(output_dir),
        "per_device_eval_batch_size": batch_size,
        "report_to": "none",
    }
    if "use_cpu" in signature:
        kwargs["use_cpu"] = device == "cpu"
    if "use_mps_device" in signature:
        kwargs["use_mps_device"] = device == "mps"
    if "fp16" in signature:
        kwargs["fp16"] = device == "cuda"
    if "dataloader_pin_memory" in signature:
        kwargs["dataloader_pin_memory"] = device == "cuda"
    return TrainingArguments(**kwargs)


def make_trainer(model, training_args, data_collator, stack: dict, train_dataset=None, eval_dataset=None):
    Trainer = stack["Trainer"]
    kwargs = {
        "model": model,
        "args": training_args,
        "data_collator": data_collator,
    }
    if train_dataset is not None:
        kwargs["train_dataset"] = train_dataset
    if eval_dataset is not None:
        kwargs["eval_dataset"] = eval_dataset
    return Trainer(**kwargs)


def accuracy_from_predictions(predictions, label_ids, np_module) -> float:
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    pred_labels = np_module.argmax(logits, axis=1)
    return float(np_module.mean(pred_labels == label_ids))


def main() -> None:
    args = parse_args()

    data = load_jsonl(args.dataset)
    if args.families:
        keep = set(args.families)
        data = [instance for instance in data if instance["family"] in keep]
        if not data:
            raise SystemExit("No data left after applying --families filter.")

    try:
        train_data, val_data, test_data = split_dataset(data, args)
    except ValueError as exc:
        raise SystemExit(
            f"{exc}\nHint: try '--split-mode instance' to reproduce the notebook behavior, "
            "or filter out the family that has too few unique equations."
        ) from exc

    print_split_report(train_data, val_data, test_data)

    if args.report_only:
        return

    stack = import_training_stack()
    np = stack["np"]
    BartTokenizer = stack["BartTokenizer"]
    BartForSequenceClassification = stack["BartForSequenceClassification"]
    Dataset = stack["Dataset"]
    DataCollatorWithPadding = stack["DataCollatorWithPadding"]
    torch = stack["torch"]

    device = resolve_device(args, torch)
    print(f"Training device: {device}")
    set_random_seed(args.seed, torch)

    families = sorted({instance["labels"]["behavioral"] for instance in data})
    label2id = {family: idx for idx, family in enumerate(families)}
    id2label = {idx: family for family, idx in label2id.items()}

    def preprocess(instances: list[dict], dialect: str) -> dict[str, list]:
        return {
            "text": [instance["dialects"][dialect] for instance in instances],
            "label": [label2id[instance["labels"]["behavioral"]] for instance in instances],
        }

    output_root = Path(args.output_root) / args.split_mode
    output_root.mkdir(parents=True, exist_ok=True)

    models_by_dialect: dict[str, object] = {}
    results: list[dict] = []

    for dialect in args.dialects:
        print(f"\n=== DIALECT: {dialect.upper()} ===")
        model_output_dir = output_root / dialect
        model_output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = BartTokenizer.from_pretrained(args.model-name if False else args.model_name)

        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"],
                max_length=args.max_input_length,
                truncation=True,
                padding="max_length",
            )
            tokenized["labels"] = examples["label"]
            return tokenized

        checkpoint = None if args.force_retrain else find_latest_checkpoint(model_output_dir)
        if checkpoint is not None:
            print(f"Loading checkpoint: {checkpoint}")
            tokenizer = BartTokenizer.from_pretrained(checkpoint)
            model = BartForSequenceClassification.from_pretrained(checkpoint)
        else:
            train_dataset = Dataset.from_dict(preprocess(train_data, dialect)).map(
                tokenize_function,
                batched=True,
                remove_columns=["text", "label"],
            )
            val_dataset = Dataset.from_dict(preprocess(val_data, dialect)).map(
                tokenize_function,
                batched=True,
                remove_columns=["text", "label"],
            )

            model = BartForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
            )
            model.to(device)

            training_args = make_training_arguments(args, model_output_dir, stack, device)
            trainer = make_trainer(
                model=model,
                training_args=training_args,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                stack=stack,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )
            trainer.train()
            trainer.save_model(str(model_output_dir))
            tokenizer.save_pretrained(str(model_output_dir))

        model.eval()
        model.to(device)

        test_dataset = Dataset.from_dict(preprocess(test_data, dialect)).map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "label"],
        )
        eval_args = make_eval_arguments(output_root / "eval", stack, device, args.batch_size)
        eval_trainer = make_trainer(
            model=model,
            training_args=eval_args,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            stack=stack,
        )
        predictions = eval_trainer.predict(test_dataset)
        accuracy = accuracy_from_predictions(predictions.predictions, predictions.label_ids, np)
        results.append({"dialect": dialect, "family_accuracy": accuracy, "n": len(test_dataset)})
        print(f"  family_acc={accuracy:.2%}")
        models_by_dialect[dialect] = (model, tokenizer)

    results_path = output_root / "bart_dialect_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved in-dialect metrics to {results_path}")

    if not args.cross_eval:
        return

    cross_eval_data = val_data if args.cross_eval_split == "val" else test_data
    print(f"\nCross-dialect matrix split: {args.cross_eval_split}")

    accuracy_matrix: dict[str, dict[str, float]] = {
        train_dialect: {} for train_dialect in args.dialects
    }

    for train_dialect in args.dialects:
        model, tokenizer = models_by_dialect[train_dialect]
        model.eval()
        model.to(device)
        eval_args = make_eval_arguments(output_root / "eval", stack, device, args.batch_size)
        eval_trainer = make_trainer(
            model=model,
            training_args=eval_args,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            stack=stack,
        )

        def tokenize_for_eval(examples):
            tokenized = tokenizer(
                examples["text"],
                max_length=args.max_input_length,
                truncation=True,
                padding="max_length",
            )
            tokenized["labels"] = examples["label"]
            return tokenized

        print(f"\nCross-evaluating model trained on {train_dialect.upper()}")
        for test_dialect in args.dialects:
            test_dataset = Dataset.from_dict(preprocess(cross_eval_data, test_dialect)).map(
                tokenize_for_eval,
                batched=True,
                remove_columns=["text", "label"],
            )
            predictions = eval_trainer.predict(test_dataset)
            accuracy = accuracy_from_predictions(
                predictions.predictions, predictions.label_ids, np
            )
            accuracy_matrix[train_dialect][test_dialect] = round(accuracy * 100, 1)
            print(f"  tested_on={test_dialect:7s}  acc={accuracy:.1%}")

    accuracy_filename = (
        "cross_dialect_accuracy.csv"
        if args.cross_eval_split == "test"
        else f"cross_dialect_accuracy_{args.cross_eval_split}.csv"
    )
    accuracy_path = output_root / accuracy_filename
    save_matrix(accuracy_path, accuracy_matrix, args.dialects)
    print(f"\nSaved cross-dialect accuracy to {accuracy_path}")


if __name__ == "__main__":
    main()
