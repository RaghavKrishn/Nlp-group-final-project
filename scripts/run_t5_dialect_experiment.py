from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path

from dialect_utils import (
    DIALECTS,
    build_target,
    equation_overlap,
    equation_signature,
    exact_input_overlap,
    family_counts,
    is_reasoning_correct,
    load_jsonl,
    parse_prediction,
    stratified_group_split,
    stratified_instance_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one T5-style seq2seq model per PDE dialect and optionally run cross-dialect evaluation."
    )
    parser.add_argument(
        "--dataset",
        default="varied_data_generation/dataset.jsonl",
        help="Path to the training dataset JSONL.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/t5_dialects",
        help="Directory where checkpoints and metrics will be written.",
    )
    parser.add_argument(
        "--model-name",
        default="t5-small",
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
            "instance reproduces the notebook behavior; equation keeps the same family+coefficients "
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
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-output-length", type=int, default=256)
    parser.add_argument(
        "--cross-eval",
        action="store_true",
        help="After in-dialect evaluation, run the full train-dialect x test-dialect matrix.",
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
    return parser.parse_args()


def choose_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


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
        import evaluate
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install at least transformers, evaluate, and torch "
            "in the environment that will run this script."
        ) from exc

    return {
        "evaluate": evaluate,
        "torch": torch,
        "Dataset": Dataset,
        "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "Seq2SeqTrainer": Seq2SeqTrainer,
        "Seq2SeqTrainingArguments": Seq2SeqTrainingArguments,
    }


def compute_metrics_bundle(test_data: list[dict], predictions_raw: list[str], rouge_module) -> dict:
    parsed = [parse_prediction(prediction) for prediction in predictions_raw]

    correct_family = sum(
        1
        for gold, pred in zip(test_data, parsed)
        if pred["family"] and pred["family"].lower() == gold["labels"]["behavioral"].lower()
    )
    family_accuracy = correct_family / len(test_data)

    total_precision = 0.0
    total_recall = 0.0
    for gold, pred in zip(test_data, parsed):
        true_ops = {item.lower() for item in gold["labels"]["operators"]}
        pred_ops = {item.lower() for item in pred["operators"]}
        if not pred_ops and not true_ops:
            total_precision += 1.0
            total_recall += 1.0
            continue
        if pred_ops:
            true_positives = len(true_ops & pred_ops)
            total_precision += true_positives / len(pred_ops)
            total_recall += true_positives / len(true_ops) if true_ops else 0.0

    operator_precision = total_precision / len(test_data)
    operator_recall = total_recall / len(test_data)
    if operator_precision + operator_recall == 0:
        operator_f1 = 0.0
    else:
        operator_f1 = (
            2 * operator_precision * operator_recall / (operator_precision + operator_recall)
        )

    rouge_scores = rouge_module.compute(
        predictions=[item["reasoning"] for item in parsed],
        references=[item["labels"]["reasoning"] for item in test_data],
        rouge_types=["rougeL"],
    )

    trash_count = 0
    for gold, pred in zip(test_data, parsed):
        label_correct = (
            pred["family"] and pred["family"].lower() == gold["labels"]["behavioral"].lower()
        )
        if label_correct and not is_reasoning_correct(gold["family"], pred["reasoning"]):
            trash_count += 1

    return {
        "family_accuracy": family_accuracy,
        "operator_precision": operator_precision,
        "operator_recall": operator_recall,
        "operator_f1": operator_f1,
        "reasoning_rouge_l": rouge_scores["rougeL"],
        "trash_score": trash_count / len(test_data),
        "trash_count": trash_count,
        "n": len(test_data),
    }


def make_training_arguments(args: argparse.Namespace, output_dir: Path, stack: dict, device: str):
    TrainingArguments = stack["Seq2SeqTrainingArguments"]
    signature = inspect.signature(TrainingArguments.__init__).parameters

    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "predict_with_generate": True,
        "generation_max_length": args.max_output_length,
        "seed": args.seed,
        "logging_steps": 50,
        "report_to": "none",
        "save_total_limit": 1,
    }

    if "evaluation_strategy" in signature:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in signature:
        kwargs["eval_strategy"] = "epoch"

    if "save_strategy" in signature:
        kwargs["save_strategy"] = "epoch"

    if "fp16" in signature:
        kwargs["fp16"] = device == "cuda"

    return TrainingArguments(**kwargs)


def make_trainer(model, tokenizer, train_dataset, val_dataset, data_collator, training_args, stack: dict):
    Trainer = stack["Seq2SeqTrainer"]
    signature = inspect.signature(Trainer.__init__).parameters
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
    }
    if "processing_class" in signature:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


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
    torch = stack["torch"]
    DatasetBase = stack["Dataset"]
    AutoTokenizer = stack["AutoTokenizer"]
    AutoModelForSeq2SeqLM = stack["AutoModelForSeq2SeqLM"]
    DataCollatorForSeq2Seq = stack["DataCollatorForSeq2Seq"]
    evaluate = stack["evaluate"]

    class Seq2SeqExamples(DatasetBase):
        def __init__(self, instances: list[dict], dialect: str, tokenizer):
            inputs = [instance["dialects"][dialect] for instance in instances]
            targets = [build_target(instance) for instance in instances]
            model_inputs = tokenizer(
                inputs,
                max_length=args.max_input_length,
                truncation=True,
                padding="max_length",
            )
            labels = tokenizer(
                text_target=targets,
                max_length=args.max_output_length,
                truncation=True,
                padding="max_length",
            )
            self.features = {
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "labels": labels["input_ids"],
            }

        def __len__(self) -> int:
            return len(self.features["input_ids"])

        def __getitem__(self, index: int) -> dict:
            return {
                key: torch.tensor(value[index], dtype=torch.long)
                for key, value in self.features.items()
            }

    output_root = Path(args.output_root) / args.split_mode
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    rouge = evaluate.load("rouge")
    device = choose_device(torch)
    print(f"Training device: {device}")

    models_by_dialect: dict[str, object] = {}
    results: list[dict] = []

    for dialect in args.dialects:
        print(f"\n=== DIALECT: {dialect.upper()} ===")
        model_output_dir = output_root / dialect
        model_output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = None if args.force_retrain else find_latest_checkpoint(model_output_dir)
        if checkpoint is not None:
            print(f"Loading checkpoint: {checkpoint}")
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
            model.to(device)

            train_dataset = Seq2SeqExamples(train_data, dialect, tokenizer)
            val_dataset = Seq2SeqExamples(val_data, dialect, tokenizer)
            training_args = make_training_arguments(args, model_output_dir, stack, device)
            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
            trainer = make_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                data_collator=data_collator,
                training_args=training_args,
                stack=stack,
            )
            trainer.train()

        model.eval()
        model.to("cpu")

        predictions_raw: list[str] = []
        for start in range(0, len(test_data), args.batch_size):
            batch = test_data[start : start + args.batch_size]
            encoded = tokenizer(
                [instance["dialects"][dialect] for instance in batch],
                return_tensors="pt",
                max_length=args.max_input_length,
                truncation=True,
                padding=True,
            )
            with torch.no_grad():
                outputs = model.generate(**encoded, max_new_tokens=args.max_output_length)
            predictions_raw.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

        metrics = compute_metrics_bundle(test_data, predictions_raw, rouge)
        metrics["dialect"] = dialect
        print(
            f"  family_acc={metrics['family_accuracy']:.2%}  "
            f"op_f1={metrics['operator_f1']:.2%}  "
            f"rougeL={metrics['reasoning_rouge_l']:.2%}  "
            f"trash={metrics['trash_score']:.2%}"
        )
        results.append(metrics)
        models_by_dialect[dialect] = model

    results_path = output_root / "dialect_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved in-dialect metrics to {results_path}")

    if not args.cross_eval:
        return

    accuracy_matrix: dict[str, dict[str, float]] = {
        train_dialect: {} for train_dialect in args.dialects
    }
    trash_matrix: dict[str, dict[str, float]] = {
        train_dialect: {} for train_dialect in args.dialects
    }

    for train_dialect in args.dialects:
        model = models_by_dialect[train_dialect]
        model.eval()
        model.to("cpu")
        print(f"\nCross-evaluating model trained on {train_dialect.upper()}")

        for test_dialect in args.dialects:
            predictions_raw = []
            for start in range(0, len(test_data), args.batch_size):
                batch = test_data[start : start + args.batch_size]
                encoded = tokenizer(
                    [instance["dialects"][test_dialect] for instance in batch],
                    return_tensors="pt",
                    max_length=args.max_input_length,
                    truncation=True,
                    padding=True,
                )
                with torch.no_grad():
                    outputs = model.generate(**encoded, max_new_tokens=args.max_output_length)
                predictions_raw.extend(
                    tokenizer.batch_decode(outputs, skip_special_tokens=True)
                )

            metrics = compute_metrics_bundle(test_data, predictions_raw, rouge)
            accuracy_matrix[train_dialect][test_dialect] = round(
                metrics["family_accuracy"] * 100, 1
            )
            trash_matrix[train_dialect][test_dialect] = round(
                metrics["trash_score"] * 100, 1
            )
            print(
                f"  tested_on={test_dialect:7s}  "
                f"acc={metrics['family_accuracy']:.1%}  "
                f"trash={metrics['trash_score']:.1%}"
            )

    accuracy_path = output_root / "cross_dialect_accuracy.csv"
    trash_path = output_root / "cross_dialect_trash.csv"
    save_matrix(accuracy_path, accuracy_matrix, args.dialects)
    save_matrix(trash_path, trash_matrix, args.dialects)
    print(f"\nSaved cross-dialect accuracy to {accuracy_path}")
    print(f"Saved cross-dialect trash to {trash_path}")


if __name__ == "__main__":
    main()
