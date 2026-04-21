from __future__ import annotations

import argparse
import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dialect_utils import (
    DIALECTS,
    build_operator_vocab,
    derive_structure_targets,
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
        description=(
            "Train a multitask BART encoder on PDE language with family classification, "
            "operator prediction, and auxiliary structural targets."
        )
    )
    parser.add_argument(
        "--dataset",
        default="varied_data_generation/dataset.jsonl",
        help="Path to the training dataset JSONL.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/bart_multitask_pde",
        help="Directory where checkpoints and metrics will be written.",
    )
    parser.add_argument(
        "--model-name",
        default="facebook/bart-base",
        help="Hugging Face model name or local checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--train-mode",
        choices=("mixed", "single"),
        default="mixed",
        help=(
            "mixed expands every PDE into all chosen dialects during training. "
            "single trains on exactly one dialect."
        ),
    )
    parser.add_argument(
        "--train-dialect",
        choices=DIALECTS,
        default="natural",
        help="Dialect used when --train-mode=single.",
    )
    parser.add_argument(
        "--eval-dialects",
        nargs="+",
        default=list(DIALECTS),
        choices=list(DIALECTS),
        help="Dialects to evaluate after training.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("instance", "equation"),
        default="instance",
        help=(
            "instance reproduces the notebook behavior; equation keeps identical family+coefficients "
            "out of multiple splits, which is stricter but may fail on repetitive families."
        ),
    )
    parser.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="Optional subset of PDE families to keep.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-input-length", type=int, default=160)
    parser.add_argument(
        "--operator-loss-weight",
        type=float,
        default=0.35,
        help="Relative weight for the operator multi-label loss.",
    )
    parser.add_argument(
        "--structure-loss-weight",
        type=float,
        default=0.15,
        help="Relative weight for each structural auxiliary loss.",
    )
    parser.add_argument(
        "--no-dialect-token",
        action="store_true",
        help="Disable explicit dialect tags in the input text.",
    )
    parser.add_argument(
        "--task-prefix",
        default="classify PDE family and infer structure:",
        help="Natural-language prefix prepended before each equation string.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only print split/leakage diagnostics; do not import transformers or train.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore any existing checkpoint under output-root and retrain from scratch.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help=(
            "Skip fine-tuning and evaluate the pretrained BART backbone directly. "
            "The multitask heads remain randomly initialized, so this is a sanity-check baseline."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Execution device. Use cpu to avoid Apple MPS issues during evaluation.",
    )
    parser.add_argument(
        "--held-out-datasets",
        nargs="*",
        default=[],
        help=(
            "Optional extra JSONL datasets to evaluate after the standard val/test split. "
            "Useful for held-out families such as Beam, KleinGordon, or ReactionDiffusion."
        ),
    )
    return parser.parse_args()


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
            f"({overlap['overlap_rate']:.1%}), unique-overlap={overlap['overlap_unique']}"
        )
    equation = equation_overlap(train_data, test_data)
    print(
        "  "
        f"equation exact-overlap={equation['overlap_examples']}/{len(test_data)} "
        f"({equation['overlap_rate']:.1%}), unique-overlap={equation['overlap_unique']}"
    )


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    checkpoints = sorted(
        (path for path in output_dir.iterdir() if path.name.startswith("checkpoint-")),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def tokenizer_source_for_output(output_root: Path, fallback: str) -> str:
    if (output_root / "tokenizer_config.json").exists():
        return str(output_root)
    return fallback


def import_training_stack():
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            BartConfig,
            BartModel,
            BartPreTrainedModel,
            BartTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install at least transformers and torch "
            "in the environment that will run this script."
        ) from exc

    return {
        "torch": torch,
        "F": F,
        "nn": nn,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "BartConfig": BartConfig,
        "BartModel": BartModel,
        "BartPreTrainedModel": BartPreTrainedModel,
        "BartTokenizer": BartTokenizer,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


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


def format_input_text(text: str, dialect: str, args: argparse.Namespace) -> str:
    pieces = []
    if not args.no_dialect_token:
        pieces.append(f"[DIALECT_{dialect.upper()}]")
    if args.task_prefix:
        pieces.append(args.task_prefix.strip())
    pieces.append(text)
    return " ".join(pieces)


def build_examples(
    instances: list[dict],
    train_mode: str,
    dialects: list[str],
    args: argparse.Namespace,
    family2id: dict[str, int],
    operator2id: dict[str, int],
    single_dialect: str | None = None,
    allow_unknown_families: bool = False,
) -> list[dict[str, Any]]:
    selected_dialects = dialects if train_mode == "mixed" else [single_dialect or args.train_dialect]
    examples: list[dict[str, Any]] = []

    for instance in instances:
        family = instance["labels"]["behavioral"]
        if family not in family2id and not allow_unknown_families:
            raise KeyError(
                f"Family '{family}' is not in the training label space. "
                "Use allow_unknown_families=True for held-out evaluation."
            )
        structure = derive_structure_targets(instance)
        operator_targets = [0.0] * len(operator2id)
        for operator in instance["labels"]["operators"]:
            operator_targets[operator2id[operator.lower()]] = 1.0

        for dialect in selected_dialects:
            examples.append(
                {
                    "text": format_input_text(instance["dialects"][dialect], dialect, args),
                    "dialect": dialect,
                    "family_label": family2id.get(family, -100),
                    "family_name": family,
                    "operator_labels": operator_targets,
                    "time_order_label": structure["time_order"],
                    "first_spatial_label": float(structure["has_first_spatial"]),
                    "second_spatial_label": float(structure["has_second_spatial"]),
                    "nonlinear_label": float(structure["nonlinear"]),
                    "spatial_var_label": structure["spatial_vars"] - 1,
                }
            )

    return examples


def compute_micro_f1(predicted: list[list[int]], gold: list[list[int]]) -> float:
    tp = fp = fn = 0
    for pred_row, gold_row in zip(predicted, gold):
        for pred_value, gold_value in zip(pred_row, gold_row):
            if pred_value == 1 and gold_value == 1:
                tp += 1
            elif pred_value == 1 and gold_value == 0:
                fp += 1
            elif pred_value == 0 and gold_value == 1:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


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
        "seed": args.seed,
        "save_total_limit": 1,
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


@dataclass
class MetricBundle:
    family_accuracy: float | None
    operator_micro_f1: float
    time_order_accuracy: float
    first_spatial_accuracy: float
    second_spatial_accuracy: float
    nonlinear_accuracy: float
    spatial_var_accuracy: float
    family_accuracy_support: int | None = None
    top_predicted_family: str | None = None
    predicted_family_distribution: dict[str, float] | None = None

    @property
    def structure_accuracy(self) -> float:
        return (
            self.time_order_accuracy
            + self.first_spatial_accuracy
            + self.second_spatial_accuracy
            + self.nonlinear_accuracy
            + self.spatial_var_accuracy
        ) / 5.0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "family_accuracy": self.family_accuracy,
            "operator_micro_f1": self.operator_micro_f1,
            "time_order_accuracy": self.time_order_accuracy,
            "first_spatial_accuracy": self.first_spatial_accuracy,
            "second_spatial_accuracy": self.second_spatial_accuracy,
            "nonlinear_accuracy": self.nonlinear_accuracy,
            "spatial_var_accuracy": self.spatial_var_accuracy,
            "structure_accuracy": self.structure_accuracy,
        }
        if self.family_accuracy_support is not None:
            payload["family_accuracy_support"] = self.family_accuracy_support
        if self.top_predicted_family is not None:
            payload["top_predicted_family"] = self.top_predicted_family
        if self.predicted_family_distribution is not None:
            payload["predicted_family_distribution"] = self.predicted_family_distribution
        return payload


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
            f"{exc}\nHint: try '--split-mode instance' to reproduce the current notebook behavior, "
            "or filter out the family that has too few unique equations."
        ) from exc

    print_split_report(train_data, val_data, test_data)

    family2id = {
        family: idx
        for idx, family in enumerate(sorted({instance["labels"]["behavioral"] for instance in data}))
    }
    id2family = {idx: family for family, idx in family2id.items()}
    operator_vocab = build_operator_vocab(data)
    operator2id = {operator: idx for idx, operator in enumerate(operator_vocab)}

    train_examples = build_examples(
        train_data,
        train_mode=args.train_mode,
        dialects=list(DIALECTS),
        args=args,
        family2id=family2id,
        operator2id=operator2id,
    )
    val_examples = build_examples(
        val_data,
        train_mode=args.train_mode,
        dialects=list(DIALECTS),
        args=args,
        family2id=family2id,
        operator2id=operator2id,
        single_dialect=args.train_dialect,
    )

    print("Expanded example counts:")
    print(f"  train_examples={len(train_examples)}")
    print(f"  val_examples={len(val_examples)}")

    if args.report_only:
        return

    stack = import_training_stack()
    torch = stack["torch"]
    F = stack["F"]
    nn = stack["nn"]
    DataLoader = stack["DataLoader"]
    Dataset = stack["Dataset"]
    BartModel = stack["BartModel"]
    BartPreTrainedModel = stack["BartPreTrainedModel"]
    BartTokenizer = stack["BartTokenizer"]
    Trainer = stack["Trainer"]

    class PDEMultitaskDataset(Dataset):
        def __init__(self, examples: list[dict[str, Any]], tokenizer, max_length: int):
            self.examples = examples
            self.encodings = tokenizer(
                [example["text"] for example in examples],
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            example = self.examples[index]
            item = {
                "input_ids": torch.tensor(self.encodings["input_ids"][index], dtype=torch.long),
                "attention_mask": torch.tensor(
                    self.encodings["attention_mask"][index], dtype=torch.long
                ),
                "labels": torch.tensor(example["family_label"], dtype=torch.long),
                "operator_labels": torch.tensor(example["operator_labels"], dtype=torch.float),
                "time_order_labels": torch.tensor(example["time_order_label"], dtype=torch.long),
                "first_spatial_labels": torch.tensor(
                    example["first_spatial_label"], dtype=torch.float
                ),
                "second_spatial_labels": torch.tensor(
                    example["second_spatial_label"], dtype=torch.float
                ),
                "nonlinear_labels": torch.tensor(example["nonlinear_label"], dtype=torch.float),
                "spatial_var_labels": torch.tensor(example["spatial_var_label"], dtype=torch.long),
            }
            return item

    class BartPDEMultitaskClassifier(BartPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            dropout_prob = (
                config.classifier_dropout
                if config.classifier_dropout is not None
                else config.dropout
            )
            self.model = BartModel(config)
            self.dropout = nn.Dropout(dropout_prob)
            hidden_size = config.d_model

            self.family_head = nn.Linear(hidden_size, len(family2id))
            self.operator_head = nn.Linear(hidden_size, len(operator_vocab))
            self.time_order_head = nn.Linear(hidden_size, 3)
            self.first_spatial_head = nn.Linear(hidden_size, 1)
            self.second_spatial_head = nn.Linear(hidden_size, 1)
            self.nonlinear_head = nn.Linear(hidden_size, 1)
            self.spatial_var_head = nn.Linear(hidden_size, 2)

            self.operator_loss_weight = args.operator_loss_weight
            self.structure_loss_weight = args.structure_loss_weight

            self.post_init()

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            operator_labels=None,
            time_order_labels=None,
            first_spatial_labels=None,
            second_spatial_labels=None,
            nonlinear_labels=None,
            spatial_var_labels=None,
        ):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.encoder_last_hidden_state

            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

            pooled = self.dropout(pooled)

            family_logits = self.family_head(pooled)
            operator_logits = self.operator_head(pooled)
            time_order_logits = self.time_order_head(pooled)
            first_spatial_logits = self.first_spatial_head(pooled).squeeze(-1)
            second_spatial_logits = self.second_spatial_head(pooled).squeeze(-1)
            nonlinear_logits = self.nonlinear_head(pooled).squeeze(-1)
            spatial_var_logits = self.spatial_var_head(pooled)

            loss = None
            if labels is not None:
                loss = F.cross_entropy(family_logits, labels)
                if operator_labels is not None:
                    loss = loss + self.operator_loss_weight * F.binary_cross_entropy_with_logits(
                        operator_logits, operator_labels
                    )
                if time_order_labels is not None:
                    loss = loss + self.structure_loss_weight * F.cross_entropy(
                        time_order_logits, time_order_labels
                    )
                if first_spatial_labels is not None:
                    loss = loss + self.structure_loss_weight * F.binary_cross_entropy_with_logits(
                        first_spatial_logits, first_spatial_labels
                    )
                if second_spatial_labels is not None:
                    loss = loss + self.structure_loss_weight * F.binary_cross_entropy_with_logits(
                        second_spatial_logits, second_spatial_labels
                    )
                if nonlinear_labels is not None:
                    loss = loss + self.structure_loss_weight * F.binary_cross_entropy_with_logits(
                        nonlinear_logits, nonlinear_labels
                    )
                if spatial_var_labels is not None:
                    loss = loss + self.structure_loss_weight * F.cross_entropy(
                        spatial_var_logits, spatial_var_labels
                    )

            return {
                "loss": loss,
                "logits": family_logits,
                "operator_logits": operator_logits,
                "time_order_logits": time_order_logits,
                "first_spatial_logits": first_spatial_logits,
                "second_spatial_logits": second_spatial_logits,
                "nonlinear_logits": nonlinear_logits,
                "spatial_var_logits": spatial_var_logits,
            }

    def evaluate_examples(model, tokenizer, examples: list[dict[str, Any]]) -> MetricBundle:
        dataset = PDEMultitaskDataset(examples, tokenizer, args.max_input_length)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        model.eval()
        model.to(device)

        family_correct = 0
        family_total = 0
        time_order_correct = 0
        first_spatial_correct = 0
        second_spatial_correct = 0
        nonlinear_correct = 0
        spatial_var_correct = 0

        operator_pred_rows: list[list[int]] = []
        operator_gold_rows: list[list[int]] = []
        predicted_family_counts = {family: 0 for family in id2family.values()}

        with torch.no_grad():
            for batch in dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                family_pred = outputs["logits"].argmax(dim=-1)
                for pred_idx in family_pred.cpu().tolist():
                    predicted_family_counts[id2family[pred_idx]] += 1
                valid_family_mask = batch["labels"] >= 0
                family_total += int(valid_family_mask.sum().item())
                if valid_family_mask.any():
                    family_correct += int(
                        (family_pred[valid_family_mask] == batch["labels"][valid_family_mask])
                        .sum()
                        .item()
                    )

                time_order_pred = outputs["time_order_logits"].argmax(dim=-1)
                time_order_correct += int(
                    (time_order_pred == batch["time_order_labels"]).sum().item()
                )

                first_spatial_pred = (outputs["first_spatial_logits"].sigmoid() >= 0.5).long()
                first_spatial_correct += int(
                    (first_spatial_pred == batch["first_spatial_labels"].long()).sum().item()
                )

                second_spatial_pred = (
                    outputs["second_spatial_logits"].sigmoid() >= 0.5
                ).long()
                second_spatial_correct += int(
                    (second_spatial_pred == batch["second_spatial_labels"].long()).sum().item()
                )

                nonlinear_pred = (outputs["nonlinear_logits"].sigmoid() >= 0.5).long()
                nonlinear_correct += int(
                    (nonlinear_pred == batch["nonlinear_labels"].long()).sum().item()
                )

                spatial_var_pred = outputs["spatial_var_logits"].argmax(dim=-1)
                spatial_var_correct += int(
                    (spatial_var_pred == batch["spatial_var_labels"]).sum().item()
                )

                operator_pred = (outputs["operator_logits"].sigmoid() >= 0.5).long()
                operator_pred_rows.extend(operator_pred.cpu().tolist())
                operator_gold_rows.extend(batch["operator_labels"].long().cpu().tolist())

        n = len(dataset)
        top_predicted_family = max(
            predicted_family_counts, key=predicted_family_counts.get
        )
        predicted_family_distribution = {
            family: count / n for family, count in predicted_family_counts.items()
        }
        return MetricBundle(
            family_accuracy=(family_correct / family_total) if family_total else None,
            operator_micro_f1=compute_micro_f1(operator_pred_rows, operator_gold_rows),
            time_order_accuracy=time_order_correct / n,
            first_spatial_accuracy=first_spatial_correct / n,
            second_spatial_accuracy=second_spatial_correct / n,
            nonlinear_accuracy=nonlinear_correct / n,
            spatial_var_accuracy=spatial_var_correct / n,
            family_accuracy_support=family_total,
            top_predicted_family=top_predicted_family,
            predicted_family_distribution=predicted_family_distribution,
        )

    output_dir_name = (
        f"{args.train_mode}_{args.train_dialect}" if args.train_mode == "single" else "mixed"
    )
    if args.skip_train:
        output_dir_name = f"{output_dir_name}_no_fine_tune"
    output_root = Path(args.output_root) / args.split_mode / output_dir_name
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = BartTokenizer.from_pretrained(tokenizer_source_for_output(output_root, args.model_name))
    if not args.no_dialect_token:
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    f"[DIALECT_{dialect.upper()}]" for dialect in DIALECTS
                ]
            }
        )

    device = resolve_device(args, torch)
    print(f"Training device: {device}")
    set_random_seed(args.seed, torch)

    train_dataset = None
    val_dataset = None

    checkpoint = (
        find_latest_checkpoint(output_root)
        if not args.force_retrain and not args.skip_train
        else None
    )
    if checkpoint is not None:
        print(f"Loading checkpoint: {checkpoint}")
        model = BartPDEMultitaskClassifier.from_pretrained(checkpoint)
        tokenizer = BartTokenizer.from_pretrained(
            tokenizer_source_for_output(output_root, args.model_name)
        )
    else:
        model = BartPDEMultitaskClassifier.from_pretrained(args.model_name)
        model.resize_token_embeddings(len(tokenizer))
        if args.skip_train:
            print(
                "Skipping fine-tuning. Evaluating the pretrained BART backbone with "
                "randomly initialized multitask heads."
            )
        else:
            train_dataset = PDEMultitaskDataset(train_examples, tokenizer, args.max_input_length)
            val_dataset = PDEMultitaskDataset(val_examples, tokenizer, args.max_input_length)
            training_args = make_training_arguments(args, output_root, stack, device)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )
            trainer.train()
            trainer.save_model(str(output_root))
            tokenizer.save_pretrained(str(output_root))

    val_metrics = {}
    test_metrics = {}
    for dialect in args.eval_dialects:
        print(f"Running evaluation for dialect={dialect}")
        val_eval_examples = build_examples(
            val_data,
            train_mode="single",
            dialects=[dialect],
            args=args,
            family2id=family2id,
            operator2id=operator2id,
            single_dialect=dialect,
        )
        test_eval_examples = build_examples(
            test_data,
            train_mode="single",
            dialects=[dialect],
            args=args,
            family2id=family2id,
            operator2id=operator2id,
            single_dialect=dialect,
        )
        val_metrics[dialect] = evaluate_examples(model, tokenizer, val_eval_examples).to_dict()
        test_metrics[dialect] = evaluate_examples(model, tokenizer, test_eval_examples).to_dict()

        family_acc = test_metrics[dialect]["family_accuracy"]
        family_acc_text = f"{family_acc:.2%}" if family_acc is not None else "n/a"
        print(
            f"Eval on {dialect.upper():7s} | "
            f"family_acc={family_acc_text} | "
            f"operator_f1={test_metrics[dialect]['operator_micro_f1']:.2%} | "
            f"structure_acc={test_metrics[dialect]['structure_accuracy']:.2%}"
        )

    held_out_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for held_out_path in args.held_out_datasets:
        held_out_name = Path(held_out_path).stem
        held_out_data = load_jsonl(held_out_path)
        unseen_families = sorted(
            {
                instance["labels"]["behavioral"]
                for instance in held_out_data
                if instance["labels"]["behavioral"] not in family2id
            }
        )
        if unseen_families:
            print(
                f"Held-out dataset {held_out_name} contains unseen families {unseen_families}. "
                "Family accuracy will be reported as n/a for those examples."
            )

        held_out_metrics[held_out_name] = {}
        for dialect in args.eval_dialects:
            held_out_examples = build_examples(
                held_out_data,
                train_mode="single",
                dialects=[dialect],
                args=args,
                family2id=family2id,
                operator2id=operator2id,
                single_dialect=dialect,
                allow_unknown_families=True,
            )
            held_out_result = evaluate_examples(model, tokenizer, held_out_examples).to_dict()
            held_out_metrics[held_out_name][dialect] = held_out_result

            held_out_family_acc = held_out_result["family_accuracy"]
            held_out_family_text = (
                f"{held_out_family_acc:.2%}" if held_out_family_acc is not None else "n/a"
            )
            print(
                f"Held-out {held_out_name:24s} | dialect={dialect.upper():7s} | "
                f"family_acc={held_out_family_text} | "
                f"operator_f1={held_out_result['operator_micro_f1']:.2%} | "
                f"structure_acc={held_out_result['structure_accuracy']:.2%} | "
                f"top_family={held_out_result['top_predicted_family']}"
            )

    summary = {
        "config": vars(args),
        "family_vocab": id2family,
        "operator_vocab": operator_vocab,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    if held_out_metrics:
        summary["held_out_metrics"] = held_out_metrics
    summary_path = output_root / "multitask_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved multitask metrics to {summary_path}")


if __name__ == "__main__":
    main()
