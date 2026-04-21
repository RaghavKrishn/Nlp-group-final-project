from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dialect_utils import DIALECTS, derive_structure_targets, load_jsonl


DEFAULT_DATASETS = [
    "varied_data_generation/held_out_kleingordon.jsonl",
    "varied_data_generation/held_out_reactiondiffusion.jsonl",
    "varied_data_generation/held_out_beam.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained multitask BART model on held-out PDE families. "
            "Family accuracy is undefined because the held-out labels are unseen during training, "
            "so this script reports fallback family predictions plus operator/structure transfer."
        )
    )
    parser.add_argument(
        "--model-dir",
        default="outputs/bart_multitask_pde/instance/mixed",
        help="Directory containing the saved multitask BART model and multitask_metrics.json.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Held-out dataset JSONLs to evaluate.",
    )
    parser.add_argument(
        "--eval-dialects",
        nargs="+",
        default=None,
        choices=list(DIALECTS),
        help="Dialects to evaluate. Defaults to the training summary's eval dialects.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override eval batch size. Defaults to the training summary batch size.",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=None,
        help="Override max input length. Defaults to the training summary max input length.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Where to save the held-out evaluation summary JSON.",
    )
    return parser.parse_args()


def import_stack():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import BartModel, BartPreTrainedModel, BartTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing evaluation dependencies. Install at least transformers and torch "
            "in the environment that will run this script."
        ) from exc

    return {
        "torch": torch,
        "nn": nn,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "BartModel": BartModel,
        "BartPreTrainedModel": BartPreTrainedModel,
        "BartTokenizer": BartTokenizer,
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


def format_input_text(
    text: str,
    dialect: str,
    task_prefix: str,
    use_dialect_token: bool,
) -> str:
    pieces = []
    if use_dialect_token:
        pieces.append(f"[DIALECT_{dialect.upper()}]")
    if task_prefix:
        pieces.append(task_prefix.strip())
    pieces.append(text)
    return " ".join(pieces)


def build_examples(
    instances: list[dict],
    dialect: str,
    operator2id: dict[str, int],
    task_prefix: str,
    use_dialect_token: bool,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    for instance in instances:
        structure = derive_structure_targets(instance)
        operator_targets = [0.0] * len(operator2id)
        for operator in instance["labels"]["operators"]:
            operator_targets[operator2id[operator.lower()]] = 1.0

        examples.append(
            {
                "text": format_input_text(
                    instance["dialects"][dialect],
                    dialect=dialect,
                    task_prefix=task_prefix,
                    use_dialect_token=use_dialect_token,
                ),
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


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(
            f"Model directory not found: {model_dir}\n"
            "Run the multitask BART training cells first so the saved model exists."
        )

    summary_path = model_dir / "multitask_metrics.json"
    if not summary_path.exists():
        raise SystemExit(
            f"Training summary not found: {summary_path}\n"
            "Run the multitask BART training/evaluation cells first."
        )

    summary = json.loads(summary_path.read_text())
    train_config = summary.get("config", {})
    id2family = {int(key): value for key, value in summary["family_vocab"].items()}
    operator_vocab = summary["operator_vocab"]
    operator2id = {operator: idx for idx, operator in enumerate(operator_vocab)}

    eval_dialects = args.eval_dialects or train_config.get("eval_dialects", list(DIALECTS))
    max_input_length = args.max_input_length or train_config.get("max_input_length", 160)
    batch_size = args.batch_size or train_config.get("batch_size", 16)
    task_prefix = train_config.get("task_prefix", "classify PDE family and infer structure:")
    use_dialect_token = not train_config.get("no_dialect_token", False)

    stack = import_stack()
    torch = stack["torch"]
    nn = stack["nn"]
    DataLoader = stack["DataLoader"]
    Dataset = stack["Dataset"]
    BartModel = stack["BartModel"]
    BartPreTrainedModel = stack["BartPreTrainedModel"]
    BartTokenizer = stack["BartTokenizer"]

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
            return {
                "input_ids": torch.tensor(self.encodings["input_ids"][index], dtype=torch.long),
                "attention_mask": torch.tensor(
                    self.encodings["attention_mask"][index], dtype=torch.long
                ),
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

            self.family_head = nn.Linear(hidden_size, len(id2family))
            self.operator_head = nn.Linear(hidden_size, len(operator_vocab))
            self.time_order_head = nn.Linear(hidden_size, 3)
            self.first_spatial_head = nn.Linear(hidden_size, 1)
            self.second_spatial_head = nn.Linear(hidden_size, 1)
            self.nonlinear_head = nn.Linear(hidden_size, 1)
            self.spatial_var_head = nn.Linear(hidden_size, 2)

            self.post_init()

        def forward(self, input_ids=None, attention_mask=None):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.encoder_last_hidden_state

            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

            pooled = self.dropout(pooled)

            return {
                "logits": self.family_head(pooled),
                "operator_logits": self.operator_head(pooled),
                "time_order_logits": self.time_order_head(pooled),
                "first_spatial_logits": self.first_spatial_head(pooled).squeeze(-1),
                "second_spatial_logits": self.second_spatial_head(pooled).squeeze(-1),
                "nonlinear_logits": self.nonlinear_head(pooled).squeeze(-1),
                "spatial_var_logits": self.spatial_var_head(pooled),
            }

    def evaluate_examples(model, tokenizer, examples: list[dict[str, Any]]) -> dict[str, Any]:
        dataset = PDEMultitaskDataset(examples, tokenizer, max_input_length)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        model.eval()
        model.to(device)

        family_counts: Counter[int] = Counter()
        confidence_sum = 0.0
        n = len(dataset)

        time_order_correct = 0
        first_spatial_correct = 0
        second_spatial_correct = 0
        nonlinear_correct = 0
        spatial_var_correct = 0

        operator_pred_rows: list[list[int]] = []
        operator_gold_rows: list[list[int]] = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                family_probs = outputs["logits"].softmax(dim=-1)
                top_confidence, family_pred = family_probs.max(dim=-1)
                family_counts.update(family_pred.cpu().tolist())
                confidence_sum += float(top_confidence.sum().item())

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

        majority_family_id, majority_count = family_counts.most_common(1)[0]
        family_distribution = {
            id2family[idx]: count / n for idx, count in sorted(family_counts.items())
        }

        time_order_accuracy = time_order_correct / n
        first_spatial_accuracy = first_spatial_correct / n
        second_spatial_accuracy = second_spatial_correct / n
        nonlinear_accuracy = nonlinear_correct / n
        spatial_var_accuracy = spatial_var_correct / n
        structure_accuracy = (
            time_order_accuracy
            + first_spatial_accuracy
            + second_spatial_accuracy
            + nonlinear_accuracy
            + spatial_var_accuracy
        ) / 5.0

        return {
            "n": n,
            "majority_predicted_family": id2family[majority_family_id],
            "majority_prediction_rate": majority_count / n,
            "mean_top1_confidence": confidence_sum / n,
            "family_prediction_distribution": family_distribution,
            "operator_micro_f1": compute_micro_f1(operator_pred_rows, operator_gold_rows),
            "time_order_accuracy": time_order_accuracy,
            "first_spatial_accuracy": first_spatial_accuracy,
            "second_spatial_accuracy": second_spatial_accuracy,
            "nonlinear_accuracy": nonlinear_accuracy,
            "spatial_var_accuracy": spatial_var_accuracy,
            "structure_accuracy": structure_accuracy,
        }

    device = resolve_device(args, torch)
    print(f"Evaluation device: {device}")

    tokenizer = BartTokenizer.from_pretrained(str(model_dir))
    model = BartPDEMultitaskClassifier.from_pretrained(str(model_dir))

    records = []
    for dataset_name in args.datasets:
        dataset_path = Path(dataset_name)
        if not dataset_path.exists():
            raise SystemExit(f"Held-out dataset not found: {dataset_path}")

        instances = load_jsonl(dataset_path)
        if not instances:
            raise SystemExit(f"Held-out dataset is empty: {dataset_path}")

        gold_family = instances[0]["family"]
        print(f"\nEvaluating held-out family: {gold_family} ({dataset_path})")
        for dialect in eval_dialects:
            examples = build_examples(
                instances,
                dialect=dialect,
                operator2id=operator2id,
                task_prefix=task_prefix,
                use_dialect_token=use_dialect_token,
            )
            metrics = evaluate_examples(model, tokenizer, examples)
            record = {
                "dataset_name": dataset_path.stem,
                "dataset_path": str(dataset_path),
                "gold_family": gold_family,
                "dialect": dialect,
                "family_accuracy": None,
                "family_accuracy_note": (
                    "Undefined for held-out families because the trained family head only "
                    "covers the five in-training PDE families."
                ),
                **metrics,
            }
            records.append(record)
            print(
                f"  dialect={dialect:7s} | "
                f"majority_seen_family={record['majority_predicted_family']:10s} | "
                f"operator_f1={record['operator_micro_f1']:.2%} | "
                f"structure_acc={record['structure_accuracy']:.2%}"
            )

    output = {
        "model_dir": str(model_dir),
        "source_summary_path": str(summary_path),
        "eval_dialects": eval_dialects,
        "records": records,
    }
    output_path = Path(args.output_json) if args.output_json else model_dir / "held_out_metrics.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved held-out evaluation to {output_path}")


if __name__ == "__main__":
    main()
