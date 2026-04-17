# main.py
# Generates two datasets:
#   dataset.jsonl         — 10,000 training instances (5 families x 2000)
#   held_out_dataset.jsonl — 2,000 held-out instances (KleinGordon only)
#                            used exclusively for zero-shot generalization testing

import json
import random
from dataset_builder import generate_instance

TRAIN_FAMILIES  = ["Heat", "Wave", "Burgers", "Laplace", "Advection"]
HELD_OUT_FAMILIES = ["KleinGordon", "ReactionDiffusion", "Beam"]

INSTANCES_PER_FAMILY = 2000
HELD_OUT_INSTANCES   = 1000  # 1000 per held-out family

SEED = 42
random.seed(SEED)


def generate_dataset(output_path, families, instances_per_family, held_out=False):
    all_instances = []
    for family in families:
        for _ in range(instances_per_family):
            instance = generate_instance(family, held_out=held_out)
            all_instances.append(instance)

    random.shuffle(all_instances)

    with open(output_path, "w") as f:
        for instance in all_instances:
            f.write(json.dumps(instance) + "\n")

    print(f"Written {len(all_instances)} instances to {output_path}")


if __name__ == "__main__":
    # Training dataset
    generate_dataset(
        output_path="dataset.jsonl",
        families=TRAIN_FAMILIES,
        instances_per_family=INSTANCES_PER_FAMILY,
        held_out=False,
    )

    # Zero-shot held-out datasets — one file per family, never seen during training
    for family in HELD_OUT_FAMILIES:
        generate_dataset(
            output_path=f"held_out_{family.lower()}.jsonl",
            families=[family],
            instances_per_family=HELD_OUT_INSTANCES,
            held_out=True,
        )
