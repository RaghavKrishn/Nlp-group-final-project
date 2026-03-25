# main.py

import json
import random

from dataset_builder import generate_instance
from pde_templates import PDE_TEMPLATES


DATASET_SIZE_PER_FAMILY = 1000


def main():

    dataset = []

    families = list(PDE_TEMPLATES.keys())

    for family in families:

        for _ in range(DATASET_SIZE_PER_FAMILY):

            instance = generate_instance(family)

            dataset.append(instance)

    random.shuffle(dataset)

    with open("dataset.jsonl", "w") as f:

        for item in dataset:

            f.write(json.dumps(item) + "\n")

    print("Dataset generated:", len(dataset))


if __name__ == "__main__":

    main()