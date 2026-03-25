# sampler.py

import random
from pde_templates import PDE_TEMPLATES


def sample_coefficients(family):

    template = PDE_TEMPLATES[family]

    coeffs = {}

    if "coefficients" not in template:
        return coeffs

    for name in template["coefficients"]:

        low, high = template["range"]

        value = round(random.uniform(low, high), 2)

        coeffs[name] = value

    return coeffs