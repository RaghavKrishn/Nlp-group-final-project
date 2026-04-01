# sampler.py

import random
from pde_templates import PDE_TEMPLATES, HELD_OUT_TEMPLATES


def sample_coefficients(family, held_out=False):
    templates = HELD_OUT_TEMPLATES if held_out else PDE_TEMPLATES
    coeff_names = templates[family]["coefficients"]
    if not coeff_names:
        return {}
    low, high = templates[family]["range"]
    return {name: round(random.uniform(low, high), 2) for name in coeff_names}
