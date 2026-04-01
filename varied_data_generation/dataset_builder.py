# dataset_builder.py

import random
from sampler import sample_coefficients
from converters import build_equation, to_latex, to_prefix, to_postfix
from pde_templates import PDE_TEMPLATES, HELD_OUT_TEMPLATES


def generate_instance(family, held_out=False):
    templates = HELD_OUT_TEMPLATES if held_out else PDE_TEMPLATES

    coeffs = sample_coefficients(family, held_out=held_out)

    expr = build_equation(family, coeffs)

    dialects = {
        "latex":   to_latex(expr),
        "prefix":  to_prefix(expr),
        "postfix": to_postfix(expr),
    }

    # Randomly pick one NL template and one reasoning template independently.
    # This is the key change from the original pipeline — the model can no longer
    # memorize a single surface pattern per family.
    nl_template = random.choice(templates[family]["nl_templates"])
    reasoning_template = random.choice(templates[family]["reasoning_templates"])

    dialects["natural"] = nl_template.format(**coeffs) if coeffs else nl_template

    reasoning = reasoning_template.format(**coeffs) if coeffs else reasoning_template

    labels = {
        "behavioral": family,
        "operators":  templates[family]["operators"],
        "reasoning":  reasoning,
    }

    return {
        "family":       family,
        "coefficients": coeffs,
        "dialects":     dialects,
        "labels":       labels,
    }
