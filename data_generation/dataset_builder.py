# dataset_builder.py

from sampler import sample_coefficients
from converters import build_equation, to_latex, to_prefix, to_postfix
from labels import behavioral_label, operator_subset, structural_reasoning
from pde_templates import PDE_TEMPLATES


def generate_instance(family):

    coeffs = sample_coefficients(family)

    expr = build_equation(family, coeffs)

    dialects = {
        "latex": to_latex(expr),
        "prefix": to_prefix(expr),
        "postfix": to_postfix(expr)
    }

    template = PDE_TEMPLATES[family]["nl_template"]

    natural = template.format(**coeffs) if coeffs else template

    dialects["natural"] = natural

    labels = {
        "behavioral": behavioral_label(family),
        "operators": operator_subset(family),
        "reasoning": structural_reasoning(family, coeffs)
    }

    instance = {
        "family": family,
        "coefficients": coeffs,
        "dialects": dialects,
        "labels": labels
    }

    return instance