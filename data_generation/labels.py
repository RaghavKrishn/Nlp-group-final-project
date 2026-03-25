# labels.py

from pde_templates import PDE_TEMPLATES


def behavioral_label(family):

    return family


def operator_subset(family):

    return PDE_TEMPLATES[family]["operators"]


def structural_reasoning(family, coeffs):

    template = PDE_TEMPLATES[family]["reasoning"]

    if coeffs:
        return template.format(**coeffs)

    return template