# pde_templates.py

PDE_TEMPLATES = {

    "Heat": {
        "coefficients": ["alpha"],
        "range": (0.1, 2.0),

        "nl_template":
        "The time derivative of u equals {alpha} times the second spatial derivative of u.",

        "reasoning":
        "Contains a first-order time derivative and second-order spatial derivative, indicating diffusion with coefficient {alpha}.",

        "operators": ["exp", "polynomial"]
    },

    "Wave": {
        "coefficients": ["c"],
        "range": (0.1, 3.0),

        "nl_template":
        "The second time derivative of u equals {c} squared times the second spatial derivative of u.",

        "reasoning":
        "Contains a second-order time derivative and second-order spatial derivative, indicating wave propagation with speed parameter {c}.",

        "operators": ["sin", "cos", "polynomial"]
    },

    "Burgers": {
        "coefficients": ["nu"],
        "range": (0.05, 1.0),

        "nl_template":
        "The time derivative of u plus u times its spatial derivative equals {nu} times the second spatial derivative of u.",

        "reasoning":
        "Contains nonlinear convection term u*u_x and diffusion term with coefficient {nu}.",

        "operators": ["tanh", "polynomial"]
    },

    "Laplace": {
        "coefficients": [],

        "nl_template":
        "The sum of the second spatial derivatives in x and y equals zero.",

        "reasoning":
        "Contains second-order spatial derivatives in x and y, forming Laplace's equation.",

        "operators": ["sin", "cos", "exp", "polynomial"]
    },

    "Advection": {
        "coefficients": ["c"],
        "range": (0.1, 2.0),

        "nl_template":
        "The time derivative of u plus {c} times its spatial derivative equals zero.",

        "reasoning":
        "Contains first-order time and spatial derivatives representing pure transport with velocity {c}.",

        "operators": ["exp", "polynomial"]
    }

}