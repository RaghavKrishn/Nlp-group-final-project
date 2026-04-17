# pde_templates.py
# Each family has multiple NL and reasoning templates.
# During generation one is picked at random — forces the model to learn
# structure, not surface patterns.

PDE_TEMPLATES = {

    "Heat": {
        "coefficients": ["alpha"],
        "range": (0.1, 3.0),

        "nl_templates": [
            "The time derivative of u equals {alpha} times the second spatial derivative of u.",
            "u changes in time at a rate proportional to {alpha} times its spatial curvature.",
            "The rate of change of u with respect to time is {alpha} multiplied by the second derivative of u with respect to x.",
            "Diffusion governs u: its temporal evolution equals {alpha} times its second-order spatial variation.",
            "u_t equals {alpha} times u_xx, describing how u spreads over time due to diffusion.",
            "The temporal rate of change of u is driven by {alpha} times the curvature of u in space.",
        ],

        "reasoning_templates": [
            "Contains a first-order time derivative and second-order spatial derivative, indicating diffusion with coefficient {alpha}.",
            "The equation has a first-order time derivative on the left and a second-order spatial derivative on the right scaled by {alpha}, characteristic of heat diffusion.",
            "Structure: u_t = {alpha} * u_xx. First-order in time, second-order in space — this is the heat equation with diffusivity {alpha}.",
            "A single time derivative balanced against a second spatial derivative scaled by {alpha} identifies this as diffusive transport.",
            "First-order temporal derivative and second-order spatial derivative with diffusion coefficient {alpha} — classic heat equation structure.",
            "One time derivative on the left; one second-order spatial derivative scaled by {alpha} on the right — the hallmark of heat diffusion.",
        ],

        "operators": ["exp", "polynomial"]
    },

    "Wave": {
        "coefficients": ["c"],
        "range": (0.1, 3.0),

        "nl_templates": [
            "The second time derivative of u equals {c} squared times the second spatial derivative of u.",
            "u oscillates such that its second temporal derivative equals {c}^2 times its second spatial derivative.",
            "The acceleration of u in time equals {c} squared times its spatial curvature.",
            "Wave propagation with speed {c}: the second-order time change equals {c}^2 times the second-order spatial change.",
            "u_tt equals {c}^2 times u_xx, governing oscillation at propagation speed {c}.",
            "The curvature of u in time is proportional to {c}^2 times its curvature in space.",
        ],

        "reasoning_templates": [
            "Contains a second-order time derivative and second-order spatial derivative, indicating wave propagation with speed parameter {c}.",
            "The equation is second-order in both time and space: u_tt = {c}^2 * u_xx, identifying it as a wave equation with wave speed {c}.",
            "Both left and right sides involve second-order derivatives — time and space respectively — scaled by {c}^2, marking this as wave behavior.",
            "Second-order temporal and spatial derivatives both present, with speed coefficient {c}, characteristic of the wave equation.",
            "u_tt and u_xx appear with scaling {c}^2; this second-order structure in both time and space signals wave propagation.",
            "Two time derivatives on the left, two spatial derivatives scaled by {c}^2 on the right — the defining pattern of a wave equation.",
        ],

        "operators": ["sin", "cos", "polynomial"]
    },

    "Burgers": {
        "coefficients": ["nu"],
        "range": (0.05, 1.5),

        "nl_templates": [
            "The time derivative of u plus u times its spatial derivative equals {nu} times the second spatial derivative of u.",
            "Nonlinear advection and diffusion: u_t plus u times u_x balances against {nu} times u_xx.",
            "The temporal change of u plus the nonlinear convection term u times u_x equals viscous diffusion scaled by {nu}.",
            "u evolves via self-advection: its time rate of change plus u times its own spatial gradient equals {nu} times its second spatial derivative.",
            "u_t plus the nonlinear term u*u_x equals {nu} times u_xx, capturing both nonlinear transport and diffusion.",
            "The sum of u_t and the product u times u_x equals {nu} times the second spatial derivative of u.",
        ],

        "reasoning_templates": [
            "Contains nonlinear convection term u*u_x and diffusion term with coefficient {nu}.",
            "The left side contains u_t and the nonlinear term u*u_x (convection); the right side is {nu}*u_xx (diffusion). This nonlinear structure characterizes Burgers equation.",
            "Nonlinear first-order spatial term u*u_x combined with second-order diffusion {nu}*u_xx identifies this as Burgers equation with viscosity {nu}.",
            "Presence of both a nonlinear convection term u*u_x and a second-order diffusion term scaled by {nu} — hallmarks of Burgers equation.",
            "The u*u_x term introduces nonlinearity; the {nu}*u_xx term provides diffusion. Together they define Burgers equation with viscosity {nu}.",
            "Nonlinear advection (u*u_x) plus linear diffusion ({nu}*u_xx) — this combination is uniquely characteristic of Burgers equation.",
        ],

        "operators": ["tanh", "polynomial"]
    },

    "Laplace": {
        "coefficients": [],

        "nl_templates": [
            "The sum of the second spatial derivatives in x and y equals zero.",
            "The second derivative of u in x plus the second derivative of u in y is zero.",
            "u has zero Laplacian: its spatial curvature in x and y sum to zero.",
            "Steady-state equilibrium: no time dependence; the second spatial derivatives in x and y cancel to zero.",
            "u_xx plus u_yy equals zero, indicating a steady-state distribution with no temporal evolution.",
            "The curvature of u in the x-direction plus the curvature in the y-direction equals zero.",
        ],

        "reasoning_templates": [
            "Contains second-order spatial derivatives in x and y, forming Laplace's equation.",
            "No time derivative present; the equation only involves u_xx + u_yy = 0, indicating steady-state equilibrium.",
            "Second-order spatial derivatives in both x and y summing to zero, with no temporal component, characterizes Laplace's equation.",
            "The absence of any time derivative and the presence of u_xx + u_yy = 0 identifies this as Laplace's equation.",
            "Only spatial second-order derivatives appear; their sum equals zero — this is Laplace's equation describing equilibrium.",
            "No u_t or u_tt term; only u_xx + u_yy = 0 — purely spatial, steady-state structure of Laplace's equation.",
        ],

        "operators": ["sin", "cos", "exp", "polynomial"]
    },

    "Advection": {
        "coefficients": ["c"],
        "range": (0.1, 3.0),

        "nl_templates": [
            "The time derivative of u plus {c} times its spatial derivative equals zero.",
            "u is transported at velocity {c}: its temporal change plus {c} times its spatial change equals zero.",
            "Pure transport: the rate of change of u in time plus {c} times the rate of change in space is zero.",
            "u propagates without diffusion: u_t plus {c} times u_x equals zero.",
            "The first-order time derivative plus {c} times the first-order spatial derivative of u equals zero, describing advection at speed {c}.",
            "u moves at constant speed {c}: the sum of u_t and {c} times u_x is zero.",
        ],

        "reasoning_templates": [
            "Contains first-order time and spatial derivatives representing pure transport with velocity {c}.",
            "The equation has a first-order time derivative and a first-order spatial derivative scaled by {c}, with no second-order terms — indicating pure advection.",
            "Only first-order derivatives appear: u_t and {c}*u_x, with no u_xx term, making this a pure transport equation with velocity {c}.",
            "Absence of second-order derivatives and presence of u_t plus {c}*u_x identifies this as the advection equation with speed {c}.",
            "First-order in both time and space, no diffusion term — u_t + {c}*u_x = 0 is characteristic of pure advective transport.",
            "No second-order spatial derivative; only u_t and {c}*u_x appear — this is advection, not diffusion, at speed {c}.",
        ],

        "operators": ["exp", "polynomial"]
    },
}

# ---------------------------------------------------------------------------
# HELD-OUT FAMILY: used only for zero-shot generalization testing.
# Never included in training data.
# Klein-Gordon: u_tt - u_xx + m^2 * u = 0
# Describes relativistic quantum fields; combines wave structure with a mass term.
# ---------------------------------------------------------------------------

HELD_OUT_TEMPLATES = {

    "KleinGordon": {
        "coefficients": ["m"],
        "range": (0.1, 2.0),

        "nl_templates": [
            "The second time derivative of u minus the second spatial derivative of u plus {m} squared times u equals zero.",
            "u_tt minus u_xx plus {m}^2 times u equals zero, describing a massive wave field.",
            "The wave operator applied to u plus a mass term {m}^2 times u equals zero.",
            "Second-order time and space derivatives of u differ by a mass correction: u_tt - u_xx + {m}^2 * u = 0.",
            "u evolves as a wave but with a restoring mass term: u_tt equals u_xx minus {m}^2 times u.",
            "The Klein-Gordon structure: temporal curvature minus spatial curvature plus {m}^2 times u equals zero.",
        ],

        "reasoning_templates": [
            "Contains second-order time and spatial derivatives with an additional linear mass term {m}^2 * u, characteristic of the Klein-Gordon equation.",
            "Like a wave equation but with an extra {m}^2 * u term — the mass term distinguishes Klein-Gordon from a pure wave equation.",
            "Second-order in both time and space (like wave), but the linear {m}^2 * u term introduces a mass correction unique to Klein-Gordon.",
            "u_tt and u_xx both appear (wave-like), plus a linear restoring term {m}^2 * u — this is the Klein-Gordon equation.",
            "The combination of second-order wave structure and linear mass term {m}^2 * u identifies this as Klein-Gordon.",
            "Wave-like second-order derivatives plus a scalar mass term {m}^2 * u — this additional term is the hallmark of Klein-Gordon over a standard wave equation.",
        ],

        "operators": ["sin", "cos", "exp", "polynomial"]
    },

    # -------------------------------------------------------------------------
    # Reaction-Diffusion: u_t = D*u_xx + r*u*(1-u)
    # Deliberately ambiguous — shares diffusion term with Heat and nonlinear
    # term structure with Burgers. Tests whether models understand both.
    # -------------------------------------------------------------------------
    "ReactionDiffusion": {
        "coefficients": ["D", "r"],
        "range": (0.1, 2.0),

        "nl_templates": [
            "The time derivative of u equals {D} times the second spatial derivative of u plus {r} times u times one minus u.",
            "u evolves through both diffusion and nonlinear reaction: u_t equals {D}*u_xx plus {r}*u*(1-u).",
            "Diffusion with rate {D} and logistic growth with rate {r}: the temporal change of u equals {D} times u_xx plus {r} times u times one minus u.",
            "u spreads spatially at rate {D} while growing logistically at rate {r}: u_t = {D}*u_xx + {r}*u*(1-u).",
            "The rate of change of u combines diffusion scaled by {D} with a nonlinear source term {r}*u*(1-u).",
            "u_t equals {D} times its second spatial derivative plus a logistic nonlinearity {r} times u times one minus u.",
        ],

        "reasoning_templates": [
            "Contains a first-order time derivative, a second-order spatial diffusion term scaled by {D}, and a nonlinear logistic source term {r}*u*(1-u).",
            "The {D}*u_xx term provides diffusion like the heat equation; the {r}*u*(1-u) term adds nonlinear reaction — together forming a reaction-diffusion equation.",
            "First-order in time, second-order in space for diffusion ({D}*u_xx), plus a nonlinear u*(1-u) source term scaled by {r} — characteristic of reaction-diffusion.",
            "Diffusion component {D}*u_xx (like heat equation) combined with logistic nonlinearity {r}*u*(1-u) distinguishes this as a reaction-diffusion equation.",
            "The equation has both a linear diffusion part ({D}*u_xx) and a nonlinear reaction part ({r}*u*(1-u)), making it a reaction-diffusion system.",
            "u_t driven by two terms: second-order spatial diffusion {D}*u_xx and nonlinear logistic growth {r}*u*(1-u) — this is the Fisher-KPP reaction-diffusion equation.",
        ],

        "operators": ["tanh", "exp", "polynomial"]
    },

    # -------------------------------------------------------------------------
    # Beam equation: u_tt + u_xxxx = 0
    # Fourth-order spatial derivative — completely unseen structure.
    # No training family has order > 2. Pure structural stress test.
    # -------------------------------------------------------------------------
    "Beam": {
        "coefficients": ["b"],
        "range": (0.1, 2.0),

        "nl_templates": [
            "The second time derivative of u plus {b} times the fourth spatial derivative of u equals zero.",
            "u_tt plus {b} times u_xxxx equals zero, describing elastic beam vibration.",
            "The acceleration of u in time plus {b} times its fourth-order spatial curvature equals zero.",
            "Beam vibration: the second-order time change of u plus {b} times the fourth-order spatial change equals zero.",
            "u evolves as a vibrating beam: u_tt plus {b} times the fourth derivative of u with respect to x equals zero.",
            "The temporal curvature of u plus {b} times the fourth-order spatial derivative equals zero, modeling elastic beam dynamics.",
        ],

        "reasoning_templates": [
            "Contains a second-order time derivative and a fourth-order spatial derivative scaled by {b}, characteristic of beam vibration equations.",
            "The u_tt term indicates wave-like temporal evolution; the {b}*u_xxxx term is a fourth-order spatial operator unique to beam/plate equations.",
            "Second-order in time like a wave equation, but fourth-order in space ({b}*u_xxxx) rather than second-order — this is a beam equation.",
            "The presence of a fourth-order spatial derivative {b}*u_xxxx with no lower-order spatial terms distinguishes this as an Euler-Bernoulli beam equation.",
            "u_tt paired with fourth-order spatial derivative {b}*u_xxxx — the fourth-order structure is the hallmark of elastic beam dynamics.",
            "Unlike wave (u_xx) or heat (u_xx), this equation has u_xxxx — a fourth-order spatial derivative scaled by {b} — indicating beam vibration.",
        ],

        "operators": ["sin", "cos", "polynomial"]
    },
}
