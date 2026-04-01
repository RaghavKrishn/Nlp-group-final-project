# converters.py

from sympy import symbols, Function, Eq, Derivative, latex

x, y, t = symbols('x y t')
u = Function('u')


def build_equation(family, coeffs):

    if family == "Heat":
        alpha = coeffs["alpha"]
        return Eq(Derivative(u(t, x), t),
                  alpha * Derivative(u(t, x), x, 2))

    if family == "Wave":
        c = coeffs["c"]
        return Eq(Derivative(u(t, x), t, 2),
                  c**2 * Derivative(u(t, x), x, 2))

    if family == "Burgers":
        nu = coeffs["nu"]
        return Eq(Derivative(u(t, x), t) + u(t, x) * Derivative(u(t, x), x),
                  nu * Derivative(u(t, x), x, 2))

    if family == "Laplace":
        return Eq(Derivative(u(x, y), x, 2) + Derivative(u(x, y), y, 2), 0)

    if family == "Advection":
        c = coeffs["c"]
        return Eq(Derivative(u(t, x), t) + c * Derivative(u(t, x), x), 0)

    if family == "KleinGordon":
        m = coeffs["m"]
        return Eq(Derivative(u(t, x), t, 2) - Derivative(u(t, x), x, 2) + m**2 * u(t, x), 0)

    if family == "ReactionDiffusion":
        D = coeffs["D"]
        r = coeffs["r"]
        return Eq(Derivative(u(t, x), t),
                  D * Derivative(u(t, x), x, 2) + r * u(t, x) * (1 - u(t, x)))

    if family == "Beam":
        b = coeffs["b"]
        return Eq(Derivative(u(t, x), t, 2) + b * Derivative(u(t, x), x, 4), 0)

    raise ValueError(f"Unknown family: {family}")


def to_latex(expr):
    return latex(expr)


def to_prefix(expr):
    from sympy import Eq, Add, Mul, Derivative, Pow

    if expr.is_Number:
        return str(expr)

    if expr.is_Symbol:
        return str(expr)

    if expr.func == Eq:
        return f"=({to_prefix(expr.lhs)}, {to_prefix(expr.rhs)})"

    if expr.func == Add:
        args = [to_prefix(a) for a in expr.args]
        result = args[0]
        for a in args[1:]:
            result = f"+({result}, {a})"
        return result

    if expr.func == Mul:
        args = [to_prefix(a) for a in expr.args]
        result = args[0]
        for a in args[1:]:
            result = f"*({result}, {a})"
        return result

    if expr.func == Pow:
        base = to_prefix(expr.args[0])
        exp = to_prefix(expr.args[1])
        return f"^({base}, {exp})"

    if expr.func == Derivative:
        inner = to_prefix(expr.args[0])
        var   = str(expr.args[1][0])
        order = int(expr.args[1][1]) if len(expr.args[1]) > 1 else 1
        # Apply d() once per order: d(d(u,x),x) for second-order
        result = inner
        for _ in range(order):
            result = f"d({result}, {var})"
        return result

    return str(expr)


def to_postfix(expr):
    from sympy import Eq, Add, Mul, Derivative, Pow

    if expr.is_Number:
        return str(expr)

    if expr.is_Symbol:
        return str(expr)

    if expr.func == Eq:
        return f"{to_postfix(expr.lhs)} {to_postfix(expr.rhs)} ="

    if expr.func == Add:
        args = [to_postfix(a) for a in expr.args]
        result = args[0]
        for a in args[1:]:
            result = f"{result} {a} +"
        return result

    if expr.func == Mul:
        args = [to_postfix(a) for a in expr.args]
        result = args[0]
        for a in args[1:]:
            result = f"{result} {a} *"
        return result

    if expr.func == Pow:
        base = to_postfix(expr.args[0])
        exp = to_postfix(expr.args[1])
        return f"{base} {exp} ^"

    if expr.func == Derivative:
        inner = to_postfix(expr.args[0])
        var   = str(expr.args[1][0])
        order = int(expr.args[1][1]) if len(expr.args[1]) > 1 else 1
        # Apply "var d" once per order: u x d x d for second-order
        result = inner
        for _ in range(order):
            result = f"{result} {var} d"
        return result

    return str(expr)
