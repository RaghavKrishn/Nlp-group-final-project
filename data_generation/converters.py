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

        return Eq(Derivative(u(t, x), t) + u(t, x) * Derivative(u(t, x), x), nu * Derivative(u(t, x), x, 2))

    if family == "Laplace":

        return Eq(Derivative(u(x, y), x, 2) + Derivative(u(x, y), y, 2), 0)

    if family == "Advection":

        c = coeffs["c"]

        return Eq(Derivative(u(t, x), t) +
                  c * Derivative(u(t, x), x), 0)


def to_latex(expr):

    return latex(expr)


def to_prefix(expr):

    from sympy import Eq, Add, Mul, Derivative

    if expr.is_Number:
        return str(expr)

    if expr.is_Symbol:
        return str(expr)

    if expr.func == Eq:
        return f"=({to_prefix(expr.lhs)}, {to_prefix(expr.rhs)})"

    if expr.func == Add:
        a, b = expr.args
        return f"+({to_prefix(a)}, {to_prefix(b)})"

    if expr.func == Mul:
        a, b = expr.args
        return f"*({to_prefix(a)}, {to_prefix(b)})"

    if expr.func == Derivative:

        inner = to_prefix(expr.args[0])
        var = str(expr.args[1][0])

        return f"d({inner},{var})"

    return str(expr)

def to_postfix(expr):

    from sympy import Eq, Add, Mul, Derivative

    if expr.is_Number:
        return str(expr)

    if expr.is_Symbol:
        return str(expr)

    if expr.func == Eq:
        return f"{to_postfix(expr.lhs)} {to_postfix(expr.rhs)} ="

    if expr.func == Add:
        a, b = expr.args
        return f"{to_postfix(a)} {to_postfix(b)} +"

    if expr.func == Mul:
        a, b = expr.args
        return f"{to_postfix(a)} {to_postfix(b)} *"

    if expr.func == Derivative:

        inner = to_postfix(expr.args[0])
        var = str(expr.args[1][0])

        return f"{inner} {var} d"

    return str(expr)