from sympy import Eq, Expr, Function, Symbol, solve

from .exceptions import MethodNotApplicableError


def isolate_derivative(ode: Eq, variable: Symbol, function: Function) -> Expr:
    """Isolate dy/dx and return it as an explicit expression f(x, y).

    Parameters
    ----------
    ode : sympy.Eq
        The first-order ODE, in any form (the derivative term can be
        on either side, mixed with other terms, etc).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    sympy.Expr
        The right-hand side `f(x, y)` of the explicit form
        `dy/dx = f(x, y)`.
    """
    derivative = function(variable).diff(variable)
    expr = ode.lhs - ode.rhs
    solutions = solve(expr, derivative)

    if len(solutions) != 1:
        raise MethodNotApplicableError(
            "ODE cannot be rewritten as dy/dx = f(x, y)",
            method="isolate_derivative",
            reason=f"solve() returned {len(solutions)} solutions for dy/dx",
        )

    return solutions[0]


def extract_m_n(ode: Eq, variable: Symbol, function: Function) -> tuple[Expr, Expr]:
    """Split the ODE into M(x, y) + N(x, y) * dy/dx = 0.

    Parameters
    ----------
    ode : sympy.Eq
        The first-order ODE, in any form (the derivative term can be
        on either side, mixed with other terms, etc).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    tuple of (sympy.Expr, sympy.Expr)
        `M(x, y)` and `N(x, y)`, such that
        `M(x, y) + N(x, y) * dy/dx = 0` is equivalent to `ode`.
    """
    derivative = function(variable).diff(variable)
    expr = ode.lhs - ode.rhs
    poly = expr.as_poly(derivative)

    if poly is None or poly.degree() > 1:
        raise MethodNotApplicableError(
            "ODE cannot be written as M(x, y) + N(x, y) * dy/dx = 0",
            method="extract_m_n",
            reason="expression is not affine in dy/dx",
        )

    n = expr.coeff(derivative, 1)
    m = expr.coeff(derivative, 0)
    return m, n
