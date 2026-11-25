from sympy import Eq, Expr, Function, Symbol, exp, integrate, simplify

from ..exceptions import MethodNotApplicableError
from ..utils import substitute_function
from .first_order import solve_exact


def find_integrating_factor_x(
    m: Expr, n: Expr, variable: Symbol, function: Function
) -> Expr:
    """Find an integrating factor mu(x) that makes M + N*dy/dx = 0 exact.

    Parameters
    ----------
    m : sympy.Expr
        `M(x, y)`, from writing the ODE as
        `M(x, y) + N(x, y) * dy/dx = 0`.
    n : sympy.Expr
        `N(x, y)`, from the same form.
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    sympy.Expr
        `mu(x)`, such that `mu(x)*M + mu(x)*N*dy/dx = 0` is exact.
    """
    m_substituted, dummy = substitute_function(m, variable, function)
    n_substituted, _ = substitute_function(n, variable, function)

    expr = simplify(
        (m_substituted.diff(dummy) - n_substituted.diff(variable)) / n_substituted
    )
    if expr.has(dummy):
        raise MethodNotApplicableError(
            "no integrating factor of the form mu(x) exists",
            method="find_integrating_factor_x",
            reason="(dM/dy - dN/dx) / N still depends on y",
        )

    return exp(integrate(expr, variable))


def find_integrating_factor_y(
    m: Expr, n: Expr, variable: Symbol, function: Function
) -> Expr:
    """Find an integrating factor mu(y) that makes M + N*dy/dx = 0 exact.

    Parameters
    ----------
    m : sympy.Expr
        `M(x, y)`, from writing the ODE as
        `M(x, y) + N(x, y) * dy/dx = 0`.
    n : sympy.Expr
        `N(x, y)`, from the same form.
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    sympy.Expr
        `mu(y)`, as a function of `function(variable)`, such that
        `mu(y)*M + mu(y)*N*dy/dx = 0` is exact.
    """
    m_substituted, dummy = substitute_function(m, variable, function)
    n_substituted, _ = substitute_function(n, variable, function)

    expr = simplify(
        (n_substituted.diff(variable) - m_substituted.diff(dummy)) / m_substituted
    )
    if expr.has(variable):
        raise MethodNotApplicableError(
            "no integrating factor of the form mu(y) exists",
            method="find_integrating_factor_y",
            reason="(dN/dx - dM/dy) / M still depends on x",
        )

    mu = exp(integrate(expr, dummy))
    return mu.subs(dummy, function(variable))


def solve_with_integrating_factor(
    m: Expr, n: Expr, variable: Symbol, function: Function
) -> Eq:
    """Solve M(x, y) + N(x, y)*dy/dx = 0 by finding an integrating factor.

    Tries an integrating factor of the form mu(x) first, then mu(y),
    then delegates to `solve_exact` on the resulting exact equation.

    Parameters
    ----------
    m : sympy.Expr
        `M(x, y)`, from writing the ODE as
        `M(x, y) + N(x, y) * dy/dx = 0`.
    n : sympy.Expr
        `N(x, y)`, from the same form.
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    sympy.Eq
        The general solution, as an implicit equation.
    """
    try:
        mu = find_integrating_factor_x(m, n, variable, function)
    except MethodNotApplicableError:
        try:
            mu = find_integrating_factor_y(m, n, variable, function)
        except MethodNotApplicableError as error:
            raise MethodNotApplicableError(
                "no integrating factor of the form mu(x) or mu(y) exists",
                method="solve_with_integrating_factor",
                reason="neither mu(x) nor mu(y) makes the equation exact",
            ) from error

    return solve_exact(m * mu, n * mu, variable, function)
