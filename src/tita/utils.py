from sympy import Eq, Expr, Function, Symbol, diff, solve, symbols
from sympy.solvers.deutils import ode_order

from .validators import validate_initial_conditions


def substitute_function(
    f: Expr, variable: Symbol, function: Function
) -> tuple[Expr, Symbol]:
    """Replace function(variable) in f with a plain symbol.

    Parameters
    ----------
    f : sympy.Expr
        An expression containing `function(variable)` (e.g. `y(x)`).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    tuple of (sympy.Expr, sympy.Symbol)
        The expression with `function(variable)` replaced, and the
        plain symbol used for the replacement.
    """
    dummy = Symbol(str(function))
    substituted = f.subs(function(variable), dummy)
    return substituted, dummy


def linear_form(ode: Eq, variable: Symbol, function: Function):
    """Rewrite an ODE as a polynomial in function and its derivatives.

    Parameters
    ----------
    ode : sympy.Eq
        The ODE, in any form (any order).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    tuple of (sympy.Poly or None, tuple of sympy.Symbol)
        The polynomial representation of `ode.lhs - ode.rhs` (or
        `None` if that expression is not polynomial in the unknown
        function and its derivatives), and the plain dummy symbols
        used for the function and each of its derivatives, in order
        (the dummy for the function itself first).
    """
    order = ode_order(ode, function(variable))
    derivatives = [function(variable).diff(variable, k) for k in range(order + 1)]
    dummies = symbols(f"d0:{order + 1}")

    expr = ode.lhs - ode.rhs
    for derivative, dummy in zip(reversed(derivatives), reversed(dummies)):
        expr = expr.subs(derivative, dummy)

    return expr.as_poly(*dummies), dummies


def solve_integration_constants(
    general_solution: Expr,
    constants: list[Symbol],
    variable: Symbol,
    initial_conditions: list[tuple[int, float, float]],
) -> Expr:
    """Solve for the arbitrary constants of a general ODE solution.

    Parameters
    ----------
    general_solution : sympy.Expr
        The general solution of the ODE, containing the arbitrary
        constants to solve for.
    constants : list of sympy.Symbol
        The arbitrary constants appearing in `general_solution`.
    variable : sympy.Symbol
        The independent variable of the ODE.
    initial_conditions : list of tuple of (int, float, float)
        Each tuple is `(order, point, value)`, meaning the `order`-th
        derivative of the solution evaluated at `point` equals `value`.

    Returns
    -------
    sympy.Expr
        The particular solution, with constants replaced by their
        solved values.
    """
    validate_initial_conditions(initial_conditions, len(constants))

    equations = []
    for order, point, value in initial_conditions:
        derivative = diff(general_solution, variable, order)
        value_at_point = derivative.subs(variable, point)
        equations.append(Eq(value_at_point, value))

    solved_constants = solve(equations, constants)

    return general_solution.subs(solved_constants)
