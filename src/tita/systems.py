from sympy import Eq, Function, Matrix, Symbol
from sympy.solvers.deutils import ode_order

from .exceptions import MethodNotApplicableError
from .utils import linear_form


def reduce_to_system(
    ode: Eq, variable: Symbol, function: Function
) -> tuple[list[Eq], list[Function]]:
    """Reduce an order-n linear ODE to an equivalent first-order system.

    Introduces state functions x1 = function, x2 = function', ...,
    xn = function^(n-1), giving the companion-form system
    x1' = x2, ..., x(n-1)' = xn, xn' = (solved from `ode`).

    Parameters
    ----------
    ode : sympy.Eq
        The ODE, in any form, assumed linear in `function` and its
        derivatives.
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    tuple of (list of sympy.Eq, list of sympy.Function)
        The system of first-order equations `xi'(variable) = ...`,
        and the state functions `[x1, ..., xn]` used.
    """
    order = ode_order(ode, function(variable))
    poly, dummies = linear_form(ode, variable, function)

    if poly is None or poly.total_degree() > 1:
        raise MethodNotApplicableError(
            "ode is not linear in function and its derivatives",
            method="reduce_to_system",
            reason="ode is not affine in function and its derivatives",
        )

    leading_dummy = dummies[-1]
    leading_coefficient = poly.coeff_monomial(leading_dummy)
    other_terms = sum(
        poly.coeff_monomial(dummy) * dummy for dummy in dummies[:-1]
    ) + poly.nth(*([0] * len(dummies)))
    highest_derivative = -other_terms / leading_coefficient

    state_functions = [Function(f"{function}_{index + 1}") for index in range(order)]

    system = []
    for index in range(order - 1):
        system.append(
            Eq(
                state_functions[index](variable).diff(variable),
                state_functions[index + 1](variable),
            )
        )

    for power, dummy in enumerate(dummies[:-1]):
        highest_derivative = highest_derivative.subs(
            dummy, state_functions[power](variable)
        )
    system.append(Eq(state_functions[-1](variable).diff(variable), highest_derivative))

    return system, state_functions


def to_matrix_form(
    system: list[Eq], variable: Symbol, state_functions: list[Function]
) -> tuple[Matrix, Matrix]:
    """Write a first-order linear system X' = A*X + b in matrix form.

    Parameters
    ----------
    system : list of sympy.Eq
        The system of first-order equations `xi'(variable) = ...`,
        linear in `state_functions`.
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    state_functions : list of sympy.Function
        The unknown functions `[x1, ..., xn]` appearing in `system`.

    Returns
    -------
    tuple of (sympy.Matrix, sympy.Matrix)
        `A` and `b`, such that `X' = A*X + b` for the state vector
        `X = [x1(variable), ..., xn(variable)]`.
    """
    substitutions = {}
    plain_symbols = []
    for state_function in state_functions:
        symbol = Symbol(str(state_function))
        substitutions[state_function(variable)] = symbol
        plain_symbols.append(symbol)

    rows_a = []
    rows_b = []
    for equation in system:
        expr = equation.rhs.subs(substitutions)
        row = [expr.coeff(symbol) for symbol in plain_symbols]
        remainder = expr - sum(
            coefficient * symbol for coefficient, symbol in zip(row, plain_symbols)
        )
        rows_a.append(row)
        rows_b.append(remainder)

    return Matrix(rows_a), Matrix(rows_b)
