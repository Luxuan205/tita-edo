from sympy import Eq, Function, Symbol
from sympy.solvers.deutils import ode_order

from .analytic.first_order import (
    solve_bernoulli,
    solve_exact,
    solve_homogeneous,
    solve_linear_first_order,
    solve_separable,
)
from .analytic.second_order import solve_constant_coeff_homogeneous
from .classifier import classify, classify_second_order, is_exact, is_ordinary
from .exceptions import MethodNotApplicableError
from .parser import extract_m_n, isolate_derivative
from .validators import validate_order


def classify_ode(ode: Eq, variable: Symbol, function: Function) -> list[str]:
    """Classify a first- or second-order ODE by its type(s).

    Parameters
    ----------
    ode : sympy.Eq
        The ODE, in any form (the derivative term(s) can be on either
        side, mixed with other terms, etc).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    list of str
        The matching ODE type names: for order 1, see
        `classifier.classify` and `classifier.is_exact`; for order 2,
        see `classifier.classify_second_order`.
    """
    if not is_ordinary(ode):
        raise MethodNotApplicableError(
            "This project only classifies ordinary differential equations",
            method="classify_ode",
            reason="ode involves more than one independent variable (a PDE)",
        )

    order = ode_order(ode, function(variable))
    validate_order(order)
    if order not in (1, 2):
        raise MethodNotApplicableError(
            "This project only classifies first- and second-order ODEs",
            method="classify_ode",
            reason=f"detected order {order}",
        )

    if order == 2:
        return classify_second_order(ode, variable, function)

    f = isolate_derivative(ode, variable, function)
    types = classify(f, variable, function)

    m, n = extract_m_n(ode, variable, function)
    if is_exact(m, n, variable, function):
        types.append("exact")

    return types


def solve_ode(ode: Eq, variable: Symbol, function: Function) -> Eq:
    """Classify a first- or second-order ODE and solve it.

    For order 1, picks one applicable method by priority (linear,
    bernoulli, homogeneous, separable, exact) when more than one type
    matches. For order 2, only constant-coefficient homogeneous
    equations are solvable so far.

    Parameters
    ----------
    ode : sympy.Eq
        The ODE, in any form (the derivative term(s) can be on either
        side, mixed with other terms, etc).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    sympy.Eq
        The general solution found.
    """
    if not is_ordinary(ode):
        raise MethodNotApplicableError(
            "This project only solves ordinary differential equations",
            method="solve_ode",
            reason="ode involves more than one independent variable (a PDE)",
        )

    order = ode_order(ode, function(variable))
    validate_order(order)

    if order == 2:
        types = classify_second_order(ode, variable, function)
        if "constant_coefficients" in types and "zero_forcing_term" in types:
            return solve_constant_coeff_homogeneous(ode, variable, function)
        raise MethodNotApplicableError(
            "no solving method available for this second-order ODE",
            method="solve_ode",
            reason=f"matched types: {types}",
        )

    if order != 1:
        raise MethodNotApplicableError(
            "This project only solves first- and second-order ODEs",
            method="solve_ode",
            reason=f"detected order {order}",
        )

    f = isolate_derivative(ode, variable, function)
    types = classify(f, variable, function)

    solvers = {
        "linear": lambda: solve_linear_first_order(f, variable, function),
        "bernoulli": lambda: solve_bernoulli(f, variable, function),
        "homogeneous": lambda: solve_homogeneous(f, variable, function),
        "separable": lambda: solve_separable(f, variable, function),
    }
    for method in ("linear", "bernoulli", "homogeneous", "separable"):
        if method in types:
            return solvers[method]()

    m, n = extract_m_n(ode, variable, function)
    if is_exact(m, n, variable, function):
        return solve_exact(m, n, variable, function)

    raise MethodNotApplicableError(
        "no solving method available for this ODE",
        method="solve_ode",
        reason=f"matched types: {types}",
    )
