from sympy import Eq, Function, Symbol
from sympy.solvers.deutils import ode_order

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
