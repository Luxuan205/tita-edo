from sympy import Eq, Function, Symbol
from sympy.solvers.deutils import ode_order

from .classifier import classify, is_exact
from .exceptions import MethodNotApplicableError
from .parser import extract_m_n, isolate_derivative
from .validators import validate_order


def classify_ode(ode: Eq, variable: Symbol, function: Function) -> list[str]:
    """Classify a first-order ODE by its type(s).

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
    list of str
        The matching ODE type names (see `classifier.classify` and
        `classifier.is_exact`).
    """
    order = ode_order(ode, function(variable))
    validate_order(order)
    if order != 1:
        raise MethodNotApplicableError(
            "This project only classifies first-order ODEs",
            method="classify_ode",
            reason=f"detected order {order}",
        )

    f = isolate_derivative(ode, variable, function)
    types = classify(f, variable, function)

    m, n = extract_m_n(ode, variable, function)
    if is_exact(m, n, variable, function):
        types.append("exact")

    return types
