from sympy import Expr, Function, Symbol, separatevars, simplify

from .utils import substitute_function


def is_separable(f: Expr, variable: Symbol, function: Function) -> bool:
    """Check whether f(x, y) can be written as g(x) * h(y).

    Parameters
    ----------
    f : sympy.Expr
        The right-hand side of dy/dx = f(x, y).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    bool
        True if f(x, y) separates into a product of a function of
        `variable` alone and a function of `function` alone.
    """
    substituted, dummy = substitute_function(f, variable, function)
    result = separatevars(substituted, symbols=[variable, dummy], dict=True)
    return result is not None


def is_linear(f: Expr, variable: Symbol, function: Function) -> bool:
    """Check whether f(x, y) has the form -p(x) * y + q(x).

    Parameters
    ----------
    f : sympy.Expr
        The right-hand side of dy/dx = f(x, y).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    bool
        True if f(x, y) is a degree-1 (or lower) polynomial in
        `function`, with coefficients that may depend on `variable`.
    """
    substituted, dummy = substitute_function(f, variable, function)
    poly = substituted.as_poly(dummy)
    return poly is not None and poly.degree() <= 1


def is_homogeneous(f: Expr, variable: Symbol, function: Function) -> bool:
    """Check whether f(x, y) is homogeneous of degree 0.

    Parameters
    ----------
    f : sympy.Expr
        The right-hand side of dy/dx = f(x, y).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    bool
        True if f(t*x, t*y) == f(x, y) for a scaling factor `t`,
        i.e. f(x, y) depends only on the ratio y/x.
    """
    substituted, dummy = substitute_function(f, variable, function)
    t = Symbol("t", positive=True)
    scaled = substituted.subs(
        {variable: t * variable, dummy: t * dummy}, simultaneous=True
    )
    return simplify(scaled - substituted) == 0


def is_bernoulli(f: Expr, variable: Symbol, function: Function) -> bool:
    """Check whether f(x, y) has the form p(x) * y + q(x) * y**n.

    Parameters
    ----------
    f : sympy.Expr
        The right-hand side of dy/dx = f(x, y).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    bool
        True if f(x, y), as a polynomial in `function`, has at most
        two nonzero terms: one of exponent 1 and one of some other
        exponent n (n != 0, n != 1).
    """
    substituted, dummy = substitute_function(f, variable, function)
    poly = substituted.as_poly(dummy)
    if poly is None:
        return False

    exponents = {exponent for (exponent,), _ in poly.terms()}
    non_linear_exponents = exponents - {0, 1}
    if len(non_linear_exponents) != 1:
        return False

    return exponents <= {1} | non_linear_exponents


def classify(f: Expr, variable: Symbol, function: Function) -> list[str]:
    """Return the names of every first-order ODE type that f(x, y) matches.

    Parameters
    ----------
    f : sympy.Expr
        The right-hand side of dy/dx = f(x, y).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    list of str
        The matching type names, among "separable", "linear",
        "homogeneous" and "bernoulli". Can be more than one, or empty.
    """
    types = []
    if is_separable(f, variable, function):
        types.append("separable")
    if is_linear(f, variable, function):
        types.append("linear")
    if is_homogeneous(f, variable, function):
        types.append("homogeneous")
    if is_bernoulli(f, variable, function):
        types.append("bernoulli")
    return types
