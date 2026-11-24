from sympy import (
    Derivative,
    Eq,
    Expr,
    Function,
    Symbol,
    checkodesol,
    separatevars,
    simplify,
    solve,
)

from .utils import linear_form, substitute_function


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


def is_exact(m: Expr, n: Expr, variable: Symbol, function: Function) -> bool:
    """Check whether M(x, y) + N(x, y) * dy/dx = 0 is exact.

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
    bool
        True if d(M)/dy equals d(N)/dx.
    """
    m_substituted, dummy = substitute_function(m, variable, function)
    n_substituted, _ = substitute_function(n, variable, function)
    difference = m_substituted.diff(dummy) - n_substituted.diff(variable)
    return simplify(difference) == 0


def is_autonomous(f: Expr, variable: Symbol, function: Function) -> bool:
    """Check whether f(x, y) does not depend explicitly on x.

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
        True if f(x, y) depends only on `function`, not on `variable`.
    """
    substituted, _ = substitute_function(f, variable, function)
    return not substituted.has(variable)


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
        "homogeneous", "bernoulli" and "autonomous". Can be more than
        one, or empty.
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
    if is_autonomous(f, variable, function):
        types.append("autonomous")
    return types


def is_ordinary(ode: Eq) -> bool:
    """Check whether ode is an ODE (a single independent variable).

    Parameters
    ----------
    ode : sympy.Eq
        The differential equation to check.

    Returns
    -------
    bool
        True if every derivative in `ode` is taken with respect to the
        same single variable (i.e. `ode` is not a PDE).
    """
    independent_variables = set()
    for derivative in ode.atoms(Derivative):
        for variable, _ in derivative.variable_count:
            independent_variables.add(variable)
    return len(independent_variables) <= 1


def is_linear_general(ode: Eq, variable: Symbol, function: Function) -> bool:
    """Check whether ode is linear in function and its derivatives.

    Unlike `is_linear` (which checks f(x, y) for a first-order ODE
    already in explicit form), this works directly on `ode` and on any
    order: it is linear if it is affine in the unknown function and
    every one of its derivatives, with no products between them and no
    nonlinear functions applied to them.

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
    bool
        True if ode is linear in `function` and its derivatives.
    """
    poly, _ = linear_form(ode, variable, function)
    return poly is not None and poly.total_degree() <= 1


def has_zero_forcing_term(ode: Eq, variable: Symbol, function: Function) -> bool:
    """Check whether ode has a zero forcing term (is homogeneous).

    This is the "homogénea/no homogénea" sense from the course's Núcleo
    branch: a linear ODE a_n(x)*y^(n) + ... + a_0(x)*y = g(x) is
    homogeneous when g(x) = 0. This is unrelated to `is_homogeneous`,
    which checks a different, first-order-specific notion (whether
    f(x, y) depends only on the ratio y/x).

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
    bool
        True if the part of ode not involving `function` or its
        derivatives is zero.
    """
    poly, dummies = linear_form(ode, variable, function)
    if poly is None:
        return False
    forcing_term = poly.nth(*([0] * len(dummies)))
    return simplify(forcing_term) == 0


def has_constant_coefficients(ode: Eq, variable: Symbol, function: Function) -> bool:
    """Check whether ode's coefficients of function and its derivatives are constant.

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
    bool
        True if none of the coefficients of `function` or its
        derivatives depend on `variable` (the forcing term, if any,
        may still depend on `variable`).
    """
    poly, _ = linear_form(ode, variable, function)
    if poly is None:
        return False

    for monomial, coefficient in poly.terms():
        if sum(monomial) == 0:
            continue
        if coefficient.has(variable):
            return False
    return True


def classify_second_order(ode: Eq, variable: Symbol, function: Function) -> list[str]:
    """Return the names of every second-order linear ODE type that ode matches.

    Parameters
    ----------
    ode : sympy.Eq
        The second-order ODE, in any form.
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    list of str
        The matching type names, among "linear", "constant_coefficients"
        and "zero_forcing_term". Can be more than one, or empty.
    """
    types = []
    if is_linear_general(ode, variable, function):
        types.append("linear")
        if has_constant_coefficients(ode, variable, function):
            types.append("constant_coefficients")
        if has_zero_forcing_term(ode, variable, function):
            types.append("zero_forcing_term")
    return types


def classify_problem_type(conditions: list[tuple[int, float, float]]) -> str:
    """Classify a set of conditions as an initial-value or boundary-value problem.

    Parameters
    ----------
    conditions : list of tuple of (int, float, float)
        Each tuple is `(order, point, value)`.

    Returns
    -------
    str
        `"ivp"` if every condition is given at the same point,
        `"bvp"` otherwise.
    """
    points = {point for _, point, _ in conditions}
    return "ivp" if len(points) <= 1 else "bvp"


def verify_solution(
    ode: Eq, candidate: Expr, variable: Symbol, function: Function
) -> bool:
    """Check whether candidate satisfies ode.

    Parameters
    ----------
    ode : sympy.Eq
        The ODE, in any form (any order).
    candidate : sympy.Expr
        A proposed expression for the unknown function (e.g. what
        `function(variable)` would equal).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    bool
        True if `candidate` satisfies `ode` identically.
    """
    is_solution, _ = checkodesol(ode, Eq(function(variable), candidate))
    return bool(is_solution)


def solution_kind(
    general_solution: Expr,
    constants: list[Symbol],
    candidate: Expr,
    variable: Symbol,
) -> str:
    """Classify a candidate solution as general, particular or singular.

    Parameters
    ----------
    general_solution : sympy.Expr
        The general solution of the ODE, containing `constants`.
    constants : list of sympy.Symbol
        The arbitrary constants appearing in `general_solution`.
    candidate : sympy.Expr
        A proposed solution to classify against `general_solution`.
    variable : sympy.Symbol
        The independent variable (e.g. `x`).

    Returns
    -------
    str
        `"general"` if `candidate` still contains any of `constants`,
        `"particular"` if it can be obtained from `general_solution` by
        choosing specific constant values, `"singular"` otherwise.
    """
    if any(candidate.has(constant) for constant in constants):
        return "general"

    solutions = solve(Eq(general_solution, candidate), constants, dict=True)
    for solution in solutions:
        fully_determined = len(solution) == len(constants)
        constant_values = not any(value.has(variable) for value in solution.values())
        if fully_determined and constant_values:
            return "particular"

    return "singular"
