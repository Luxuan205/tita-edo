from sympy import (
    Eq,
    Expr,
    Function,
    Rational,
    Symbol,
    exp,
    integrate,
    separatevars,
    simplify,
    solve,
    symbols,
)

from ..exceptions import MethodNotApplicableError
from ..utils import substitute_function


def solve_separable(f: Expr, variable: Symbol, function: Function) -> Eq:
    """Solve a separable first-order ODE dy/dx = g(x) * h(y).

    Parameters
    ----------
    f : sympy.Expr
        The right-hand side of dy/dx = f(x, y), assumed separable.
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    sympy.Eq
        The general solution, as an implicit equation relating
        `function(variable)` to `variable` and one arbitrary
        constant `C1`.
    """
    substituted, dummy = substitute_function(f, variable, function)
    result = separatevars(substituted, symbols=[variable, dummy], dict=True)

    if result is None:
        raise MethodNotApplicableError(
            "f(x, y) does not separate into g(x) * h(y)",
            method="solve_separable",
            reason="separatevars could not split the expression",
        )

    g = result[variable] * result["coeff"]
    h = result[dummy]

    constant = Symbol("C1")
    lhs = integrate(1 / h, dummy)
    rhs = integrate(g, variable)

    general_solution = Eq(lhs, rhs + constant)
    return general_solution.subs(dummy, function(variable))


def solve_linear_first_order(f: Expr, variable: Symbol, function: Function) -> Eq:
    """Solve a linear first-order ODE dy/dx + p(x) * y = q(x).

    Parameters
    ----------
    f : sympy.Expr
        The right-hand side of dy/dx = f(x, y), assumed linear in
        `function` (i.e. f(x, y) = -p(x) * y + q(x)).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    sympy.Eq
        The general solution `function(variable) = ...`, found via
        the integrating factor exp(integral of p(x)).
    """
    substituted, dummy = substitute_function(f, variable, function)
    poly = substituted.as_poly(dummy)

    if poly is None or poly.degree() > 1:
        raise MethodNotApplicableError(
            "f(x, y) is not linear in y",
            method="solve_linear_first_order",
            reason="f(x, y) is not a degree-1 (or lower) polynomial in y",
        )

    p = -poly.coeff_monomial(dummy)
    q = poly.coeff_monomial(1)

    integrating_factor = exp(integrate(p, variable))
    constant = Symbol("C1")
    rhs_integral = integrate(integrating_factor * q, variable)

    y_solution = (rhs_integral + constant) / integrating_factor
    return Eq(function(variable), y_solution)


def solve_bernoulli(f: Expr, variable: Symbol, function: Function) -> Eq:
    """Solve a Bernoulli first-order ODE dy/dx = -p(x) * y + q(x) * y**n.

    Uses the substitution v = y**(1 - n), which turns the equation
    into a linear ODE in v, delegating to `solve_linear_first_order`.

    Parameters
    ----------
    f : sympy.Expr
        The right-hand side of dy/dx = f(x, y), assumed to have the
        Bernoulli form.
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    sympy.Eq
        The general solution `function(variable) = ...`.
    """
    substituted, dummy = substitute_function(f, variable, function)
    poly = substituted.as_poly(dummy)

    if poly is None:
        raise MethodNotApplicableError(
            "f(x, y) is not a polynomial in y",
            method="solve_bernoulli",
            reason="f(x, y) could not be written as a polynomial in y",
        )

    exponents = {exponent for (exponent,), _ in poly.terms()}
    non_linear_exponents = exponents - {0, 1}
    if len(non_linear_exponents) != 1 or not exponents <= {1} | non_linear_exponents:
        raise MethodNotApplicableError(
            "f(x, y) does not have the Bernoulli form -p(x)*y + q(x)*y**n",
            method="solve_bernoulli",
            reason=f"unexpected exponents {exponents} for y",
        )

    n = non_linear_exponents.pop()
    p = -poly.coeff_monomial(dummy)
    q = poly.coeff_monomial(dummy**n)

    v = Function(str(function) + "_v")
    one_minus_n = 1 - n
    f_v = -one_minus_n * p * v(variable) + one_minus_n * q

    v_solution = solve_linear_first_order(f_v, variable, v)
    y_solution = v_solution.rhs ** Rational(1, one_minus_n)
    return Eq(function(variable), y_solution)


def solve_homogeneous(f: Expr, variable: Symbol, function: Function) -> Eq:
    """Solve a homogeneous first-order ODE dy/dx = f(x, y), f(tx, ty) = f(x, y).

    Uses the substitution v = y / x, which turns the equation into a
    separable ODE in v, delegating to `solve_separable`.

    Parameters
    ----------
    f : sympy.Expr
        The right-hand side of dy/dx = f(x, y), assumed homogeneous of
        degree 0.
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    sympy.Eq
        The general solution, as an implicit equation relating
        `function(variable)` to `variable` and one arbitrary
        constant `C1`.
    """
    substituted, dummy = substitute_function(f, variable, function)
    v_symbol = symbols("v")
    f_of_v = simplify(substituted.subs(dummy, v_symbol * variable))

    if f_of_v.has(variable):
        raise MethodNotApplicableError(
            "f(x, y) is not homogeneous of degree 0",
            method="solve_homogeneous",
            reason="f(x, v*x) still depends on x after substitution",
        )

    v = Function(str(function) + "_v")
    f_v = (f_of_v - v_symbol) / variable
    f_v_applied = f_v.subs(v_symbol, v(variable))

    v_solution = solve_separable(f_v_applied, variable, v)
    return v_solution.subs(v(variable), function(variable) / variable)


def solve_exact(m: Expr, n: Expr, variable: Symbol, function: Function) -> Eq:
    """Solve an exact first-order ODE M(x, y) + N(x, y) * dy/dx = 0.

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
        The general solution, as an implicit equation
        `potential(x, y) = C1`, where potential is the function whose
        partial derivatives are M and N.
    """
    m_substituted, dummy = substitute_function(m, variable, function)
    n_substituted, _ = substitute_function(n, variable, function)

    potential = integrate(m_substituted, variable)
    remaining = simplify(n_substituted - potential.diff(dummy))
    potential += integrate(remaining, dummy)

    constant = Symbol("C1")
    general_solution = Eq(potential, constant)
    return general_solution.subs(dummy, function(variable))


def solve_riccati(
    f: Expr, variable: Symbol, function: Function, particular_solution: Expr
) -> Eq:
    """Solve a Riccati ODE dy/dx = q0(x) + q1(x)*y + q2(x)*y**2.

    Requires a known particular solution. Uses the substitution
    y = particular_solution + 1 / v, which turns the equation into a
    linear ODE in v, delegating to `solve_linear_first_order`.

    Parameters
    ----------
    f : sympy.Expr
        The right-hand side of dy/dx = f(x, y), assumed to have the
        Riccati form.
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).
    particular_solution : sympy.Expr
        A known solution of the ODE (in terms of `variable`).

    Returns
    -------
    sympy.Eq
        The general solution `function(variable) = ...`.
    """
    substituted, dummy = substitute_function(f, variable, function)
    poly = substituted.as_poly(dummy)

    if poly is None or poly.degree() > 2:
        raise MethodNotApplicableError(
            "f(x, y) is not a degree-2 (or lower) polynomial in y",
            method="solve_riccati",
            reason="f(x, y) does not have the Riccati form",
        )

    q1 = poly.coeff_monomial(dummy)
    q2 = poly.coeff_monomial(dummy**2)

    v = Function(str(function) + "_v")
    coefficient = q1 + 2 * q2 * particular_solution
    f_v = -coefficient * v(variable) - q2

    v_solution = solve_linear_first_order(f_v, variable, v)
    y_solution = particular_solution + 1 / v_solution.rhs
    return Eq(function(variable), y_solution)


def solve_by_substitution(
    f: Expr,
    variable: Symbol,
    function: Function,
    substitution: Eq,
    new_function: Function,
) -> Expr:
    """Rewrite dy/dx = f(x, y) as dv/dx = g(x, v) using a given substitution.

    The result is a new right-hand side, ready to be classified and
    solved with another technique (e.g. `solve_separable`,
    `solve_linear_first_order`).

    Parameters
    ----------
    f : sympy.Expr
        The right-hand side of dy/dx = f(x, y).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).
    substitution : sympy.Eq
        An equation relating `new_function(variable)` to `variable`
        and `function(variable)` (e.g. `Eq(v(x), x + y(x))`).
    new_function : sympy.Function
        The new unknown function introduced by `substitution`.

    Returns
    -------
    sympy.Expr
        `g(x, v)`, the right-hand side of the transformed ODE
        dv/dx = g(x, v).
    """
    y_solutions = solve(substitution, function(variable))
    if len(y_solutions) != 1:
        raise MethodNotApplicableError(
            "substitution cannot be solved uniquely for function(variable)",
            method="solve_by_substitution",
            reason=f"solve() returned {len(y_solutions)} solutions",
        )

    y_expr = y_solutions[0]
    dy_dx = y_expr.diff(variable)
    f_substituted = f.subs(function(variable), y_expr)

    new_derivative = new_function(variable).diff(variable)
    g_solutions = solve(Eq(dy_dx, f_substituted), new_derivative)
    if len(g_solutions) != 1:
        raise MethodNotApplicableError(
            "could not isolate the new derivative after substitution",
            method="solve_by_substitution",
            reason=f"solve() returned {len(g_solutions)} solutions",
        )

    return g_solutions[0]


def orthogonal_trajectories(
    family: Eq, constant: Symbol, variable: Symbol, function: Function
) -> Expr:
    """Return f(x, y) for the ODE of the family orthogonal to the given one.

    Parameters
    ----------
    family : sympy.Eq
        The family of curves, as `family(x, y, constant) = 0`.
    constant : sympy.Symbol
        The parameter distinguishing curves within the family.
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    sympy.Expr
        `f(x, y)`, the right-hand side of dy/dx = f(x, y) satisfied by
        the orthogonal family.
    """
    derivative = function(variable).diff(variable)
    implicit = family.lhs - family.rhs
    differentiated = implicit.diff(variable)

    constant_solutions = solve(implicit, constant)
    if len(constant_solutions) != 1:
        raise MethodNotApplicableError(
            "family could not be solved uniquely for constant",
            method="orthogonal_trajectories",
            reason=f"solve() returned {len(constant_solutions)} solutions",
        )

    differentiated = differentiated.subs(constant, constant_solutions[0])
    slope_solutions = solve(Eq(differentiated, 0), derivative)
    if len(slope_solutions) != 1:
        raise MethodNotApplicableError(
            "could not isolate dy/dx from the differentiated family",
            method="orthogonal_trajectories",
            reason=f"solve() returned {len(slope_solutions)} solutions",
        )

    return -1 / slope_solutions[0]
