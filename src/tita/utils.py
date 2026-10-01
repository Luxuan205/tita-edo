from sympy import Eq, Expr, Symbol, diff, solve

from .validators import validate_initial_conditions


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

    Raises
    ------
    InvalidInputError
        If the number of initial conditions does not match the number
        of constants.

    Examples
    --------
    >>> from sympy import symbols, cos, sin
    >>> x, C1, C2 = symbols("x C1 C2")
    >>> general_solution = C1 * cos(x) + C2 * sin(x)
    >>> solve_integration_constants(
    ...     general_solution, [C1, C2], x, [(0, 0, 1), (1, 0, 0)]
    ... )
    cos(x)
    """
    validate_initial_conditions(initial_conditions, len(constants))

    equations = []
    for order, point, value in initial_conditions:
        derivative = diff(general_solution, variable, order)
        value_at_point = derivative.subs(variable, point)
        equations.append(Eq(value_at_point, value))

    solved_constants = solve(equations, constants)

    return general_solution.subs(solved_constants)
