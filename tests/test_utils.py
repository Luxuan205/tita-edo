import pytest
from sympy import Eq, Function, cos, sin, symbols

from tita.exceptions import InvalidInputError
from tita.utils import linear_form, solve_integration_constants, substitute_function

x = symbols("x")
y = Function("y")


def test_substitute_function_replaces_applied_function():
    substituted, dummy = substitute_function(x * y(x), x, y)
    assert str(dummy) == "y"
    assert substituted == x * dummy


def test_linear_form_returns_polynomial_for_linear_equation():
    ode = Eq(y(x).diff(x, 2) + y(x), 0)
    poly, dummies = linear_form(ode, x, y)
    assert poly is not None
    assert poly.total_degree() == 1
    assert len(dummies) == 3


def test_linear_form_returns_higher_degree_for_nonlinear_product():
    ode = Eq(y(x).diff(x, 2) + y(x) * y(x).diff(x), 0)
    poly, _ = linear_form(ode, x, y)
    assert poly.total_degree() == 2


def test_linear_form_returns_none_for_non_polynomial_equation():
    ode = Eq(y(x).diff(x, 2) + sin(y(x)), 0)
    poly, _ = linear_form(ode, x, y)
    assert poly is None


def test_solve_integration_constants_finds_particular_solution():
    c1, c2 = symbols("C1 C2")
    general_solution = c1 * cos(x) + c2 * sin(x)
    particular = solve_integration_constants(
        general_solution, [c1, c2], x, [(0, 0, 1), (1, 0, 0)]
    )
    assert particular == cos(x)


def test_solve_integration_constants_rejects_mismatched_conditions():
    c1, c2 = symbols("C1 C2")
    general_solution = c1 * cos(x) + c2 * sin(x)
    with pytest.raises(InvalidInputError):
        solve_integration_constants(general_solution, [c1, c2], x, [(0, 0, 1)])
