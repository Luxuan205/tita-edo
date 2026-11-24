import pytest
from sympy import Eq, Function, solve, symbols

from tita.analytic.integrating_factor import (
    find_integrating_factor_x,
    find_integrating_factor_y,
    solve_with_integrating_factor,
)
from tita.classifier import verify_solution
from tita.exceptions import MethodNotApplicableError

x = symbols("x")
y = Function("y")


def test_find_integrating_factor_x_matches_known_factor():
    m, n = y(x), x * (x - 1)
    assert find_integrating_factor_x(m, n, x, y) == x ** (-2)


def test_find_integrating_factor_x_rejects_when_no_factor_of_x_exists():
    with pytest.raises(MethodNotApplicableError):
        find_integrating_factor_x(y(x) ** 2, x, x, y)


def test_find_integrating_factor_y_matches_known_factor():
    m, n = y(x), -x
    assert find_integrating_factor_y(m, n, x, y) == y(x) ** (-2)


def test_solve_with_integrating_factor_matches_known_general_solution():
    m, n = y(x), x * (x - 1)
    ode = Eq(y(x).diff(x), -m / n)
    general_solution = solve_with_integrating_factor(m, n, x, y)
    explicit = solve(general_solution, y(x))[0]
    assert verify_solution(ode, explicit, x, y) is True


def test_solve_with_integrating_factor_rejects_when_neither_factor_exists():
    with pytest.raises(MethodNotApplicableError):
        solve_with_integrating_factor(x + y(x) ** 2, x**2 - y(x), x, y)
