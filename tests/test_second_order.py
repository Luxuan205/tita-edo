import pytest
from sympy import Eq, Function, symbols

from tita.analytic.second_order import (
    characteristic_equation,
    solve_constant_coeff_homogeneous,
)
from tita.classifier import verify_solution
from tita.exceptions import MethodNotApplicableError

x = symbols("x")
y = Function("y")


def test_characteristic_equation_matches_known_polynomial():
    ode = Eq(y(x).diff(x, 2) - 3 * y(x).diff(x) + 2 * y(x), 0)
    r = symbols("r")
    assert characteristic_equation(ode, x, y) == Eq(r**2 - 3 * r + 2, 0)


def test_characteristic_equation_rejects_nonlinear_equation():
    ode = Eq(y(x).diff(x, 2) + y(x) * y(x).diff(x), 0)
    with pytest.raises(MethodNotApplicableError):
        characteristic_equation(ode, x, y)


def test_solve_constant_coeff_homogeneous_distinct_real_roots():
    ode = Eq(y(x).diff(x, 2) - 3 * y(x).diff(x) + 2 * y(x), 0)
    general_solution = solve_constant_coeff_homogeneous(ode, x, y)
    assert verify_solution(ode, general_solution.rhs, x, y) is True


def test_solve_constant_coeff_homogeneous_repeated_root():
    ode = Eq(y(x).diff(x, 2) - 4 * y(x).diff(x) + 4 * y(x), 0)
    general_solution = solve_constant_coeff_homogeneous(ode, x, y)
    assert verify_solution(ode, general_solution.rhs, x, y) is True


def test_solve_constant_coeff_homogeneous_complex_conjugate_roots():
    ode = Eq(y(x).diff(x, 2) + y(x), 0)
    general_solution = solve_constant_coeff_homogeneous(ode, x, y)
    assert verify_solution(ode, general_solution.rhs, x, y) is True
