import pytest
from sympy import Eq, Function, Matrix, sin, symbols

from tita.exceptions import MethodNotApplicableError
from tita.systems import reduce_to_system, to_matrix_form

x = symbols("x")
y = Function("y")


def test_reduce_to_system_matches_known_companion_form():
    ode = Eq(y(x).diff(x, 2) + 3 * y(x).diff(x) + 2 * y(x), sin(x))
    system, state_functions = reduce_to_system(ode, x, y)

    x1, x2 = state_functions
    assert system[0] == Eq(x1(x).diff(x), x2(x))
    assert system[1] == Eq(x2(x).diff(x), -2 * x1(x) - 3 * x2(x) + sin(x))


def test_reduce_to_system_handles_third_order_equation():
    ode = Eq(y(x).diff(x, 3) - y(x), x)
    system, state_functions = reduce_to_system(ode, x, y)
    assert len(system) == 3
    assert len(state_functions) == 3


def test_reduce_to_system_rejects_nonlinear_equation():
    ode = Eq(y(x).diff(x, 2) + y(x) * y(x).diff(x), 0)
    with pytest.raises(MethodNotApplicableError):
        reduce_to_system(ode, x, y)


def test_to_matrix_form_matches_known_matrices():
    ode = Eq(y(x).diff(x, 2) + 3 * y(x).diff(x) + 2 * y(x), sin(x))
    system, state_functions = reduce_to_system(ode, x, y)
    a_matrix, b_matrix = to_matrix_form(system, x, state_functions)

    assert a_matrix == Matrix([[0, 1], [-2, -3]])
    assert b_matrix == Matrix([[0], [sin(x)]])
