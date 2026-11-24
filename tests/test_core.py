import pytest
from sympy import Eq, Function, symbols

from tita.classifier import verify_solution
from tita.core import classify_ode, solve_ode
from tita.exceptions import MethodNotApplicableError

x = symbols("x")
y = Function("y")


def test_classify_ode_matches_multiple_types():
    ode = Eq(y(x).diff(x), x * y(x))
    assert set(classify_ode(ode, x, y)) == {"separable", "linear"}


def test_classify_ode_resolves_unordered_equation():
    ode = Eq(y(x).diff(x) - x, y(x))
    assert "linear" in classify_ode(ode, x, y)


def test_classify_ode_returns_empty_list_when_nothing_matches():
    ode = Eq(y(x).diff(x), y(x) ** 2 + x)
    assert classify_ode(ode, x, y) == []


def test_classify_ode_detects_exact_type():
    ode = Eq((2 * x * y(x) + 3) + (x**2 - 1) * y(x).diff(x), 0)
    assert "exact" in classify_ode(ode, x, y)


def test_classify_ode_rejects_higher_order_equation():
    ode = Eq(y(x).diff(x, 3), 0)
    with pytest.raises(MethodNotApplicableError):
        classify_ode(ode, x, y)


def test_classify_ode_classifies_second_order_equation():
    ode = Eq(y(x).diff(x, 2) + y(x), 0)
    types = classify_ode(ode, x, y)
    assert set(types) == {"linear", "constant_coefficients", "zero_forcing_term"}


def test_classify_ode_rejects_partial_differential_equation():
    t = symbols("t")
    u = Function("u")
    pde = Eq(u(x, t).diff(x) + u(x, t).diff(t), 0)
    with pytest.raises(MethodNotApplicableError):
        classify_ode(pde, x, u)


def test_solve_ode_prefers_linear_when_multiple_types_match():
    ode = Eq(y(x).diff(x), x * y(x))
    general_solution = solve_ode(ode, x, y)
    assert verify_solution(ode, general_solution.rhs, x, y) is True


def test_solve_ode_solves_exact_equation():
    ode = Eq((2 * x * y(x) + 3) + (x**2 - 1) * y(x).diff(x), 0)
    general_solution = solve_ode(ode, x, y)
    explicit_ode = Eq(y(x).diff(x), -(2 * x * y(x) + 3) / (x**2 - 1))
    assert verify_solution(explicit_ode, general_solution.rhs, x, y) is True


def test_solve_ode_solves_second_order_equation():
    ode = Eq(y(x).diff(x, 2) + y(x), 0)
    general_solution = solve_ode(ode, x, y)
    assert verify_solution(ode, general_solution.rhs, x, y) is True


def test_solve_ode_rejects_second_order_without_a_known_method():
    ode = Eq(y(x).diff(x, 2) + x * y(x), 0)
    with pytest.raises(MethodNotApplicableError):
        solve_ode(ode, x, y)


def test_solve_ode_rejects_first_order_without_a_known_method():
    ode = Eq(y(x).diff(x), y(x) ** 2 + x)
    with pytest.raises(MethodNotApplicableError):
        solve_ode(ode, x, y)
