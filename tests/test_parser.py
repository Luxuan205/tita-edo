import pytest
from sympy import Eq, Function, symbols

from tita.exceptions import MethodNotApplicableError
from tita.parser import extract_m_n, isolate_derivative

x = symbols("x")
y = Function("y")
derivative = y(x).diff(x)


def test_isolate_derivative_rearranges_unordered_equation():
    ode = Eq(derivative - x, y(x))
    assert isolate_derivative(ode, x, y) == x + y(x)


def test_isolate_derivative_accepts_already_explicit_equation():
    ode = Eq(derivative, y(x) - 2 * x)
    assert isolate_derivative(ode, x, y) == y(x) - 2 * x


def test_isolate_derivative_rejects_equation_without_derivative():
    ode = Eq(y(x), x)
    with pytest.raises(MethodNotApplicableError):
        isolate_derivative(ode, x, y)


def test_extract_m_n_splits_exact_equation():
    ode = Eq((2 * x * y(x) + 3) + (x**2 - 1) * derivative, 0)
    m, n = extract_m_n(ode, x, y)
    assert m == 2 * x * y(x) + 3
    assert n == x**2 - 1


def test_extract_m_n_works_on_explicit_form():
    ode = Eq(derivative, x + y(x))
    m, n = extract_m_n(ode, x, y)
    assert m == -x - y(x)
    assert n == 1


def test_extract_m_n_rejects_nonlinear_derivative():
    ode = Eq(derivative**2, x)
    with pytest.raises(MethodNotApplicableError):
        extract_m_n(ode, x, y)
