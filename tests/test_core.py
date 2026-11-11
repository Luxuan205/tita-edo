import pytest
from sympy import Eq, Function, symbols

from tita.core import classify_ode
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
    ode = Eq(y(x).diff(x, 2) + y(x), 0)
    with pytest.raises(MethodNotApplicableError):
        classify_ode(ode, x, y)
