from sympy import Function, Integer, symbols

from tita.classifier import (
    classify,
    is_bernoulli,
    is_exact,
    is_homogeneous,
    is_linear,
    is_separable,
)

x = symbols("x")
y = Function("y")


def test_is_separable_true_for_product_form():
    assert is_separable(x * (1 + y(x)), x, y) is True


def test_is_separable_false_for_sum_form():
    assert is_separable(x + y(x), x, y) is False


def test_is_linear_true_for_affine_form():
    assert is_linear(-3 * y(x) + x**2, x, y) is True


def test_is_linear_false_for_quadratic_term():
    assert is_linear(y(x) ** 2, x, y) is False


def test_is_homogeneous_true_for_ratio_form():
    assert is_homogeneous(y(x) / x, x, y) is True


def test_is_homogeneous_false_for_non_ratio_form():
    assert is_homogeneous(y(x) - 2 * x, x, y) is False


def test_is_bernoulli_true_for_power_term():
    assert is_bernoulli(y(x) + x * y(x) ** 3, x, y) is True


def test_is_bernoulli_false_with_extra_constant_term():
    assert is_bernoulli(y(x) ** 2 + x, x, y) is False


def test_is_exact_true_for_matching_partials():
    m = 2 * x * y(x) + 3
    n = x**2 - 1
    assert is_exact(m, n, x, y) is True


def test_is_exact_false_for_mismatched_partials():
    m = -x - y(x)
    n = Integer(1)
    assert is_exact(m, n, x, y) is False


def test_classify_returns_multiple_matching_types():
    types = classify(x * y(x), x, y)
    assert set(types) == {"separable", "linear"}


def test_classify_returns_empty_list_when_nothing_matches():
    assert classify(y(x) ** 2 + x, x, y) == []
