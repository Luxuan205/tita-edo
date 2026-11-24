from sympy import Eq, Function, Integer, cos, sin, symbols

from tita.classifier import (
    classify,
    classify_problem_type,
    classify_second_order,
    has_constant_coefficients,
    has_zero_forcing_term,
    is_autonomous,
    is_bernoulli,
    is_exact,
    is_homogeneous,
    is_linear,
    is_linear_general,
    is_ordinary,
    is_separable,
    solution_kind,
    verify_solution,
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


def test_is_autonomous_true_without_explicit_x():
    assert is_autonomous(y(x) ** 2, x, y) is True


def test_is_autonomous_false_with_explicit_x():
    assert is_autonomous(x + y(x), x, y) is False


def test_is_ordinary_true_for_single_variable_equation():
    ode = Eq(y(x).diff(x), x + y(x))
    assert is_ordinary(ode) is True


def test_is_ordinary_false_for_partial_differential_equation():
    t = symbols("t")
    u = Function("u")
    pde = Eq(u(x, t).diff(x) + u(x, t).diff(t), 0)
    assert is_ordinary(pde) is False


def test_is_linear_general_true_for_constant_coefficient_equation():
    ode = Eq(y(x).diff(x, 2) + y(x), 0)
    assert is_linear_general(ode, x, y) is True


def test_is_linear_general_false_for_nonlinear_equation():
    ode = Eq(y(x).diff(x, 2) + y(x) * y(x).diff(x), 0)
    assert is_linear_general(ode, x, y) is False


def test_has_zero_forcing_term_true_for_homogeneous_equation():
    ode = Eq(y(x).diff(x, 2) + y(x), 0)
    assert has_zero_forcing_term(ode, x, y) is True


def test_has_zero_forcing_term_false_for_nonzero_forcing():
    ode = Eq(y(x).diff(x, 2) + y(x), sin(x))
    assert has_zero_forcing_term(ode, x, y) is False


def test_has_constant_coefficients_true():
    ode = Eq(y(x).diff(x, 2) + y(x), 0)
    assert has_constant_coefficients(ode, x, y) is True


def test_has_constant_coefficients_false_for_variable_coefficient():
    ode = Eq(y(x).diff(x, 2) + x * y(x).diff(x) + y(x), 0)
    assert has_constant_coefficients(ode, x, y) is False


def test_classify_second_order_matches_multiple_types():
    ode = Eq(y(x).diff(x, 2) + y(x), 0)
    types = classify_second_order(ode, x, y)
    assert set(types) == {"linear", "constant_coefficients", "zero_forcing_term"}


def test_classify_second_order_empty_for_nonlinear_equation():
    ode = Eq(y(x).diff(x, 2) + y(x) * y(x).diff(x), 0)
    assert classify_second_order(ode, x, y) == []


def test_classify_problem_type_ivp_for_same_point():
    assert classify_problem_type([(0, 0, 1), (1, 0, 0)]) == "ivp"


def test_classify_problem_type_bvp_for_different_points():
    assert classify_problem_type([(0, 0, 1), (0, 1, 2)]) == "bvp"


def test_verify_solution_true_for_correct_candidate():
    ode = Eq(y(x).diff(x, 2) + y(x), 0)
    assert verify_solution(ode, cos(x), x, y) is True


def test_verify_solution_false_for_incorrect_candidate():
    ode = Eq(y(x).diff(x, 2) + y(x), 0)
    assert verify_solution(ode, x, x, y) is False


def test_solution_kind_general_for_expression_with_constants():
    c1, c2 = symbols("C1 C2")
    general_solution = c1 * cos(x) + c2 * sin(x)
    assert solution_kind(general_solution, [c1, c2], general_solution, x) == "general"


def test_solution_kind_particular_for_fixed_constants():
    c1, c2 = symbols("C1 C2")
    general_solution = c1 * cos(x) + c2 * sin(x)
    assert solution_kind(general_solution, [c1, c2], cos(x), x) == "particular"


def test_solution_kind_singular_when_unreachable_from_general_solution():
    c1, c2 = symbols("C1 C2")
    general_solution = c1 * cos(x) + c2 * sin(x)
    assert solution_kind(general_solution, [c1, c2], x, x) == "singular"
