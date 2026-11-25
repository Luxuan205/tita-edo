import pytest
from sympy import Eq, Function, log, sin, solve, symbols

from tita.analytic.first_order import (
    orthogonal_trajectories,
    solve_bernoulli,
    solve_by_substitution,
    solve_exact,
    solve_homogeneous,
    solve_linear_first_order,
    solve_riccati,
    solve_separable,
)
from tita.classifier import verify_solution
from tita.exceptions import MethodNotApplicableError

x = symbols("x")
y = Function("y")


def test_solve_separable_matches_known_general_solution():
    general_solution = solve_separable(x * y(x), x, y)
    assert general_solution == Eq(log(y(x)), symbols("C1") + x**2 / 2)


def test_solve_separable_solution_satisfies_the_ode():
    ode = Eq(y(x).diff(x), x * y(x))
    general_solution = solve_separable(x * y(x), x, y)
    explicit = solve(general_solution, y(x))[0]
    assert verify_solution(ode, explicit, x, y) is True


def test_solve_separable_handles_x_only_right_hand_side():
    ode = Eq(y(x).diff(x), x**2)
    general_solution = solve_separable(x**2, x, y)
    assert verify_solution(ode, general_solution.rhs, x, y) is True


def test_solve_separable_rejects_non_separable_equation():
    with pytest.raises(MethodNotApplicableError):
        solve_separable(x + y(x), x, y)


def test_solve_linear_first_order_matches_known_general_solution():
    general_solution = solve_linear_first_order(x - 2 * y(x), x, y)
    ode = Eq(y(x).diff(x), x - 2 * y(x))
    assert verify_solution(ode, general_solution.rhs, x, y) is True


def test_solve_linear_first_order_handles_zero_coefficient():
    ode = Eq(y(x).diff(x), x)
    general_solution = solve_linear_first_order(x, x, y)
    assert verify_solution(ode, general_solution.rhs, x, y) is True


def test_solve_linear_first_order_rejects_nonlinear_equation():
    with pytest.raises(MethodNotApplicableError):
        solve_linear_first_order(y(x) ** 2, x, y)


def test_solve_bernoulli_matches_known_general_solution():
    ode = Eq(y(x).diff(x), y(x) + x * y(x) ** 3)
    general_solution = solve_bernoulli(y(x) + x * y(x) ** 3, x, y)
    assert verify_solution(ode, general_solution.rhs, x, y) is True


def test_solve_bernoulli_handles_zero_linear_coefficient():
    ode = Eq(y(x).diff(x), x * y(x) ** 2)
    general_solution = solve_bernoulli(x * y(x) ** 2, x, y)
    assert verify_solution(ode, general_solution.rhs, x, y) is True


def test_solve_bernoulli_rejects_wrong_shape():
    with pytest.raises(MethodNotApplicableError):
        solve_bernoulli(y(x) ** 2 + y(x) + x, x, y)


def test_solve_bernoulli_rejects_non_polynomial_equation():
    with pytest.raises(MethodNotApplicableError):
        solve_bernoulli(sin(y(x)), x, y)


def test_solve_homogeneous_matches_known_general_solution():
    ode = Eq(y(x).diff(x), (x + y(x)) / x)
    general_solution = solve_homogeneous((x + y(x)) / x, x, y)
    explicit = solve(general_solution, y(x))[0]
    assert verify_solution(ode, explicit, x, y) is True


def test_solve_homogeneous_rejects_non_homogeneous_equation():
    with pytest.raises(MethodNotApplicableError):
        solve_homogeneous(x + y(x) ** 2, x, y)


def test_solve_exact_matches_known_general_solution():
    m, n = 2 * x * y(x) + 3, x**2 - 1
    ode = Eq(y(x).diff(x), -m / n)
    general_solution = solve_exact(m, n, x, y)
    explicit = solve(general_solution, y(x))[0]
    assert verify_solution(ode, explicit, x, y) is True


def test_solve_riccati_matches_known_general_solution():
    ode = Eq(y(x).diff(x), 1 - y(x) ** 2)
    general_solution = solve_riccati(1 - y(x) ** 2, x, y, 1)
    assert verify_solution(ode, general_solution.rhs, x, y) is True


def test_solve_riccati_rejects_cubic_equation():
    with pytest.raises(MethodNotApplicableError):
        solve_riccati(y(x) ** 3, x, y, 1)


def test_solve_by_substitution_reduces_to_known_result():
    v = Function("v")
    substitution = Eq(v(x), x + y(x))
    g = solve_by_substitution((x + y(x)) ** 2, x, y, substitution, v)
    assert g == v(x) ** 2 + 1


def test_solve_by_substitution_rejects_unsolvable_substitution():
    v = Function("v")
    substitution = Eq(v(x), y(x) ** 2 - y(x) ** 2)
    with pytest.raises(MethodNotApplicableError):
        solve_by_substitution(x, x, y, substitution, v)


def test_solve_by_substitution_rejects_substitution_that_drops_new_function():
    v = Function("v")
    substitution = Eq(v(x) - v(x), y(x) - x)
    with pytest.raises(MethodNotApplicableError):
        solve_by_substitution(x, x, y, substitution, v)


def test_orthogonal_trajectories_of_concentric_circles_are_lines():
    c = symbols("c")
    family = Eq(x**2 + y(x) ** 2, c)
    f = orthogonal_trajectories(family, c, x, y)
    assert f == y(x) / x


def test_orthogonal_trajectories_rejects_family_unsolvable_for_constant():
    c = symbols("c")
    family = Eq(x**2 + y(x) ** 2, c**2)
    with pytest.raises(MethodNotApplicableError):
        orthogonal_trajectories(family, c, x, y)


def test_orthogonal_trajectories_rejects_family_without_the_function():
    c = symbols("c")
    family = Eq(x, c)
    with pytest.raises(MethodNotApplicableError):
        orthogonal_trajectories(family, c, x, y)
