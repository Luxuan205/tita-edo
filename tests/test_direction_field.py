import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402
from sympy import Function, symbols  # noqa: E402

from tita.plotting.direction_field import plot_direction_field  # noqa: E402

x = symbols("x")
y = Function("y")


def test_plot_direction_field_returns_a_figure():
    figure = plot_direction_field(x * y(x), x, y, (-2, 2), (-2, 2))
    assert isinstance(figure, Figure)


def test_plot_direction_field_handles_constant_f():
    figure = plot_direction_field(x * 0 + 1, x, y, (-1, 1), (-1, 1))
    assert isinstance(figure, Figure)


def test_plot_direction_field_handles_singularity():
    figure = plot_direction_field(1 / x, x, y, (-1, 1), (-1, 1), density=5)
    assert isinstance(figure, Figure)
