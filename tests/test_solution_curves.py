import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sympy import Function, cos, symbols  # noqa: E402

from tita.exceptions import InvalidInputError  # noqa: E402
from tita.plotting.solution_curves import (  # noqa: E402
    plot_direction_field_with_solutions,
    plot_solution_curve,
)

x = symbols("x")
y = Function("y")


def test_plot_solution_curve_returns_a_figure():
    figure = plot_solution_curve(cos(x), x, y, (-5, 5))
    assert isinstance(figure, Figure)


def test_plot_solution_curve_handles_singularity():
    figure = plot_solution_curve(1 / (x - 1), x, y, (-2, 2), density=10)
    assert isinstance(figure, Figure)


def test_plot_solution_curve_rejects_invalid_range():
    with pytest.raises(InvalidInputError):
        plot_solution_curve(cos(x), x, y, (2, -2))


def test_plot_solution_curve_masks_values_outside_y_range():
    figure = plot_solution_curve(x**2, x, y, (-10, 10), y_range=(-3, 3), density=21)
    y_values = figure.axes[0].lines[0].get_ydata()
    assert np.isfinite(y_values).any()
    assert np.nanmax(y_values) <= 3


def test_plot_solution_curve_rejects_invalid_y_range():
    with pytest.raises(InvalidInputError):
        plot_solution_curve(cos(x), x, y, (-5, 5), y_range=(3, -3))


def test_plot_direction_field_with_solutions_returns_single_figure():
    figure = plot_direction_field_with_solutions(
        x * y(x), x, y, (-3, 3), (-3, 3), [cos(x), 1 / (1 - x)]
    )
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1


def test_plot_direction_field_with_solutions_handles_no_solutions():
    figure = plot_direction_field_with_solutions(x * y(x), x, y, (-3, 3), (-3, 3), [])
    assert isinstance(figure, Figure)
