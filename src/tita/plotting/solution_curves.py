import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sympy import Expr, Function, Symbol, lambdify

from ..validators import validate_range
from .direction_field import plot_direction_field


def plot_solution_curve(
    solution: Expr,
    variable: Symbol,
    function: Function,
    x_range: tuple[float, float],
    y_range: tuple[float, float] = None,
    density: int = 200,
    ax: Axes = None,
) -> Figure:
    """Plot an explicit solution curve function(variable) = solution.

    Parameters
    ----------
    solution : sympy.Expr
        An explicit expression for `function(variable)` in terms of
        `variable` (no free arbitrary constants).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).
    x_range : tuple of (float, float)
        The (min, max) range to plot along `variable`.
    y_range : tuple of (float, float), optional
        If given, values outside this range are treated as off-screen
        (not drawn), so a vertical asymptote does not get rendered as
        a near-vertical line connecting the top and bottom of the plot.
    density : int
        Number of points to evaluate along `x_range`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure and axes are created if omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plotted solution curve.
    """
    validate_range(x_range, "x_range")
    if y_range is not None:
        validate_range(y_range, "y_range")

    solution_numeric = lambdify(variable, solution, "numpy")
    x_values = np.linspace(*x_range, density)

    with np.errstate(divide="ignore", invalid="ignore"):
        y_values = solution_numeric(x_values)
    y_values = np.broadcast_to(y_values, x_values.shape).astype(float)
    y_values = np.where(np.isfinite(y_values), y_values, np.nan)

    if y_range is not None:
        out_of_range = (y_values < y_range[0]) | (y_values > y_range[1])
        y_values = np.where(out_of_range, np.nan, y_values)

    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    ax.plot(x_values, y_values)
    ax.set_xlabel(str(variable))
    ax.set_ylabel(str(function))
    return figure


def plot_direction_field_with_solutions(
    f: Expr,
    variable: Symbol,
    function: Function,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    solutions: list[Expr],
    density: int = 20,
) -> Figure:
    """Plot the direction field of dy/dx = f(x, y) with solution curves overlaid.

    Parameters
    ----------
    f : sympy.Expr
        The right-hand side of dy/dx = f(x, y).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).
    x_range : tuple of (float, float)
        The (min, max) range to plot along `variable`.
    y_range : tuple of (float, float)
        The (min, max) range to plot along `function`.
    solutions : list of sympy.Expr
        Explicit solution curves to overlay on the direction field.
    density : int
        Number of grid points along each axis of the direction field.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the direction field and solution curves.
    """
    figure = plot_direction_field(f, variable, function, x_range, y_range, density)
    axes = figure.axes[0]

    for solution in solutions:
        plot_solution_curve(
            solution, variable, function, x_range, y_range=y_range, ax=axes
        )

    return figure
