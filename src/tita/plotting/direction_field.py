import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sympy import Expr, Function, Symbol, lambdify

from ..utils import substitute_function
from ..validators import validate_range


def plot_direction_field(
    f: Expr,
    variable: Symbol,
    function: Function,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    density: int = 20,
) -> Figure:
    """Plot the direction field of dy/dx = f(x, y).

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
    density : int
        Number of grid points along each axis.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plotted direction field.
    """
    validate_range(x_range, "x_range")
    validate_range(y_range, "y_range")

    substituted, dummy = substitute_function(f, variable, function)
    f_numeric = lambdify((variable, dummy), substituted, "numpy")

    x_values = np.linspace(*x_range, density)
    y_values = np.linspace(*y_range, density)
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = f_numeric(x_grid, y_grid)
    slopes = np.broadcast_to(slopes, x_grid.shape).astype(float)

    valid = np.isfinite(slopes)
    dx = np.where(valid, 1.0, np.nan)
    dy = np.where(valid, slopes, np.nan)
    norm = np.hypot(dx, dy)
    dx, dy = dx / norm, dy / norm

    figure, axes = plt.subplots()
    axes.quiver(x_grid, y_grid, dx, dy, angles="xy", pivot="mid")
    axes.set_xlabel(str(variable))
    axes.set_ylabel(str(function))
    return figure
