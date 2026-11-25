from sympy import Eq, Function, solve, symbols

from .core import classify_ode, solve_ode
from .exceptions import MethodNotApplicableError
from .parser import isolate_derivative
from .plotting.solution_curves import plot_direction_field_with_solutions


def main() -> None:
    """Run a short demo of tita: classify, solve, and plot a few ODEs.

    Builds a handful of example ODEs with `sympy`, classifies each one
    with `core.classify_ode`, solves it with `core.solve_ode`, and
    saves a direction field (with the solution curve overlaid, when
    possible) as a PNG file next to this script.

    Returns
    -------
    None
    """
    x = symbols("x")
    y = Function("y")

    examples = [
        ("separable + linear", Eq(y(x).diff(x), x * y(x))),
        ("exact", Eq((2 * x * y(x) + 3) + (x**2 - 1) * y(x).diff(x), 0)),
        ("second order, constant coefficients", Eq(y(x).diff(x, 2) + y(x), 0)),
    ]

    for name, ode in examples:
        print(f"\n--- {name}: {ode} ---")
        try:
            types = classify_ode(ode, x, y)
            print(f"types: {types}")
        except MethodNotApplicableError as error:
            print(f"could not classify: {error}")
            continue

        try:
            general_solution = solve_ode(ode, x, y)
            print(f"general solution: {general_solution}")
        except MethodNotApplicableError as error:
            print(f"could not solve: {error}")
            continue

        if ode.has(y(x).diff(x, 2)):
            continue

        f = isolate_derivative(ode, x, y)
        explicit_solutions = solve(general_solution, y(x))
        example_curves = [
            solution.subs({constant: 1 for constant in solution.free_symbols - {x}})
            for solution in explicit_solutions
        ]
        figure = plot_direction_field_with_solutions(
            f, x, y, (-3, 3), (-3, 3), example_curves
        )
        filename = f"direction_field_{name.replace(' ', '_').replace('+', 'and')}.png"
        figure.savefig(filename)
        print(f"saved direction field to {filename}")


if __name__ == "__main__":
    main()
