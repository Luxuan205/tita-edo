from sympy import Expr, Function, Symbol


def substitute_function(
    f: Expr, variable: Symbol, function: Function
) -> tuple[Expr, Symbol]:
    """Replace function(variable) in f with a plain symbol.

    Parameters
    ----------
    f : sympy.Expr
        An expression containing `function(variable)` (e.g. `y(x)`).
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    tuple of (sympy.Expr, sympy.Symbol)
        The expression with `function(variable)` replaced, and the
        plain symbol used for the replacement.
    """
    dummy = Symbol(str(function))
    substituted = f.subs(function(variable), dummy)
    return substituted, dummy
