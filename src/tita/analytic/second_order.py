from sympy import Eq, Function, Symbol, cos, exp, sin
from sympy import roots as sympy_roots

from ..exceptions import MethodNotApplicableError
from ..utils import linear_form


def characteristic_equation(ode: Eq, variable: Symbol, function: Function) -> Eq:
    """Build the characteristic equation of a constant-coefficient linear ODE.

    Parameters
    ----------
    ode : sympy.Eq
        The ODE, in any form (any order), assumed linear with constant
        coefficients.
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    sympy.Eq
        The characteristic equation `a_n*r**n + ... + a_0 = 0`, where
        `r` stands in for each derivative's exponential rate.
    """
    poly, dummies = linear_form(ode, variable, function)
    if poly is None or poly.total_degree() > 1:
        raise MethodNotApplicableError(
            "ode is not linear in function and its derivatives",
            method="characteristic_equation",
            reason="ode is not affine in function and its derivatives",
        )

    r = Symbol("r")
    characteristic = sum(
        poly.coeff_monomial(dummy) * r**power for power, dummy in enumerate(dummies)
    )
    return Eq(characteristic, 0)


def solve_constant_coeff_homogeneous(
    ode: Eq, variable: Symbol, function: Function
) -> Eq:
    """Solve a_n*y^(n) + ... + a_0*y = 0 with constant coefficients.

    Builds the characteristic equation and combines its roots into the
    general solution: `C * exp(r*x)` per real root (times `x**k` for
    repeated roots), and `exp(a*x) * (C1*cos(b*x) + C2*sin(b*x))` per
    complex conjugate pair `a +- b*i`.

    Parameters
    ----------
    ode : sympy.Eq
        The ODE, in any form, assumed linear, homogeneous (zero
        forcing term) and with constant coefficients.
    variable : sympy.Symbol
        The independent variable (e.g. `x`).
    function : sympy.Function
        The unknown function (e.g. `Function("y")`, not yet applied
        to `variable`).

    Returns
    -------
    sympy.Eq
        The general solution `function(variable) = ...`.
    """
    char_eq = characteristic_equation(ode, variable, function)
    r = Symbol("r")
    root_multiplicities = sympy_roots(char_eq.lhs, r)

    terms = []
    constant_index = 1
    processed_conjugates = set()

    for root, multiplicity in root_multiplicities.items():
        if root in processed_conjugates:
            continue

        if root.is_real:
            for power in range(multiplicity):
                constant = Symbol(f"C{constant_index}")
                constant_index += 1
                terms.append(constant * variable**power * exp(root * variable))
        else:
            processed_conjugates.add(root.conjugate())
            alpha, beta = root.as_real_imag()
            for power in range(multiplicity):
                cosine_constant = Symbol(f"C{constant_index}")
                constant_index += 1
                sine_constant = Symbol(f"C{constant_index}")
                constant_index += 1
                terms.append(
                    variable**power
                    * exp(alpha * variable)
                    * (
                        cosine_constant * cos(beta * variable)
                        + sine_constant * sin(beta * variable)
                    )
                )

    return Eq(function(variable), sum(terms))
