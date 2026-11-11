from sympy import Function, symbols

from tita.utils import substitute_function

x = symbols("x")
y = Function("y")


def test_substitute_function_replaces_applied_function():
    substituted, dummy = substitute_function(x * y(x), x, y)
    assert str(dummy) == "y"
    assert substituted == x * dummy
