import pytest

from tita.exceptions import (
    InvalidInputError,
    MethodNotApplicableError,
    ODEError,
    SingularityError,
)


def test_method_not_applicable_error_stores_attributes():
    error = MethodNotApplicableError("message", method="solve_x", reason="not linear")
    assert isinstance(error, ODEError)
    assert error.method == "solve_x"
    assert error.reason == "not linear"


def test_singularity_error_stores_point():
    error = SingularityError("message", point=0)
    assert isinstance(error, ODEError)
    assert error.point == 0


def test_invalid_input_error_stores_param():
    error = InvalidInputError("message", param="order")
    assert isinstance(error, ODEError)
    assert error.param == "order"


def test_invalid_input_error_defaults_param_to_none():
    error = InvalidInputError("message")
    assert error.param is None


def test_ode_error_is_raisable_as_an_exception():
    with pytest.raises(ODEError):
        raise ODEError("message")
