import pytest

from tita.exceptions import InvalidInputError
from tita.validators import validate_initial_conditions, validate_order, validate_range


def test_validate_order_accepts_positive_integer():
    validate_order(1)


def test_validate_order_rejects_zero():
    with pytest.raises(InvalidInputError):
        validate_order(0)


def test_validate_order_rejects_non_integer():
    with pytest.raises(InvalidInputError):
        validate_order(1.5)


def test_validate_initial_conditions_accepts_matching_count():
    validate_initial_conditions([(0, 0, 1)], 1)


def test_validate_initial_conditions_rejects_mismatched_count():
    with pytest.raises(InvalidInputError):
        validate_initial_conditions([(0, 0, 1)], 2)


def test_validate_range_accepts_increasing_pair():
    validate_range((-1, 1), "x_range")


def test_validate_range_rejects_non_increasing_pair():
    with pytest.raises(InvalidInputError):
        validate_range((1, -1), "x_range")
