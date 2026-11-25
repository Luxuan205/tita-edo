from .exceptions import InvalidInputError


def validate_order(order):
    if not isinstance(order, int) or order <= 0:
        raise InvalidInputError(
            f"Order must be a positive integer, got {order}", param="order"
        )


def validate_initial_conditions(initial_conditions, order):
    if len(initial_conditions) != order:
        raise InvalidInputError(
            f"Expected {order} initial conditions, got {len(initial_conditions)}",
            param="initial_conditions",
        )


def validate_range(value_range: tuple[float, float], param: str) -> None:
    """Validate that a (min, max) range is strictly increasing.

    Parameters
    ----------
    value_range : tuple of (float, float)
        The range to check.
    param : str
        Name of the parameter being validated, used in error messages.
    """
    low, high = value_range
    if low >= high:
        raise InvalidInputError(
            f"{param} must be an increasing (min, max) pair, got {value_range}",
            param=param,
        )
