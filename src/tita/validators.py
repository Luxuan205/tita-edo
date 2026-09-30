from .exceptions import InvalidInputError


def validate_order(order):
    if not isinstance(order, int) or order <= 0:
        raise InvalidInputError(
            f"Order must be a positive integer, got {order}", param="order"
        )


def validate_initial_conditions(initial_conditions, order):
    if len(initial_conditions) != order:
        raise InvalidInputError(
            f"Expected {order} initial conditions, got {len(initial_conditions)}", param = "initial_conditions"
        )
