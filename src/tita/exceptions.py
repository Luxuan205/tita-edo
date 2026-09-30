class ODEError(Exception):
    """Base exception class in the library"""


class MethodNotApplicableError(ODEError):
    """Raised when an solving method is wrongly applied"""

    def __init__(self, message, method, reason):
        super().__init__(message)
        self.method = method
        self.reason = reason


class ConvergenceError(ODEError):
    """Raised when a numeric method fails to converge."""

    def __init__(self, message, iterations=None, tolerance=None):
        super().__init__(message)
        self.iterations = iterations
        self.tolerance = tolerance


class SingularityError(ODEError):
    """Raised when a singularity is found"""

    def __init__(self, message, point):
        super().__init__(message)
        self.point = point
