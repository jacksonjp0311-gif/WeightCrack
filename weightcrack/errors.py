"""Host errors. A file write is not a behavioral pass."""


class EditRejected(Exception):
    """The edit is invalid and no candidate bundle was published."""


class PolicyViolation(Exception):
    """The edit is well-typed and outside the host budget."""


class VerificationFailure(Exception):
    """A saved bundle, replay, or receipt failed closed."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status
