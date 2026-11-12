"""Structured errors. Everything user-facing is a code + message, never a stack trace."""


class DataError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message

    def to_dict(self):
        return {"code": self.code, "message": self.message}


class PlanRejected(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class ExecutionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message

    def to_dict(self):
        return {"code": self.code, "message": self.message}
