#!/usr/bin/env python


__all__ = [
    "ImplementationError",
    "MaskerError",
    "MaskerResultError",
    "MaskerExecutionError"
]


class ImplementationError(Exception):
    pass


class MaskerError(Exception):
    def __init__(self, masker, string: str):
        self.masker = masker
        super().__init__(string)

    def __str__(self):
        try:
            code = self.masker.__code__
            lineno = code.co_firstlineno
            filename = code.co_filename
            loc = f'\n\tThe masker was defined here: File "{filename}", line {lineno}'
        except AttributeError:
            loc = ""

        return super().__str__() + loc


class MaskerExecutionError(MaskerError, RuntimeError):
    pass


class MaskerResultError(MaskerError, TypeError, ValueError):
    # this should catch all Exceptions that arise
    # in `BaseVariable._check_masker_result()`
    pass

