"""One exception hierarchy of the IEC measuring bench.

The exceptions of this module mean that a step could not produce a correct result by its
contract. An admissible empty result is not an exception. When an error of an external
library is wrapped, the original cause is kept through :meth:`IECError.from_cause`.
"""

from __future__ import annotations

from typing import ClassVar, Self


class IECError(Exception):
    """Root of all exceptions raised by the IEC bench itself.

    Args:
        message: Human-readable description of the error with context for finding it.
        error_code: Optional override of the default machine code of the class.
    """

    default_error_code: ClassVar[str] = "IEC_ERROR"

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code if error_code is not None else self.default_error_code

    @classmethod
    def from_cause(
        cls,
        message: str,
        cause: BaseException,
        *,
        error_code: str | None = None,
    ) -> Self:
        """Create a project exception that keeps the original cause.

        Args:
            message: Description of the error at the project level.
            cause: Original exception of the external boundary.
            error_code: Optional override of the machine code.

        Returns:
            A new instance of the calling class with the cause in ``__cause__``.

        Side effects:
            None; the calling code raises the returned exception itself.
        """
        error = cls(message, error_code=error_code)
        error.__cause__ = cause
        return error


class ConfigError(IECError):
    """The configuration is missing, incomplete or invalid before the run."""

    default_error_code: ClassVar[str] = "CONFIG_ERROR"


class VocabError(IECError):
    """Building or loading the vocabulary cannot produce a correct result."""

    default_error_code: ClassVar[str] = "VOCAB_ERROR"


class ProviderError(IECError):
    """A model or tokenizer adapter finished the operation with an error."""

    default_error_code: ClassVar[str] = "PROVIDER_ERROR"


class ValidationError(IECError):
    """The answer validator cannot produce a correct result by its contract."""

    default_error_code: ClassVar[str] = "VALIDATION_ERROR"


class ContractViolationError(IECError):
    """Data at a module boundary violate an explicit project contract."""

    default_error_code: ClassVar[str] = "CONTRACT_VIOLATION"


class PassportError(IECError):
    """The passport of a measurement cannot be computed from its published points."""

    default_error_code: ClassVar[str] = "PASSPORT_ERROR"
