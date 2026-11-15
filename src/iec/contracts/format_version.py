"""The one rule of reading the format version of a stored contract.

Every stored data format carries ``format_version`` (specification §5): a positive integer that
the writer puts first into the object. An object without the field is of version 1, the format
written before the versions were introduced. A version newer than the one the code writes is
refused, because the fields of a newer format may mean what this code does not know.
"""

from __future__ import annotations

from collections.abc import Mapping

from iec.infrastructure.exceptions import ContractViolationError

#: Error code of a stored object of a newer format than this code reads.
FORMAT_VERSION_UNSUPPORTED = "FORMAT_VERSION_UNSUPPORTED"


def check_format_version(raw: object, current: int, *, invalid_code: str, what: str) -> None:
    """Check that this code reads the format of a stored object.

    Args:
        raw: The stored object as read from JSON.
        current: The version of the format that this code writes.
        invalid_code: The error code of a broken object of this kind.
        what: The kind of the object for the message, for example ``"a point record"``.

    Raises:
        ContractViolationError: ``invalid_code`` - the object is not a JSON object or its
            version is not a positive integer; ``FORMAT_VERSION_UNSUPPORTED`` - the version is
            newer than ``current``.
    """
    if not isinstance(raw, Mapping):
        raise ContractViolationError(f"{what} is not a JSON object", error_code=invalid_code)
    version = raw.get("format_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ContractViolationError(
            f"format_version of {what} is {version!r}, not a positive integer",
            error_code=invalid_code,
        )
    if version > current:
        raise ContractViolationError(
            f"{what} of format {version} is newer than {current}: update the package",
            error_code=FORMAT_VERSION_UNSUPPORTED,
        )
