"""Telemetry of a model call: the technical fields of the provider taken from the raw answer.

The event of a model call carries the token counts and the finish reason of the provider
(specification §8). Both model adapters take them from ``RawOutput`` by this one function, so the
rule of the fields lives in one place.
"""

from __future__ import annotations

from iec.contracts import RawOutput
from iec.infrastructure.telemetry import ProviderFields


def provider_fields(output: RawOutput) -> ProviderFields:
    """Return the token counts and the finish reason of a model answer for its telemetry event.

    Args:
        output: The raw answer of the model in project fields.

    Returns:
        The provider fields; a count the provider did not report is left out of ``usage``, and an
        answer without a counter has no ``usage`` at all.
    """
    usage: dict[str, int] | None = None
    if output.usage is not None:
        counts = output.usage.as_dict()
        usage = {name: count for name, count in counts.items() if count is not None}
    return ProviderFields(usage=usage, finish_reason=output.finish_reason)
