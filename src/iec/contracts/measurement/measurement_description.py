"""Description of a published measurement: model, protocol, vocabulary, tariff, files.

``measurement.json`` carries ``format_version`` - the version of the format of the description
(specification §5: every data format has a version). A description without the field is of
version 1, the format of the first published measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from iec.contracts.format_version import check_format_version
from iec.contracts.language_model.model_config import ModelConfig
from iec.contracts.language_model.tariff import Tariff
from iec.infrastructure.exceptions import ContractViolationError

#: Version of the format of a measurement description that this code writes and reads.
MEASUREMENT_DESCRIPTION_FORMAT_VERSION = 1
#: Error code of a description that is not of the published format.
MEASUREMENT_DESCRIPTION_INVALID = "MEASUREMENT_DESCRIPTION_INVALID"


@dataclass(frozen=True, slots=True)
class MeasurementDescription:
    """Facts shared by all points of one published measurement.

    ``files`` maps a file name of the measurement directory to its SHA-256 hex digest.
    ``tariff`` is ``None`` when the money of the measurement is not computed.
    """

    model: ModelConfig
    protocol_version: str
    vocab_id: str
    vocab_hash: str
    tariff: Tariff | None = None
    files: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the description as one JSON object of the current format."""
        return {
            "format_version": MEASUREMENT_DESCRIPTION_FORMAT_VERSION,
            "model": self.model.as_dict(),
            "protocol_version": self.protocol_version,
            "vocab_id": self.vocab_id,
            "vocab_hash": self.vocab_hash,
            "tariff": None if self.tariff is None else self.tariff.as_dict(),
            "files": dict(sorted(self.files.items())),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MeasurementDescription:
        """Read a description of the published format; unknown keys are ignored.

        Raises:
            ContractViolationError: ``FORMAT_VERSION_UNSUPPORTED`` - the description is of a
                newer format than ``MEASUREMENT_DESCRIPTION_FORMAT_VERSION``;
                ``MEASUREMENT_DESCRIPTION_INVALID`` - the description is not a JSON object, its
                version is not a positive integer, a required key is missing or a value has a
                wrong type.
        """
        check_format_version(
            raw,
            MEASUREMENT_DESCRIPTION_FORMAT_VERSION,
            invalid_code=MEASUREMENT_DESCRIPTION_INVALID,
            what="a measurement description",
        )
        try:
            tariff = raw.get("tariff")
            return cls(
                model=ModelConfig.from_dict(raw["model"]),
                protocol_version=str(raw["protocol_version"]),
                vocab_id=str(raw["vocab_id"]),
                vocab_hash=str(raw["vocab_hash"]),
                tariff=None
                if tariff is None
                else Tariff(
                    input_usd_per_million=float(tariff["input_usd_per_million"]),
                    output_usd_per_million=float(tariff["output_usd_per_million"]),
                ),
                files={
                    str(name): str(digest) for name, digest in (raw.get("files") or {}).items()
                },
            )
        except (AttributeError, KeyError, ValueError, TypeError) as cause:
            raise ContractViolationError.from_cause(
                f"not a measurement description of the published format: {cause!r}",
                cause,
                error_code=MEASUREMENT_DESCRIPTION_INVALID,
            ) from cause
