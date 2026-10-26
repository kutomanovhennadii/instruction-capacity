"""Description of a published measurement: model, protocol, vocabulary, tariff, files."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from iec.contracts.model_config import ModelConfig
from iec.contracts.tariff import Tariff


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
        """Return the description as one JSON object."""
        return {
            "model": self.model.as_dict(),
            "protocol_version": self.protocol_version,
            "vocab_id": self.vocab_id,
            "vocab_hash": self.vocab_hash,
            "tariff": None if self.tariff is None else self.tariff.as_dict(),
            "files": dict(sorted(self.files.items())),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MeasurementDescription:
        """Read the object written by :meth:`as_dict`; unknown keys are ignored."""
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
            files={str(name): str(digest) for name, digest in (raw.get("files") or {}).items()},
        )
