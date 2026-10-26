"""Configuration of the model under measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from iec.contracts.gen_params import GenParams


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Provider-independent description of the model that answers the points.

    ``provider`` selects the adapter in the builder registry; ``model_id`` is passed to the
    provider as is and is never interpreted by the bench. ``tokenizer_name`` names the
    tokenizer that measures prompt and reference lengths (``None`` - the model id is used).
    ``context_window_tokens`` limits the prompt together with the answer reserve
    ``gen_params.max_tokens``; ``response_ceiling_tokens`` limits the answer itself;
    ``None`` means the limit is not declared and is not checked.
    """

    model_id: str
    provider: str
    gen_params: GenParams
    tokenizer_name: str | None = None
    context_window_tokens: int | None = None
    response_ceiling_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the serializable snapshot of the configuration."""
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "gen_params": self.gen_params.as_dict(),
            "tokenizer_name": self.tokenizer_name,
            "context_window_tokens": self.context_window_tokens,
            "response_ceiling_tokens": self.response_ceiling_tokens,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        """Read the snapshot written by :meth:`as_dict`; unknown keys are ignored."""
        params = raw["gen_params"]
        return cls(
            model_id=str(raw["model_id"]),
            provider=str(raw["provider"]),
            gen_params=GenParams(
                temperature=float(params["temperature"]),
                max_tokens=int(params["max_tokens"]),
                top_p=float(params["top_p"]),
                seed=None if params.get("seed") is None else int(params["seed"]),
            ),
            tokenizer_name=raw.get("tokenizer_name"),
            context_window_tokens=raw.get("context_window_tokens"),
            response_ceiling_tokens=raw.get("response_ceiling_tokens"),
        )
