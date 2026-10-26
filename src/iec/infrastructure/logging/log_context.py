"""Typed context of IEC log records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LogContext:
    """Identifiers of one logical operation or measurement point.

    ``module_name`` is always present. The other fields are optional and appear in the
    record prefix only when the calling code knows them.
    """

    module_name: str
    step_name: str | None = None
    model_id: str | None = None
    m: int | None = None
    len_tokens: int | None = None
    seed_rules: int | None = None
    seed_text: int | None = None
    seed_order: int | None = None

    def render(self) -> str:
        """Return the set fields as a stable ``[key=value ...]`` prefix."""
        parts = [f"module={self.module_name}"]
        if self.step_name:
            parts.append(f"step={self.step_name}")
        if self.model_id:
            parts.append(f"model={self.model_id}")
        if self.m is not None:
            parts.append(f"m={self.m}")
        if self.len_tokens is not None:
            parts.append(f"len={self.len_tokens}")
        if self.seed_rules is not None:
            parts.append(f"seed_rules={self.seed_rules}")
        if self.seed_text is not None:
            parts.append(f"seed_text={self.seed_text}")
        if self.seed_order is not None:
            parts.append(f"seed_order={self.seed_order}")
        return "[" + " ".join(parts) + "]"

    def as_dict(self) -> dict[str, str | int | None]:
        """Return all context fields, unset values included as ``None``."""
        return {
            "module_name": self.module_name,
            "step_name": self.step_name,
            "model_id": self.model_id,
            "m": self.m,
            "len_tokens": self.len_tokens,
            "seed_rules": self.seed_rules,
            "seed_text": self.seed_text,
            "seed_order": self.seed_order,
        }
