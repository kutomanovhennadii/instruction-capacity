"""Universal contract of a language model call."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from iec.contracts import GenParams, Prompt, RawOutput


@runtime_checkable
class ModelClient(Protocol):
    """The boundary between the bench and the adapter of a particular provider.

    An implementation converts the format of the external SDK into ``RawOutput`` right at
    this boundary. The rest of the bench sees only the opaque ``model_id`` and project
    types.
    """

    @property
    def model_id(self) -> str:
        """Return the opaque model identifier, not interpreted by the bench."""
        ...

    def generate(self, prompt: Prompt, params: GenParams) -> RawOutput:
        """Pass the ready prompt to the model and return its raw answer as a project type."""
        ...
