"""Mirror tests of the contract of a language model call."""

from iec.contracts import GenParams, Prompt, RawOutput
from iec.model_client import ModelClient


class _EchoModel:
    """A model that satisfies the contract: an identifier and one call."""

    model_id = "echo"

    def generate(self, prompt: Prompt, params: GenParams) -> RawOutput:
        """Return the prompt as the answer."""
        del params
        return RawOutput(text=prompt.text)


class _NameOnly:
    """An object with the identifier of a model but without the call."""

    model_id = "silent"


def test_an_object_with_an_identifier_and_a_call_is_a_model_client() -> None:
    assert isinstance(_EchoModel(), ModelClient)


def test_an_object_without_the_call_is_not_a_model_client() -> None:
    assert not isinstance(_NameOnly(), ModelClient)
