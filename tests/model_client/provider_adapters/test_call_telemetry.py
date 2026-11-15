"""Mirror tests of the telemetry fields of a model call."""

from iec.contracts import RawOutput, TokenUsage
from iec.model_client.provider_adapters.call_telemetry import provider_fields


def test_counts_and_finish_reason_of_the_answer_go_to_the_event() -> None:
    usage = TokenUsage(prompt_tokens=17, response_tokens=3)

    fields = provider_fields(RawOutput(text="ok", usage=usage, finish_reason="stop"))

    assert fields.usage == {"prompt_tokens": 17, "response_tokens": 3}
    assert fields.finish_reason == "stop"


def test_a_count_the_provider_did_not_report_is_left_out() -> None:
    usage = TokenUsage(prompt_tokens=17, response_tokens=None)

    fields = provider_fields(RawOutput(text="ok", usage=usage, finish_reason="length"))

    assert fields.usage == {"prompt_tokens": 17}
    assert fields.finish_reason == "length"


def test_an_answer_without_a_counter_has_no_usage() -> None:
    fields = provider_fields(RawOutput(text=""))

    assert fields.usage is None
    assert fields.finish_reason is None
