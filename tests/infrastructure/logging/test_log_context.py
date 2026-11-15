"""Mirror tests of the IEC log context."""

import dataclasses

import pytest

from iec.infrastructure.logging import LogContext


def test_render_always_contains_module_name() -> None:
    assert LogContext(module_name="builder").render() == "[module=builder]"


def test_render_adds_step_and_fields_in_their_order() -> None:
    context = LogContext(
        module_name="runner",
        step_name="run",
        fields=(("model", "mock-1"), ("m", 8), ("len", 4096)),
    )

    assert context.render() == "[module=runner step=run model=mock-1 m=8 len=4096]"


def test_render_preserves_zero_values() -> None:
    context = LogContext(module_name="generator", fields=(("seed_rules", 0), ("seed_text", 0)))

    assert context.render() == "[module=generator seed_rules=0 seed_text=0]"


def test_context_is_immutable() -> None:
    context = LogContext(module_name="runner")

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(context, "module_name", "other")
