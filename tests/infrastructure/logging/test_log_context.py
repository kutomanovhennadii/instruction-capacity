"""Mirror tests of the IEC log context."""

from iec.infrastructure.logging import LogContext


def test_render_contains_only_present_fields() -> None:
    context = LogContext(module_name="runner", model_id="mock-1", m=8, len_tokens=4096)

    assert context.render() == "[module=runner model=mock-1 m=8 len=4096]"


def test_render_always_contains_module_name() -> None:
    assert LogContext(module_name="builder").render() == "[module=builder]"


def test_render_preserves_zero_matrix_values() -> None:
    context = LogContext(
        module_name="generator",
        seed_rules=0,
        seed_text=0,
        seed_order=0,
    )

    assert context.render() == "[module=generator seed_rules=0 seed_text=0 seed_order=0]"


def test_render_places_permutation_axis_in_key_order() -> None:
    """The prefix axes follow the order of the point identity: seed_order after seed_text."""
    context = LogContext(
        module_name="rule_generator",
        m=8,
        seed_rules=7,
        seed_text=11,
        seed_order=4,
    )

    assert context.render() == "[module=rule_generator m=8 seed_rules=7 seed_text=11 seed_order=4]"


def test_render_omits_permutation_axis_when_unknown() -> None:
    context = LogContext(module_name="vocab_builder", m=8)

    assert "seed_order=" not in context.render()


def test_as_dict_contains_unset_fields() -> None:
    context = LogContext(module_name="validator", step_name="validate")

    assert context.as_dict() == {
        "module_name": "validator",
        "step_name": "validate",
        "model_id": None,
        "m": None,
        "len_tokens": None,
        "seed_rules": None,
        "seed_text": None,
        "seed_order": None,
    }
