"""The assignment of the forms: the shipped file, its schema and the errors of a block."""

from __future__ import annotations

from pathlib import Path

import pytest

from iec.infrastructure.exceptions import ConfigError, ContractViolationError
from iec.passport.laws.forms.form_registry import LAW_FORM_UNKNOWN
from iec.passport.laws.forms.hyperbolic_forms import (
    AsymptoticHyperbolicForm,
    SlantedHyperbolicForm,
    TwoSlopeHyperbolicForm,
)
from iec.passport.laws.forms.law_assignment import (
    LAW_ASSIGNMENT_CONFIG_INVALID,
    LAW_BLOCK_UNKNOWN,
    build_law_assignment,
    load_law_assignment_config,
)
from iec.passport.laws.forms.linear_forms import LinearForm, LogarithmicForm, LogLinearLoadForm

SHIPPED = Path(__file__).resolve().parents[4] / "config" / "passport_laws.yaml"
MINIMAL = """\
format_version: 2
quantities:
  H:
    blocks:
      table_size: {forms: [logarithmic]}
    spread: {V: logarithmic}
"""
#: The forms of a block with the slanted hyperbola established: then its alternatives.
HYPERBOLA_FIRST = ["slanted_hyperbolic", "linear", "hyperbolic", "binary_exponential"]
#: The forms of the density of H: the two-slope hyperbola established, the slanted one among the alternatives.
TWO_SLOPE_FIRST = ["two_slope_hyperbolic", *HYPERBOLA_FIRST]


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "passport_laws.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_shipped_assignment_gives_the_forms_of_the_passport() -> None:
    assignment = build_law_assignment(load_law_assignment_config(SHIPPED))
    assert [form.name for form in assignment.forms_of("P_sync", "density")] == HYPERBOLA_FIRST
    assert assignment.form_of("P_sync", "density") == SlantedHyperbolicForm()
    assert assignment.form_of("P_sync", "grid_slices") == AsymptoticHyperbolicForm()
    assert assignment.form_of("H", "grid_slices").name == "binary_exponential"
    assert [form.name for form in assignment.forms_of("H", "density")] == TWO_SLOPE_FIRST
    assert assignment.form_of("H", "density") == TwoSlopeHyperbolicForm()
    assert [form.name for form in assignment.forms_of("H", "rule_position")] == HYPERBOLA_FIRST
    assert assignment.form_of("F", "length") == LinearForm()
    assert [form.name for form in assignment.forms_of("F", "answer_share")] == [
        "exponential",
        "linear",
        "hyperbolic",
    ]
    assert assignment.form_of("F", "density") == LogLinearLoadForm(2.0**-10)
    # The saturation is not an alternative of any block.
    assert all(
        form.name != "saturation" for forms in assignment.blocks.values() for form in forms
    )
    assert assignment.spread_form_of("P_sync", "M").name == "fractional_hyperbolic"
    assert assignment.spread_form_of("F", "V").name == "fractional_hyperbolic"
    assert assignment.spread_form_of("H", "P") == LogarithmicForm()


def test_the_shipped_assignment_names_every_block_and_every_axis() -> None:
    assignment = build_law_assignment(load_law_assignment_config(SHIPPED))
    assert len(assignment.blocks) == 13
    assert len(assignment.spread) == 9


@pytest.mark.parametrize(
    "text",
    [MINIMAL.replace("format_version: 2", "format_version: 1"), MINIMAL + "extra: 1\n"],
)
def test_a_file_outside_the_schema_is_a_configuration_error(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError) as error:
        load_law_assignment_config(_write(tmp_path, text))
    assert error.value.error_code == LAW_ASSIGNMENT_CONFIG_INVALID


def test_an_unknown_axis_of_the_spread_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as error:
        load_law_assignment_config(_write(tmp_path, MINIMAL.replace("{V:", "{Q:")))
    assert error.value.error_code == LAW_ASSIGNMENT_CONFIG_INVALID


def test_a_missing_file_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as error:
        load_law_assignment_config(tmp_path / "absent.yaml")
    assert error.value.error_code == LAW_ASSIGNMENT_CONFIG_INVALID


def test_an_unknown_form_is_rejected_when_the_assignment_is_built(tmp_path: Path) -> None:
    text = MINIMAL.replace("{forms: [logarithmic]}", "{forms: [cubic]}")
    config = load_law_assignment_config(_write(tmp_path, text))
    with pytest.raises(ConfigError) as error:
        build_law_assignment(config)
    assert error.value.error_code == LAW_FORM_UNKNOWN


def test_a_block_or_an_axis_the_assignment_does_not_name_is_a_contract_violation(
    tmp_path: Path,
) -> None:
    assignment = build_law_assignment(load_law_assignment_config(_write(tmp_path, MINIMAL)))
    assert assignment.form_of("H", "table_size") == LogarithmicForm()
    with pytest.raises(ContractViolationError) as block_error:
        assignment.form_of("H", "length")
    with pytest.raises(ContractViolationError) as axis_error:
        assignment.spread_form_of("H", "M")
    assert block_error.value.error_code == LAW_BLOCK_UNKNOWN
    assert axis_error.value.error_code == LAW_BLOCK_UNKNOWN
