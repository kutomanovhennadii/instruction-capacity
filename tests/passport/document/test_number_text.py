"""Tests of the numbers of the text of the passport: a small parameter of a law never reads as zero."""

from iec.passport.document.number_text import number, parameter


def test_parameter_keeps_fixed_digits_for_ordinary_numbers() -> None:
    assert parameter(13.968368) == number(13.968368) == "13.97"
    assert parameter(-1.280721, 3) == "−1.281"
    assert parameter(0.0) == "0.00"


def test_parameter_keeps_two_significant_digits_of_a_small_number() -> None:
    assert parameter(0.0006516823860805488) == "0.00065"
    assert parameter(0.0019852869044342998) == "0.0020"
    assert parameter(-0.004) == "−0.0040"
    assert parameter(0.0004, 3) == "0.00040"


def test_parameter_writes_a_tiny_number_by_a_power_of_ten() -> None:
    assert parameter(1.234e-6) == "1.2·10⁻⁶"
    assert parameter(-5e-9) == "−5.0·10⁻⁹"
