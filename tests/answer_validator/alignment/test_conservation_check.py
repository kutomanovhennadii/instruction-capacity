"""Mirror tests of the conservation of places."""

import pytest

from iec.answer_validator.alignment.aligned_answer import GROUP_ANCHOR, PLACE_MATCHED, Place
from iec.answer_validator.alignment.conservation_check import check_conservation
from iec.infrastructure.exceptions import ContractViolationError


def test_conservation_violation_is_a_contract_error() -> None:
    with pytest.raises(ContractViolationError) as caught:
        check_conservation(
            (Place(PLACE_MATCHED, 0, 0, "x", GROUP_ANCHOR),), reference_words=2, answer_words=1
        )

    assert caught.value.error_code == "ALIGNMENT_CONSERVATION"
