"""Mirror tests of the result of the alignment."""

import numpy as np

from iec.answer_validator.alignment.aligned_answer import (
    GROUP_ANCHOR,
    GROUP_CODES,
    GROUP_DELETED,
    KIND_DELETED,
    KIND_MATCHED,
    PLACE_DELETED,
    PLACE_MATCHED,
    AlignedAnswer,
    Place,
)


def test_places_are_built_from_the_arrays() -> None:
    aligned = AlignedAnswer(
        kinds=np.array([KIND_MATCHED, KIND_DELETED], dtype=np.int8),
        reference_positions=np.array([0, 1], dtype=np.int64),
        answer_positions=np.array([0, -1], dtype=np.int64),
        groups=np.array([GROUP_CODES[GROUP_ANCHOR], GROUP_CODES[GROUP_DELETED]], dtype=np.int8),
        core_words=("lake",),
        core_line_sizes=np.array([1], dtype=np.int64),
        answer_words=1,
        wrapper_lines=0,
        episodes=1,
        pairings=(),
    )

    assert aligned.places == (
        Place(PLACE_MATCHED, 0, 0, "lake", GROUP_ANCHOR),
        Place(PLACE_DELETED, 1, None, None, GROUP_DELETED),
    )
