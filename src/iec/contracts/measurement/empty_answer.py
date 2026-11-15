"""The empty answer: an answered point whose answer holds no word.

A model may answer with nothing - an empty text or white space only. Such an answer carries no
replacement to score: the point executor calls the model once more and records a second empty
answer as a failure, and the readers of the measurements bought before this rule - the passport and
the models of the purchase - skip such records by this one function.
"""

from __future__ import annotations

from iec.contracts.measurement.point_record import PointRecord, PointStatus


def has_no_words(text: str | None) -> bool:
    """Return whether a text holds no word: it is missing, empty or white space only.

    ``str.isspace`` judges white space by the same rule as ``str.split`` and stops at the first
    character that is not white space, so a long answer is not split into words only to learn that it
    has some.
    """
    return text is None or not text or text.isspace()


def is_empty_answer(record: PointRecord) -> bool:
    """Return whether a record is an answered point whose answer holds no word.

    An answered point without an answer at all is a broken record, not an empty answer: its readers
    report it as such.
    """
    return (
        record.status is PointStatus.OK
        and record.answer is not None
        and has_no_words(record.answer)
    )
