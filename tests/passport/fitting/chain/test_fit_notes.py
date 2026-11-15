"""The catalog of notes: every code is its own name, and the catalog lists them all."""

from __future__ import annotations

import logging

from iec.passport.fitting.chain import fit_notes
from iec.passport.fitting.chain.fit_notes import NOTE_CODES, FitNote


def test_every_code_is_its_own_name_and_the_catalog_lists_it() -> None:
    names = [name for name in vars(fit_notes) if name.isupper() and name != "NOTE_CODES"]

    assert all(getattr(fit_notes, name) == name for name in names)
    assert NOTE_CODES == frozenset(names)
    assert len(NOTE_CODES) == 12


def test_a_note_is_a_warning_unless_told_otherwise() -> None:
    assert FitNote("FIT_NO_DATA", "text").level == logging.WARNING
