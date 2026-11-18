"""The saved fit of a passport: the first stage of the three quantities as a JSON file.

A passport is computed in two stages (``passport_data``): the fits of the three quantities from the
trials of the points, and the laws, levels and capacities over the fits. The second stage reads of a
fit its coefficients, their full covariance, the variance of the request shift and the law of the
spread, and of the design of a quantity the bins of its axes, the cells of its grid and their
abscissae (``quantities.steps.section_fit``). This module writes the first stage into a JSON file and
reads it back, so a change of the laws recomputes the passport from the file in seconds. JSON keeps a
floating-point number exactly, so the passport from the file is that of the full build to the last
digit. The file carries ``format_version``; a newer version is refused.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from iec.contracts.format_version import check_format_version
from iec.infrastructure.exceptions import PassportError
from iec.infrastructure.files import write_text_atomically
from iec.passport.fitting.chain.fit_notes import FitNote
from iec.passport.fitting.load_spread import LoadSpread, SpreadAxis
from iec.passport.fitting.variable_fit import VariableFit
from iec.passport.quantities.background_design import AnswerShareDesign, BinTally
from iec.passport.quantities.steps.section_fit import SectionFit
from iec.passport.trials.design.folded_design import AxisBin, AxisTable, GridCell, VariableDesign
from iec.passport.trials.layout.bin_layout import BinLayout

#: The version of the format of the saved fit that this code writes.
FITS_FORMAT_VERSION = 1
#: Error code of a file that is not a saved fit.
PASSPORT_FITS_INVALID = "PASSPORT_FITS_INVALID"
#: The kinds of the design of a quantity in the file.
FOLDED, ANSWER_SHARE = "folded", "answer_share"

T = TypeVar("T")


@dataclass(frozen=True, slots=True, eq=False)
class PassportFits:
    """The first stage of a passport: the facts of the measurement, the layout and the three fits."""

    model_id: str
    name: str
    points: int
    weighting: Mapping[str, Any]
    layout: BinLayout
    sections: Mapping[str, SectionFit]


def _plain(value: Any) -> Any:
    """Return a value with lists for tuples and arrays and plain numbers: its JSON form."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _tupled(value: Any) -> Any:
    """Return a value with tuples for lists: the form of the frozen data classes."""
    if isinstance(value, list):
        return tuple(_tupled(item) for item in value)
    return value


def _record(item: Any) -> dict[str, Any]:
    """Return the JSON form of a data class whose fields are plain values."""
    return {field.name: _plain(getattr(item, field.name)) for field in fields(item)}


def _flat(cls: type[T], data: Mapping[str, Any]) -> T:
    """Return a data class whose fields are plain values from its JSON form."""
    return cls(**{field.name: _tupled(data[field.name]) for field in fields(cls)})  # type: ignore[arg-type]


def _fit_record(fit: VariableFit) -> dict[str, Any]:
    """Return the JSON form of a fit: its plain fields, the law of the spread and the notes."""
    record = {
        field.name: _plain(getattr(fit, field.name))
        for field in fields(fit)
        if field.name not in ("spread", "notes")
    }
    spread = fit.spread
    record["spread"] = (
        None
        if spread is None
        else {"gamma": _plain(spread.gamma), "axes": [_record(axis) for axis in spread.axes]}
    )
    record["notes"] = [_record(note) for note in fit.notes]
    return record


def _spread_of(data: Mapping[str, Any] | None) -> LoadSpread | None:
    """Return the law of the spread from its JSON form; ``None`` - a fit without it."""
    if data is None:
        return None
    length, density, rules = (_flat(SpreadAxis, axis) for axis in data["axes"])
    return LoadSpread(_tupled(data["gamma"]), (length, density, rules))


def _fit_of(data: Mapping[str, Any]) -> VariableFit:
    """Return a fit from its JSON form."""
    plain = {
        field.name: _tupled(data[field.name])
        for field in fields(VariableFit)
        if field.name not in ("spread", "notes")
    }
    return VariableFit(
        **plain,
        spread=_spread_of(data["spread"]),
        notes=tuple(_flat(FitNote, note) for note in data["notes"]),
    )


def _design_record(design: VariableDesign) -> dict[str, Any]:
    """Return the JSON form of a design without the rows of its fit."""
    return {
        "variable": design.variable,
        "names": list(design.names),
        "cells": [_record(cell) for cell in design.cells],
        "axes": {
            letter: {
                "letter": table.letter,
                "reference": table.reference,
                "bins": [_record(item) for item in table.bins],
            }
            for letter, table in design.axes.items()
        },
        "abscissae": _plain(design.abscissae),
    }


def _design_of(data: Mapping[str, Any]) -> VariableDesign:
    """Return a design without the rows of its fit from its JSON form."""
    names = tuple(data["names"])
    return VariableDesign(
        variable=data["variable"],
        x=np.zeros((0, len(names))),
        failures=np.zeros(0),
        trials=np.zeros(0),
        run=np.zeros(0, dtype=np.int64),
        point_keys=(),
        names=names,
        cells=tuple(_flat(GridCell, cell) for cell in data["cells"]),
        axes={
            letter: AxisTable(
                table["letter"],
                tuple(_flat(AxisBin, item) for item in table["bins"]),
                table["reference"],
            )
            for letter, table in data["axes"].items()
        },
        abscissae={name: dict(point) for name, point in data["abscissae"].items()},
    )


def _tally_record(tally: BinTally) -> dict[str, Any]:
    """Return the JSON form of the tally of a bin: the weights of its requests with failures in order."""
    return {
        "trials": tally.trials,
        "failures": tally.failures,
        "weighted": tally.weighted,
        "total": tally.total,
        "failing": None if tally.failing is None else list(tally.failing.values()),
    }


def _tally_of(data: Mapping[str, Any]) -> BinTally:
    """Return the tally of a bin from its JSON form; the requests are numbered in their order."""
    failing = data["failing"]
    return BinTally(
        data["trials"],
        data["failures"],
        data["weighted"],
        data["total"],
        None if failing is None else {str(place): weight for place, weight in enumerate(failing)},
    )


def _section_record(section: SectionFit) -> dict[str, Any]:
    """Return the JSON form of the first stage of a quantity."""
    design = section.design
    if isinstance(design, AnswerShareDesign):
        shaped: dict[str, Any] = {
            "kind": ANSWER_SHARE,
            "design": _design_record(design.design),
            "bins": {
                letter: {name: _tally_record(tally) for name, tally in tallies.items()}
                for letter, tallies in design.bins.items()
            },
            "centers": _plain(design.centers),
            "references": dict(design.references),
            "largest_words": design.largest_words,
        }
    else:
        shaped = {"kind": FOLDED, "design": _design_record(design)}
    return {**shaped, "fit": _fit_record(section.fit), "support": section.support}


def _section_of(data: Mapping[str, Any]) -> SectionFit:
    """Return the first stage of a quantity from its JSON form."""
    folded = _design_of(data["design"])
    design: VariableDesign | AnswerShareDesign = folded
    if data["kind"] == ANSWER_SHARE:
        design = AnswerShareDesign(
            design=folded,
            bins={
                letter: {name: _tally_of(tally) for name, tally in tallies.items()}
                for letter, tallies in data["bins"].items()
            },
            centers={letter: dict(centers) for letter, centers in data["centers"].items()},
            references=dict(data["references"]),
            largest_words=int(data["largest_words"]),
        )
    return SectionFit(design, _fit_of(data["fit"]), data["support"])


def write_passport_fits(fits: PassportFits, path: Path) -> Path:
    """Write the saved fit of a passport, whole or not at all.

    Args:
        fits: The first stage of the passport.
        path: The JSON file.

    Returns:
        The path written.
    """
    data = {
        "format_version": FITS_FORMAT_VERSION,
        "model_id": fits.model_id,
        "name": fits.name,
        "points": fits.points,
        "weighting": _plain(fits.weighting),
        "layout": _record(fits.layout),
        "sections": {quantity: _section_record(section) for quantity, section in fits.sections.items()},
    }
    write_text_atomically(path, json.dumps(data, ensure_ascii=False) + "\n")
    return path


def read_passport_fits(path: Path) -> PassportFits:
    """Read the saved fit of a passport.

    Raises:
        PassportError: ``PASSPORT_FITS_INVALID`` - the file is missing or is not JSON.
        ContractViolationError: ``PASSPORT_FITS_INVALID`` - the file is not a JSON object or its
            version is broken; ``FORMAT_VERSION_UNSUPPORTED`` - its version is newer than this code.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PassportError(
            f"the saved fit {path} cannot be read: {error}", error_code=PASSPORT_FITS_INVALID
        ) from error
    check_format_version(
        data, FITS_FORMAT_VERSION, invalid_code=PASSPORT_FITS_INVALID, what="the saved fit"
    )
    return PassportFits(
        model_id=data["model_id"],
        name=data["name"],
        points=int(data["points"]),
        weighting=data["weighting"],
        layout=_flat(BinLayout, data["layout"]),
        sections={quantity: _section_of(section) for quantity, section in data["sections"].items()},
    )
