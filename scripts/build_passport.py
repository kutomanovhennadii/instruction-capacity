"""Build the passport of a published measurement.

Usage (from the repository of the public part, with the package installed):

    python scripts/build_passport.py artefacts/<measurement>
    python scripts/build_passport.py artefacts/<measurement> --out-dir my_passport
    python scripts/build_passport.py artefacts/<measurement> --laws config/passport_laws.yaml --figures config/passport_figures.yaml --workers 8
    python scripts/build_passport.py artefacts/<measurement> --save-fits fits.json
    python scripts/build_passport.py artefacts/<measurement> --from-fits fits.json

The command reads ``measurement.json`` and the points of the measurement directory, ``points.jsonl.gz``
or its numbered parts ``points-001.jsonl.gz``, ``points-002.jsonl.gz`` and so on, aligns
every answer with its reference, fits the three quantities with the forms of the laws of ``--laws``
and writes ``passport_data.json``, ``passport.md`` and ``figures/`` by the plan of ``--figures`` into
``--out-dir`` (by default the measurement directory itself). The points and the quantities are
computed in a pool of ``--workers`` processes (by default one per processor; 1 - without a pool);
the passport is the same either way. With ``--save-fits`` the command also writes the saved fit of
the passport, the first stage of its three quantities (``iec.passport.fit_store``); with
``--from-fits`` it reads one instead of the points and recomputes only the laws, the levels, the
capacities and the document, in seconds, the same passport to the last digit. Progress lines go to
the standard output. The model is not called. Exit code: 0 - the passport is written; 1 - an error of
the bench, printed to stderr.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from iec.infrastructure.exceptions import IECError
from iec.passport.document.charts.figure_plan import load_figure_plan
from iec.passport.document.renderer import write_documents
from iec.passport.fit_store import read_passport_fits, write_passport_fits
from iec.passport.laws.forms.law_assignment import build_law_assignment, load_law_assignment_config
from iec.passport.measurement_files import read_description, read_points, write_passport_data
from iec.passport.passport_data import (
    calculation_pool,
    measurement_trials,
    passport_fits,
    passport_from_fits,
)

DEFAULT_LAWS = Path(__file__).resolve().parents[1] / "config" / "passport_laws.yaml"
DEFAULT_FIGURES = Path(__file__).resolve().parents[1] / "config" / "passport_figures.yaml"
DEFAULT_WORKERS = os.cpu_count() or 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Build the passport of a published measurement.")
    parser.add_argument("measurement_dir", type=Path, help="published measurement directory")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where to write the passport (default: the measurement directory)",
    )
    parser.add_argument(
        "--laws",
        type=Path,
        default=DEFAULT_LAWS,
        help=f"the assignment of the forms of the laws (default {DEFAULT_LAWS})",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=DEFAULT_FIGURES,
        help=f"the plan of the figures of the passport (default {DEFAULT_FIGURES})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"worker processes of the calculation, 1 - without a pool (default {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--save-fits",
        type=Path,
        default=None,
        help="also write the saved fit of the passport, the first stage of its three quantities, to this JSON file",
    )
    parser.add_argument(
        "--from-fits",
        type=Path,
        default=None,
        help="compute the passport from a saved fit instead of the points: the laws, the capacities, the document",
    )
    return parser.parse_args(argv)


def print_summary(
    passport: Mapping[str, Any], data_path: Path, passport_md: Path, figures: Sequence[Path]
) -> None:
    """Print what was computed for every quantity of the passport and where it was written."""
    print("passport built")
    print(f"  measurement: {passport['model']['experiment_id']} (model {passport['model']['id']})")
    for name in ("P_sync", "H", "F"):
        section = passport[name]
        fit = section["fit"]
        print(
            f"  {name}: model {section['model']}, {fit['points']} points, {fit['trials']} trials, "
            f"{fit['failures']} failures, sigma2 {fit['sigma2']}, {section['status']}"
        )
        for row in section.get("capacities", ()):
            if row["method"] == "law":
                # The measure of the density has no level of quality.
                level = "" if row["level"] is None else f" q={row['level']:g}"
                print(f"    {row['key']}{level}: {row['value']} ({row['status']})")
    print(f"  written: {data_path}, {passport_md}")
    print(f"  figures: {len(figures)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command; return 0 when the passport is written."""
    args = parse_args(argv)
    directory = args.measurement_dir.resolve()
    out_dir = (args.out_dir or directory).resolve()
    started = time.monotonic()

    def progress(message: str) -> None:
        print(f"[{time.monotonic() - started:7.1f} s] {message}", flush=True)

    try:
        assignment = build_law_assignment(load_law_assignment_config(args.laws.resolve()))
        plan = load_figure_plan(args.figures.resolve())
        if args.from_fits is not None:
            fits = read_passport_fits(args.from_fits.resolve())
        else:
            description = read_description(directory)
            records = read_points(directory)
            with calculation_pool(args.workers) as executor:
                trials = measurement_trials(description, records, progress, executor=executor)
                fits = passport_fits(
                    description, records, trials, directory.name, progress, executor=executor
                )
            if args.save_fits is not None:
                write_passport_fits(fits, args.save_fits.resolve())
        passport = passport_from_fits(fits, assignment, progress)
        out_dir.mkdir(parents=True, exist_ok=True)
        data_path = write_passport_data(passport, out_dir)
        passport_md, figures = write_documents(passport, out_dir, assignment, plan)
    except IECError as error:
        print(f"passport build failed [{error.error_code}]: {error.message}", file=sys.stderr)
        return 1
    print_summary(passport, data_path, passport_md, figures)
    return 0


if __name__ == "__main__":
    logging.disable(logging.INFO)
    raise SystemExit(main())
