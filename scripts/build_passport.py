"""Build the passport of a published measurement.

Usage (from the repository of the public part, with the package installed):

    python scripts/build_passport.py artefacts/<measurement>
    python scripts/build_passport.py artefacts/<measurement> --out-dir my_passport

The command reads ``measurement.json`` and ``points.jsonl.gz`` of the measurement directory and
writes ``passport_data.json``, ``passport.md`` and ``figures/`` into ``--out-dir`` (by default
the measurement directory itself). The model is not called. Exit code: 0 - the passport is
written; 1 - an error of the bench, printed to stderr.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from iec.infrastructure.exceptions import IECError
from iec.passport.document.renderer import write_documents
from iec.passport.measurement_files import read_description, read_points, write_passport_data
from iec.passport.passport_data import build_passport_data


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
    return parser.parse_args(argv)


def print_summary(
    passport: Mapping[str, Any], data_path: Path, passport_md: Path, figures: Mapping[str, Path]
) -> None:
    """Print what was computed for every model of the passport and where it was written."""
    print("passport built")
    print(f"  measurement:   {passport['model']['experiment_id']} (model {passport['model']['id']})")
    for model, fit in passport["fit"].items():
        section = passport["passport"][model]
        approximation = section.get("approximation") or {}
        print(f"  model:         {model}")
        print(
            f"  fit:           {fit['n_runs']} points, {fit['n_trials']} trials, "
            f"{section['experiment']['parameters']} parameters, "
            f"{section['experiment']['excluded_parameters']} excluded, sigma2 {fit['sigma2']}"
        )
        print(f"  approximation: {approximation.get('status')}")
        for row in approximation.get("public") or []:
            if row["key"] == "iec_capacity":
                print(
                    f"  IEC q={row['level']:g} {row['method']}: {row['value']} "
                    f"(SE {row['se']}, {row['status']})"
                )
    print(f"  written:       {data_path}, {passport_md}")
    print(f"  figures:       {', '.join(sorted(figures)) or 'none'}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command; return 0 when the passport is written."""
    args = parse_args(argv)
    directory = args.measurement_dir.resolve()
    out_dir = (args.out_dir or directory).resolve()
    # The validator writes one INFO line per point; the summary below says what matters.
    logging.disable(logging.INFO)
    try:
        passport = build_passport_data(
            read_description(directory), read_points(directory), directory.name
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        data_path = write_passport_data(passport, out_dir)
        passport_md, figures = write_documents(passport, out_dir)
    except IECError as error:
        print(f"passport build failed [{error.error_code}]: {error.message}", file=sys.stderr)
        return 1
    finally:
        logging.disable(logging.NOTSET)
    print_summary(passport, data_path, passport_md, figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
