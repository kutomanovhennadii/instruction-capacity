"""Verify a published measurement: files, vocabulary, tasks of the points and the passport.

Usage (from the repository of the public part, with the package installed):

    python scripts/verify.py artefacts/<measurement>
    python scripts/verify.py artefacts/<measurement> --vocab-dir assets/vocab

Four checks: the SHA-256 of the files listed in ``measurement.json``; the vocabulary against
its reference version; the rule list and the source text of every point rebuilt from its
seeds; the passport recomputed from the points against ``passport_data.json`` (the composition
exactly, the numbers within the tolerance of computation). Exit code: 0 - all four passed;
1 - a check failed or an error occurred.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from iec.infrastructure.exceptions import IECError
from iec.passport.verify import VerificationReport, verify_measurement

DEFAULT_VOCAB_DIR = Path(__file__).resolve().parents[1] / "assets" / "vocab"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Verify a published measurement.")
    parser.add_argument("measurement_dir", type=Path, help="published measurement directory")
    parser.add_argument(
        "--vocab-dir",
        type=Path,
        default=DEFAULT_VOCAB_DIR,
        help=f"folder of vocabularies and their references (default {DEFAULT_VOCAB_DIR})",
    )
    return parser.parse_args(argv)


def print_report(report: VerificationReport) -> None:
    """Print every check with its differences and the result."""
    for check in report.checks:
        print(f"{check.name}: {'PASSED' if check.passed else 'FAILED'}")
        for line in check.details:
            print(f"  - {line}")
    print()
    print("RESULT: PASSED" if report.passed else "RESULT: FAILED")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command; return 0 when every check passed."""
    args = parse_args(argv)
    # The validator writes one INFO line per point; the report below says what matters.
    logging.disable(logging.INFO)
    try:
        report = verify_measurement(args.measurement_dir.resolve(), args.vocab_dir.resolve())
    except IECError as error:
        print(f"verification failed [{error.error_code}]: {error.message}", file=sys.stderr)
        return 1
    finally:
        logging.disable(logging.NOTSET)
    print_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
