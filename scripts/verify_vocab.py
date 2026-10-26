"""Verify the vocabulary V against its reference version.

Usage (from the repository of the public part, with the package installed):

    python scripts/verify_vocab.py
    python scripts/verify_vocab.py --vocab-id v1_intersection
    python scripts/verify_vocab.py --skip-provenance   # do not compare library versions

A thin wrapper over ``iec.vocab.VocabVerifier``: the checks live in the package because
the measuring code calls them too. Exit code: 0 - the vocabulary matches its reference;
1 - a mismatch or an error. Neither the network nor the NLTK corpora are needed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from iec.infrastructure.exceptions import IECError
from iec.vocab import VerificationResult, VocabVerifier

DEFAULT_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets" / "vocab"
DEFAULT_VOCAB_ID = "v1_intersection"


def print_report(result: VerificationResult, *, provenance_checked: bool) -> None:
    """Print the verification report to stdout."""
    vocab = result.vocab
    print(f"vocab_id : {vocab.vocab_id}")
    print(f"size     : {vocab.size}")
    print(f"hash     : {vocab.hash}")
    print("integrity: OK (recomputed hash matches the artifact)")

    if result.mismatches:
        print("reference: MISMATCH")
        for problem in result.mismatches:
            print(f"  - {problem}")
    else:
        built_at = result.reference_built_at or "unknown date"
        print(f"reference: OK (matches the version fixed on {built_at})")

    if provenance_checked:
        if result.provenance_drift:
            # Not an error: the vocabulary is intact and matches its reference, but a
            # rebuild in this environment may give another result.
            print("provenance: DRIFT (artifact is valid; rebuild may differ)")
            for item in result.provenance_drift:
                print(f"  - {item}")
        else:
            print("provenance: OK (library versions match the reference)")

    if result.is_valid:
        print("\nRESULT: PASSED - artifact matches the reference version")
    else:
        print("\nRESULT: FAILED - artifact does not match the reference version")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Verify the vocabulary V against its reference version.",
    )
    parser.add_argument(
        "--vocab-id",
        default=DEFAULT_VOCAB_ID,
        help=f"vocabulary to verify (default {DEFAULT_VOCAB_ID})",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ASSETS_DIR,
        help=f"folder of vocabularies and references (default {DEFAULT_ASSETS_DIR})",
    )
    parser.add_argument(
        "--skip-provenance",
        action="store_true",
        help="do not compare library versions with the reference",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command; return 0 when the vocabulary matches its reference."""
    args = parse_args(argv)
    check_provenance = not args.skip_provenance
    try:
        result = VocabVerifier(args.assets_dir).verify(
            args.vocab_id,
            check_provenance=check_provenance,
        )
    except IECError as error:
        print(f"vocab verification failed [{error.error_code}]: {error.message}", file=sys.stderr)
        return 1

    print_report(result, provenance_checked=check_provenance)
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
