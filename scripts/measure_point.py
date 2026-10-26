"""Measure one point: build the task, call the model, validate, append the record.

Usage (from the repository of the public part):

    python scripts/measure_point.py --model gpt-4o-mini-2024-07-18 --tokenizer gpt-4o-mini \
        --max-tokens 16384 --D 16 --n 5 --m 8 --seed-rules 1 --seed-text 2 \
        --out my_points.jsonl

The API key is read from the environment variable OPENAI_API_KEY; a .env file found in
the current directory or above it is loaded first. The record of the point is appended
to ``--out`` as one JSON line; the status of the point is printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from iec.contracts import GenParams, ModelConfig, PointConditions
from iec.model_client import build_default_model_builder
from iec.point_execution import PointExecutor, append_record
from iec.vocab import VocabStore

_DEFAULT_VOCAB_DIR = Path(__file__).resolve().parents[1] / "assets" / "vocab"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure one point of the IEC protocol.")
    parser.add_argument("--provider", default="openai", help="provider key: openai or mock")
    parser.add_argument("--model", required=True, help="model id passed to the provider")
    parser.add_argument("--tokenizer", default=None, help="tokenizer name; default: model id")
    parser.add_argument(
        "--max-tokens", type=int, required=True, help="answer reserve passed to the provider"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gen-seed", type=int, default=None, help="provider sampling seed")
    parser.add_argument("--context-window", type=int, default=None)
    parser.add_argument("--response-ceiling", type=int, default=None)
    parser.add_argument("--vocab-dir", type=Path, default=_DEFAULT_VOCAB_DIR)
    parser.add_argument("--vocab-id", default="v1_intersection")
    parser.add_argument("--D", type=int, required=True, help="distinct words of the text")
    parser.add_argument("--n", type=int, required=True, help="occurrences of each word")
    parser.add_argument("--m", type=int, required=True, help="number of rules")
    parser.add_argument("--seed-rules", type=int, required=True)
    parser.add_argument("--seed-text", type=int, required=True)
    parser.add_argument("--seed-order", type=int, default=0, help="0 keeps the drawn order")
    parser.add_argument("--out", type=Path, required=True, help="JSON-lines file to append to")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command; return the process exit code (0 - the record is written)."""
    args = _parser().parse_args(argv)
    load_dotenv(find_dotenv(usecwd=True))
    config = ModelConfig(
        model_id=args.model,
        provider=args.provider,
        gen_params=GenParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            seed=args.gen_seed,
        ),
        tokenizer_name=args.tokenizer,
        context_window_tokens=args.context_window,
        response_ceiling_tokens=args.response_ceiling,
    )
    client, tokenizer = build_default_model_builder().build(config)
    vocab = VocabStore(args.vocab_dir).load(args.vocab_id)
    conditions = PointConditions(
        D=args.D,
        n=args.n,
        m=args.m,
        seed_rules=args.seed_rules,
        seed_text=args.seed_text,
        seed_order=args.seed_order,
    )
    outcome = PointExecutor(client, tokenizer, vocab, config).execute(conditions)
    append_record(args.out, outcome.record)
    print(f"status={outcome.record.status.value} record appended to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
