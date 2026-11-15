"""Build a vocabulary V from its YAML config and write the reference of its version.

Usage (from the repository of the public part, with the package installed):

    python scripts/build_vocab.py
    python scripts/build_vocab.py --config config/vocab.yaml --assets-dir my_vocab
    python scripts/build_vocab.py --config config/vocab/v1_nemotron.yaml --assets-dir my_vocab

The command reads the config, loads the tokenizers by their ``provider`` from the tokenizer
registry of ``iec.model_client``, takes the intersection of the single-token word sets and saves
``<assets-dir>/<vocab_id>.json``. Next to it goes the reference of the version,
``<vocab_id>.reference.yaml``: its format version, the hash, the size, the source, the tokenizers,
the parameters, the number of candidates, the single-token words per tokenizer and the library
versions; a reference already there is kept as it is. The build summary is printed to stdout; the
word counts after every filter and per tokenizer go to the package log (stderr), with a line for
every 500 probes of a tokenizer. The first run downloads the NLTK corpora and the Hugging Face
tokenizer files.

The command is a thin entry point: the build, the guard of a version and the reference live in
the package ``iec.vocab``. The providers of tokenizers of the standard registry:

    tiktoken      the tokenizers of OpenAI, shipped with the library
    hf            a Hugging Face tokenizer through transformers
    hfjson        the file tokenizer.json of a Hugging Face repository
    gemini_count  the count of a Gemini model by countTokens, needs GEMINI_API_KEY

The key is read from the environment; a .env file found in the current directory or above it is
loaded first.

A changed vocabulary is a new version: if ``<vocab_id>.json`` is already saved with another
hash, the command refuses to overwrite it and asks for a new ``vocab_id``. To check a published
vocabulary, rebuild it into another folder and compare the printed hash with the hash in
``assets/vocab/<vocab_id>.reference.yaml``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from iec.contracts import Vocab
from iec.infrastructure.exceptions import IECError
from iec.infrastructure.logging import LogContext, attach_file_handler, get_logger
from iec.model_client import Tokenizer, TokenizerBuilder, build_default_tokenizer_builder
from iec.vocab import (
    CountingTokenizer,
    TokenizerSpec,
    VocabBuildConfig,
    VocabBuilder,
    VocabStore,
    load_vocab_config,
    write_reference,
)

_MODULE_NAME = "build_vocab"
_PUBLIC_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = _PUBLIC_ROOT / "config" / "vocab.yaml"
DEFAULT_ASSETS_DIR = _PUBLIC_ROOT / "assets" / "vocab"

_logger = get_logger(LogContext(module_name=_MODULE_NAME))


def build_tokenizers(
    specs: Sequence[TokenizerSpec], builder: TokenizerBuilder | None = None
) -> list[Tokenizer]:
    """Load the tokenizers of the config in their order through the tokenizer registry.

    Args:
        specs: The tokenizers of the config.
        builder: The registry of tokenizer factories; the standard registry by default.

    Returns:
        The tokenizers in the order of the config.

    Raises:
        ConfigError: ``UNKNOWN_TOKENIZER_PROVIDER`` - the provider is not in the registry;
            ``GEMINI_API_KEY_MISSING`` - the count of a Gemini model has no key.
        ProviderError: a factory could not load its tokenizer (not suppressed).

    Side effects:
        The first load of a Hugging Face tokenizer downloads its files from the Hub.
    """
    registry = builder if builder is not None else build_default_tokenizer_builder()
    tokenizers: list[Tokenizer] = []
    for spec in specs:
        _logger.info(f"loading tokenizer {spec.provider}/{spec.name}")
        tokenizers.append(registry.build(spec.provider, spec.name))
    return tokenizers


def print_summary(vocab: Vocab, path: Path) -> None:
    """Print the build summary to stdout; per-filter counts are in the package log."""
    print("vocab build complete")
    print(f"  vocab_id:   {vocab.vocab_id}")
    print(f"  size:       {vocab.size}")
    print(f"  hash:       {vocab.hash}")
    print(f"  source:     {vocab.source}")
    print(f"  tokenizers: {', '.join(vocab.tokenizer_names)}")
    print(f"  params:     {vocab.build_params}")
    print(f"  saved to:   {path}")


def build_and_save(config: VocabBuildConfig, assets_dir: Path) -> Vocab:
    """Build the vocabulary of the config, save it as a new or the same version, add the reference.

    Args:
        config: The build config.
        assets_dir: Folder to save the vocabulary and its reference to.

    Returns:
        The built vocabulary.

    Raises:
        IECError: a failure of the bench - an unknown provider, a missing key, an empty
            source or intersection, a version conflict, a tokenizer that does not load.

    Side effects:
        Writes ``<assets_dir>/<vocab_id>.json`` and, when absent, its reference; prints the
        summary to stdout.
    """
    counters = [CountingTokenizer(item) for item in build_tokenizers(config.tokenizer_specs)]
    vocab = VocabBuilder(config, counters).build()
    path = VocabStore(assets_dir).save_new_version(vocab)
    print_summary(vocab, path)
    reference, written = write_reference(vocab, counters, assets_dir)
    print(f"  reference:  {reference} ({'written' if written else 'kept as it is'})")
    for counter in counters:
        print(f"  {counter.name}: {counter.single} of {counter.probes} single-token")
    return vocab


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Build the vocabulary V (monosyllabic single-token nouns).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML build config (default {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ASSETS_DIR,
        help=f"folder to save the vocabulary to (default {DEFAULT_ASSETS_DIR})",
    )
    parser.add_argument("--log", type=Path, default=None, help="live log of the build")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command; return 0 on success and 1 on an error of the bench."""
    args = parse_args(argv)
    if args.log is not None:
        print(f"vocab build log: {attach_file_handler(args.log)}", flush=True)
    load_dotenv(find_dotenv(usecwd=True), override=False)
    try:
        config = load_vocab_config(args.config)
        build_and_save(config, args.assets_dir)
    except IECError as error:
        print(f"vocab build failed [{error.error_code}]: {error.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
