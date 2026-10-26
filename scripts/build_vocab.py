"""Build the vocabulary V from its YAML config.

Usage (from the repository of the public part, with the package installed):

    python scripts/build_vocab.py
    python scripts/build_vocab.py --config config/vocab.yaml --assets-dir my_vocab

The command reads the config, loads the tokenizers by their ``provider``, takes the
intersection of the single-token word sets and saves ``<assets-dir>/<vocab_id>.json``. The
build summary is printed to stdout; the word counts after every filter and per tokenizer
go to the package log (stderr). The first run downloads the NLTK corpora and the Hugging
Face tokenizer files.

A changed vocabulary is a new version: if ``<vocab_id>.json`` is already saved with another
hash, the command refuses to overwrite it and asks for a new ``vocab_id``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from iec.contracts import Vocab
from iec.infrastructure.exceptions import ConfigError, IECError, VocabError
from iec.infrastructure.logging import LogContext, get_logger
from iec.model_client import Tokenizer, build_tiktoken_tokenizer
from iec.model_client.hf_tokenizer import build_hf_tokenizer
from iec.vocab import TokenizerSpec, VocabBuildConfig, VocabBuilder, VocabStore, load_vocab_config

_MODULE_NAME = "build_vocab"
_PUBLIC_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = _PUBLIC_ROOT / "config" / "vocab.yaml"
DEFAULT_ASSETS_DIR = _PUBLIC_ROOT / "assets" / "vocab"

TokenizerFactory = Callable[[str], Tokenizer]

# Registry of tokenizer factories; the key equals the ``provider`` field of TokenizerSpec.
# Adding a tokenizer = an adapter behind the Tokenizer contract, a line here, a line in the
# config - and a new vocab_id, because a new tokenizer set gives a new vocabulary version.
TOKENIZER_FACTORIES: dict[str, TokenizerFactory] = {
    "tiktoken": build_tiktoken_tokenizer,
    "hf": build_hf_tokenizer,
}

_logger = get_logger(LogContext(module_name=_MODULE_NAME))


def build_tokenizers(specs: Sequence[TokenizerSpec]) -> list[Tokenizer]:
    """Load the tokenizers of the config in their order.

    Raises:
        ConfigError: ``UNKNOWN_TOKENIZER_PROVIDER`` - the provider is not in the registry.
        ProviderError: a factory could not load its tokenizer (not suppressed).

    Side effects:
        The first load of a Hugging Face tokenizer downloads its files from the Hub.
    """
    tokenizers: list[Tokenizer] = []
    for spec in specs:
        factory = TOKENIZER_FACTORIES.get(spec.provider)
        if factory is None:
            raise ConfigError(
                f"unknown tokenizer provider '{spec.provider}' "
                f"(known: {sorted(TOKENIZER_FACTORIES)})",
                error_code="UNKNOWN_TOKENIZER_PROVIDER",
            )
        _logger.info(f"loading tokenizer {spec.provider}/{spec.name}")
        tokenizers.append(factory(spec.name))
    return tokenizers


def save_new_version(store: VocabStore, vocab: Vocab) -> Path:
    """Save the vocabulary unless another vocabulary is saved under the same id.

    Saving the same vocabulary again is allowed. A damaged existing file is not
    overwritten: its load error is raised.

    Raises:
        VocabError: ``VOCAB_VERSION_CONFLICT`` - ``<vocab_id>.json`` exists with another
            hash; a changed vocabulary needs a new ``vocab_id``.
    """
    if store.exists(vocab.vocab_id):
        existing = store.load(vocab.vocab_id)
        if existing.hash != vocab.hash:
            raise VocabError(
                f"vocab '{vocab.vocab_id}' is already saved with hash {existing.hash}, "
                f"the new build has hash {vocab.hash}: a changed vocabulary is a new "
                "version - set a new vocab_id in the config",
                error_code="VOCAB_VERSION_CONFLICT",
            )
    return store.save(vocab)


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
    """Build the vocabulary of the config and save it as a new or the same version."""
    tokenizers = build_tokenizers(config.tokenizer_specs)
    vocab = VocabBuilder(config, tokenizers).build()
    path = save_new_version(VocabStore(assets_dir), vocab)
    print_summary(vocab, path)
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command; return 0 on success and 1 on an error of the bench."""
    args = parse_args(argv)
    try:
        config = load_vocab_config(args.config)
        build_and_save(config, args.assets_dir)
    except IECError as error:
        print(f"vocab build failed [{error.error_code}]: {error.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
