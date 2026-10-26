# vocab — building and verifying the vocabulary `V`

The vocabulary `V` is the set of English words from which the rule pairs and the background
of the source text are drawn. This layer builds it, stores it as a versioned file and checks
that a file is exactly the published version. The measurement vocabulary is
`public/assets/vocab/v1_intersection.json` with its version reference
`v1_intersection.reference.yaml` next to it.

## What is here

- `word_source.py` — the only boundary of the layer with the NLTK corpora (CMU Pronouncing
  Dictionary and WordNet). It returns a plain set of strings, so the builder core does not
  depend on NLTK and is tested without the network.
- `vocab_builder.py` — the core: the single-token filter per tokenizer, the intersection of
  the sets, assembling `Vocab` with its hash.
- `vocab_config.py` — `VocabBuildConfig` and `TokenizerSpec` (Pydantic v2), loading from YAML.
- `vocab_store.py` — saving and loading `Vocab` as JSON with an integrity check by the hash.
- `vocab_verifier.py` — checking a file against the reference of its version.

Entry points: `scripts/build_vocab.py` builds, `scripts/verify_vocab.py` checks; the config
is `config/vocab.yaml`.

## Build pipeline

```text
word_source.load_noun_monosyllables(config)     # corpus -> candidates
    lowercase ASCII -> length >= MIN_WORD_LENGTH -> <= max_syllables syllables
    -> nouns (WordNet) -> singular (lemmatizer)
VocabBuilder.build()                            # candidates -> V
    single token for each tokenizer -> intersection -> hash -> Vocab
```

Every step logs the word count before and after itself, and the single-token filter logs
the size of the set per tokenizer. These numbers are the control funnel of the reference, so
the step logs are a result, not debugging noise.

## Two non-obvious things in the code

**The single-token probe uses a leading space.** `VocabBuilder` checks not `"sword"` but
`" sword"` (the constant `PROBE_PREFIX`), while `Vocab` stores the word without the space.
In the BPE vocabularies of GPT and Qwen the vocabulary unit is the word after a separator,
and that is the form in which the word stands in the protocol prompt; the bare form of many
words splits into two tokens. Measured on 5871 monosyllabic nouns: 1886 single-token words
without the space against 3284 with it. The substitution is made by the builder, not by the
tokenizer adapter: the `Tokenizer` contract checks the given string as is.

**Thresholds are fixed in the code, not in the config.** `MIN_WORD_LENGTH = 3`,
`PROBE_PREFIX`, the part of speech and the number are constants and steps of the pipeline.
`config/vocab.yaml` holds only `vocab_id`, `source`, `max_syllables`, `ascii_lower_only` and
the tokenizer set. The instrument has one mode of operation; otherwise one would have to
recover from a file which mode built it.

## Version check

A vocabulary file comes with the **reference of its version** —
`assets/vocab/{vocab_id}.reference.yaml`, filled in once when the version is fixed. It answers
the question "is this the data on which the published results were obtained".

`VocabVerifier` checks three levels:

| Level | What it catches | Reaction |
|---|---|---|
| Integrity (`VocabStore.load`) | damage and manual edits of the file | `VocabError` |
| Match with the reference | a substituted version: a rebuilt vocabulary is internally consistent, but its hash differs | mismatches in `VerificationResult.mismatches` |
| Environment | drift of library versions | `provenance_drift` — a warning, the file stays valid |

```python
# Measuring code: a mismatch is an error, numbers on another vocabulary are not comparable.
vocab = VocabVerifier(Path("assets/vocab")).require_valid("v1_intersection")

# A person: a detailed report without an exception, the caller decides.
result = VocabVerifier(Path("assets/vocab")).verify("v1_intersection")
```

The check needs neither the network, nor the NLTK corpora, nor tokenizers.

## Key signatures

```python
load_noun_monosyllables(config: VocabBuildConfig) -> set[str]
VocabBuilder(config: VocabBuildConfig, tokenizers: Sequence[Tokenizer], *,
             word_source: WordSource | None = None)
VocabBuilder.build() -> Vocab
load_vocab_config(path: Path) -> VocabBuildConfig
VocabStore(base_dir: Path)
VocabStore.save(vocab: Vocab) -> Path
VocabStore.load(vocab_id: str) -> Vocab
VocabStore.exists(vocab_id: str) -> bool
VocabVerifier(base_dir: Path)
VocabVerifier.verify(vocab_id: str, *, check_provenance: bool = True) -> VerificationResult
VocabVerifier.require_valid(vocab_id: str, *, check_provenance: bool = False) -> Vocab
```

The `word_source` parameter exists for tests: with a substitute source the core is checked
on an arbitrary word set without NLTK and the network.

## Errors

| Code | When |
|---|---|
| `VOCAB_SOURCE_UNAVAILABLE` | an NLTK corpus is unavailable or empty |
| `VOCAB_NO_TOKENIZERS` | the tokenizer list is empty — the intersection is undefined |
| `VOCAB_EMPTY_SOURCE` | the source gave no candidates |
| `VOCAB_EMPTY_INTERSECTION` | the intersection of single-token sets is empty |
| `VOCAB_NOT_FOUND` | the vocabulary file is missing |
| `VOCAB_FILE_INVALID` | the vocabulary file is damaged or incomplete |
| `VOCAB_HASH_MISMATCH` | the recomputed hash differs from the stored one |
| `VOCAB_REFERENCE_MISMATCH` | the file does not match its version reference (`require_valid`) |
| `VOCAB_REFERENCE_INVALID` | the version reference is missing or invalid (`ConfigError`) |
| `VOCAB_CONFIG_INVALID` | the build config is missing or violates the schema (`ConfigError`) |
| `UNKNOWN_TOKENIZER_PROVIDER` | the config names a provider absent from the registry of `scripts/build_vocab.py` (`ConfigError`) |
| `VOCAB_VERSION_CONFLICT` | `scripts/build_vocab.py` refuses to overwrite a saved vocabulary with another hash under the same `vocab_id` |

A tokenizer error during the word check is not suppressed and surfaces as `ProviderError`:
silently skipping words would change the vocabulary unnoticed.

## Adding a tokenizer

1. An adapter behind the `Tokenizer` contract in `iec.model_client`.
2. A line `provider -> factory` in `TOKENIZER_FACTORIES` of `scripts/build_vocab.py`.
3. A line in `config/vocab.yaml`.
4. A **new `vocab_id`** and a new reference file: a new tokenizer set gives a new vocabulary
   version. `v1_intersection` stays as it is; the build command refuses to overwrite it
   with another composition.

## Boundaries

The layer does not generate rules, text or prompts and does not score answers. From
`iec.model_client` it needs only the `Tokenizer` contract and the tokenizer factories.

Method: [The vocabulary](../../../docs/02-vocabulary.md).
