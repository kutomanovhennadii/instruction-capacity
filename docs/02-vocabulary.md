# The vocabulary

The words of every rule and every text come from one fixed vocabulary, `v1_intersection`: 2220
lowercase English nouns of one syllable, each a single token for every tokenizer of the
reference set. The vocabulary is part of the instrument: any property that makes one word differ
from another more than the fact of being a rule word would leak into the measurement. Every
filter below answers one question: which property of a word could replace the measured quantity.

Files: `assets/vocab/v1_intersection.json` (the words), `assets/vocab/v1_intersection.reference.yaml`
(its reference), `config/vocab.yaml` (how it is built).

## Source

The candidates come from the **CMU Pronouncing Dictionary** (`cmudict.0.7a`, 123 455 entries)
through NLTK. Its phonetic transcription gives the number of syllables exactly: every vowel
phoneme carries a stress digit, so the syllables are the phonemes that end in a digit. Part of
speech and number come from **WordNet 3.0**, also through NLTK. Both corpora are frozen, so the
result does not drift with time.

## Filters

| Step | Words | Why |
|---|---:|---|
| CMUdict entries | 123 455 | |
| lowercase ASCII letters only | 115 533 | one written form: no capitals, hyphens, apostrophes or digits |
| at least three letters | 115 318 | drops letter names, clippings and function words (`in`, `or`, `be`) that also occur in the instruction itself |
| one syllable | 14 884 | short, frequent, uniform material |
| nouns in WordNet | 5 750 | one part of speech: a noun replaced by a noun keeps the grammar of the stream |
| singular: the word equals its noun lemma | 3 581 | no `arm` / `arms` pairs, which would turn a replacement into a test of morphology |
| single token for every tokenizer: the intersection | **2 220** | see below |

Proper names that WordNet lists in lowercase (`james`, `bach`) stay: as interchangeable labels
they are as good as any noun.

## Single-token words

A rule word must be exactly one token for every compared model. Otherwise the number of rules and
the length of the prompt in tokens would not be independent, and a word of two tokens could be
replaced half-way, an outcome that is neither success nor failure.

The vocabulary is the **intersection** over the tokenizer set, not a separate vocabulary per
model: models are compared on the same material.

| Tokenizer | Encoding | Single-token candidates |
|---|---|---:|
| tiktoken `gpt-4o` | `o200k_base` | 2 536 |
| tiktoken `gpt-4o-mini` | `o200k_base` | 2 536 |
| tiktoken `gpt-3.5-turbo` | `cl100k_base` | 2 250 |
| Hugging Face `Qwen/Qwen2.5-0.5B` | `Qwen2TokenizerFast` | 2 252 |
| intersection | | **2 220** |

`gpt-4o` and `gpt-4o-mini` share one encoding, so the intersection is taken over three distinct
tokenizers.

**The probe is the word with a leading space** (`" sword"`). In the vocabularies of these
tokenizers the unit is a word together with the space before it, and that is how a word appears
in the rule list and in the text. Without the space many ordinary nouns split in two (`sword`,
`plague`, `throne`, `beach`); probing the bare form discarded 47% of the usable nouns. The
vocabulary stores the words without the space: the space belongs to the probe.

Claude is not in the set: Anthropic publishes no local tokenizer.

## The result

2220 words of three to eight letters: 483 of three letters, 961 of four, 624 of five, 139 of
six, 11 of seven and 2 of eight. A task with `m` rules uses `2m` distinct words, so
$m \le 1110$; a point also needs $D + m \le 2220$, because the replacements never occur in the
text.

## Identity of the version and its check

The version is identified by the hash of its content, the SHA-256 of the sorted word list in a
canonical JSON form:

```text
sha256:5a91097540bec6584f7ffc94bffb4e5a6a2a9087f28828fa44d7214ccfa75906
```

The hash is recomputed whenever the vocabulary is loaded. A separate **reference** file records
the hash, the size, the source, the tokenizer set, the build parameters and the library versions.
`scripts/verify_vocab.py` checks three levels: integrity (the hash recomputed from the words
equals the recorded one); the reference (a vocabulary rebuilt with other filters has a correct
hash of its own but differs from the reference); the environment (library versions; a difference
is a warning, not an error).

`scripts/build_vocab.py` rebuilds the file from the corpora. In the recorded environment (Python
3.11, nltk 3.10.0, tiktoken 0.13.0, transformers 4.57.6) it reproduces the published file byte for
byte. A new tokenizer set makes a **new vocabulary** with a name of its own; the number in a name
counts the versions of that vocabulary. The published measurements stay on `v1_intersection`,
and the build command refuses to overwrite a saved version with a different hash.

## Vocabularies of other families of models

The same pipeline builds a vocabulary for a family of models with a common tokenizer: the same
source, the same filters and the same probe, with another tokenizer set, so each is a vocabulary of
its own, in its first version. The models of OpenAI stay on `v1_intersection`. No published measurement uses these
vocabularies yet. Measurements on different vocabularies share the source and the rules of
selection, not the words.

| Vocabulary | Tokenizers | Words |
|---|---|---:|
| `v1_nemotron` | the files of three Nemotron models (`hfjson`) | 2 135 |
| `v1_gemma4` | the files of two Gemma 4 models (`hfjson`) and the count of `gemini-3.1-flash-lite` (`gemini_count`) | 2 776 |
| `v1_gemini35fl` | the count of `gemini-3.5-flash-lite` (`gemini_count`) | 2 776 |

`v1_gemma4` and `v1_gemini35fl` hold the same words and have the same hash. The words and the
reference of a vocabulary are in `assets/vocab/`, its build config is in `config/vocab/`.

**`hfjson`** reads the file `tokenizer.json` of a Hugging Face repository directly with the library
`tokenizers`. It is the same token vocabulary and the same merge rules as through `transformers`,
which does not load the tokenizer of Gemma 4 without the package `protobuf`.

**`gemini_count`** serves a model whose tokenizer is not published. The model counts the tokens
itself through the method `countTokens` of the Gemini API, and a probe is a single token when the
count of the text `word` with the probe minus the count of `word` equals one. The method is checked
on `gemini-3.1-flash-lite`, which has the tokenizer of Gemma 4: the count of the model and the file
give the same 2 776 words.

Anyone can check a vocabulary by rebuilding it into another folder:

```bash
python scripts/build_vocab.py --config config/vocab/v1_nemotron.yaml --assets-dir my_vocab
```

The hash the command prints must equal the hash in `assets/vocab/v1_nemotron.reference.yaml`.
`v1_nemotron` needs only the tokenizer files; a repository closed by a license needs access from
your account. A vocabulary with `gemini_count` also needs `GEMINI_API_KEY`: the count is free, and
the check works while the provider serves the model.

## Limitations

- The one-syllable filter is the strictest step; allowing two syllables would give about 5 250
  words. It is kept for the uniformity of the material; the single-token test itself is exact.
- Word frequency in natural language is not controlled.
- Rare collisions remain: `are` is a WordNet noun (a unit of area) and also a verb form.
