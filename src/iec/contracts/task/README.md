# task — the contracts of the task of a point

The vocabulary, the rule list, the source text, the prompt and the reference answer: everything the generators build for one
point before the model is called.

## What is here

| Module | Role |
|---|---|
| `vocab.py` | `Vocab`: `vocab_id`, `words`, `hash`, the tokenizer names; `compute_hash(words)` is the canonical hash |
| `rule_table.py` | `Rule` and `RuleTable`: pairs `source -> target` in the order shown to the model; `m`, `sources`, `targets` |
| `source_text.py` | `SourceText`: the word stream and the positions of the words to replace; `assemble_text`, the one rule of the text form |
| `prompt.py` | `Prompt`: the exact request text and an optional payload position |
| `reference.py` | `Reference`: the reference answer `Y*`, assembled by the same rule as the source text |

## Boundaries

The construction invariants — distinct words, disjoint groups, words of the vocabulary — are ensured by the generators;
these contracts only carry the result.
