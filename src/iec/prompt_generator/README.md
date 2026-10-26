# prompt_generator — rule list, source text and prompt

Builds the measuring load of one point from the vocabulary and `PointConditions`: the rule
list, the word stream and the assembled prompt. The same conditions and vocabulary give the
same prompt byte for byte.

## What is here

- `rule_generator.py` — `RuleGenerator(vocab)`: the rule list of one point.

  ```python
  RuleGenerator(vocab: Vocab)                         # empty vocabulary -> VocabError
  RuleGenerator.generate(params: PointConditions) -> RuleTable
  ```

  `2m` distinct words are drawn from the **sorted** vocabulary in one draw of
  `random.Random(seed_rules)`: the first half are the words to replace, the second half
  their replacements. The groups never intersect, so cascades are impossible.

  `seed_order` sets the order of the same rules: `0` keeps the drawn order, any other value
  permutes the lines with `random.Random(seed_order).sample(range(m), m)`. The pairs depend
  only on `seed_rules`, so different orders of the same pairs separate the cost of a rule
  position from the difficulty of a particular word pair.

  Errors: `ConfigError(RULES_M_INVALID)` for `m < 1`,
  `ConfigError(RULES_SEED_ORDER_INVALID)` for a negative `seed_order`,
  `ConfigError(RULES_VOCAB_TOO_SMALL)` for `m > vocab.max_pairs`.

- `text_generator.py` — `TextGenerator(vocab, tokenizer)`: the word stream of one point.

  ```python
  TextGenerator(vocab: Vocab, tokenizer: Tokenizer)
  TextGenerator.generate(params: PointConditions, rule_table: RuleTable) -> SourceText
  ```

  The lexicon is the `m` words to replace plus `D - m` background words; replacement words
  never occur in the stream. Each lexicon word is repeated exactly `n` times and the whole
  multiset is shuffled by `random.Random(seed_text)`, so frequencies are balanced by
  construction. Lines hold `WORDS_PER_LINE = 10` words (a protocol constant). The tokenizer
  measures `len_tokens` of the ready text once. Error: `ConfigError(TEXT_DISTINCT_INVALID)`
  for `D < m` or `D + m >` vocabulary size.

- `prompt_builder.py` — `PromptBuilder()`: the prompt of one point.

  ```python
  PromptBuilder.build(rule_table: RuleTable, source_text: SourceText) -> Prompt
  ```

  Three blocks in a fixed order: instruction, rules (`source -> target` line by line in
  table order), text verbatim. `PROMPT_TEMPLATE` is a code constant, not a config: the
  wording of the task is part of the measurement protocol.

## Boundaries

The layer knows no models and providers and reads no configs: every parameter comes in
`PointConditions` from the caller. Consistency of the conditions with the vocabulary is
checked here, because only the generators hold both.

Method: [Task construction](../../../docs/03-task-construction.md).
