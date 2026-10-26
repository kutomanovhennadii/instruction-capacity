# Task construction

A task is built from six numbers, the **conditions of the point**, and the vocabulary. The same
vocabulary and the same six numbers give the same prompt byte for byte; `verify` relies on this
to rebuild every published task from its seeds.

| Field | Meaning |
|---|---|
| `D` | distinct words of the text |
| `n` | occurrences of each distinct word; the text has `v = D·n` words |
| `m` | rules; `1 ≤ m ≤ D` and `D + m ≤ 2220`, the size of the vocabulary |
| `seed_rules` | which `2m` words become the rules and how they pair |
| `seed_text` | which background words fill the text and in what order the words stand |
| `seed_order` | the order of the rule lines; `0` keeps the drawn order |

## Example

```text
Rules:
lake -> stone
dog -> cloud

Text:
tree dog fish lake sun
cat lake tree dog fish
```

Four trials: two occurrences of `lake` and two of `dog`. In real tasks the lines are ten words
long.

## The rule list

`2m` **distinct** words are drawn from the vocabulary sorted alphabetically, by
`random.Random(seed_rules).sample(words, 2 * m)`; the first `m` are the words to replace, the last
`m` their replacements, paired in order. Sorting makes the draw independent of the order in which
a set yields its elements. The two groups never intersect, so no rule chains into another
(`lake -> stone`, `stone -> fish`), and every replacement has one meaning.

The order of the lines is a measured axis, the rule number `r`. With `seed_order = 0` the lines
stay in the drawn order; any other value permutes them by
`random.Random(seed_order).sample(range(m), m)`. The pairs depend only on `seed_rules`, so the
same pairs in a different order separate the cost of a position in the table from the difficulty
of a particular word pair. A real permutation seed is never below 2, so `0` is unambiguous. With
`m = 1` every permutation is the identity.

## The text

The lexicon of the text is the `m` words to replace plus `D − m` background words. With
`rng = random.Random(seed_text)` the background is drawn as `rng.sample(eligible, D - m)` from
the sorted vocabulary without the `2m` rule words, so the replacements never occur in the text:
in a right answer a replacement appears only where a rule fired. Every lexicon word is repeated
exactly `n` times and the stream is shuffled by the same `rng`. All words, those to replace and
the background, therefore have exactly equal frequencies, and no word stands out by the
statistics of the stream. The stream is cut into lines of ten words (the last line may be
shorter); there is no punctuation. The lines are part of the measurement: the validator uses
them to resynchronize the answer with the reference ([answer validation](04-answer-validation.md)).

## The prompt

The template is a constant of the protocol, version `0.1`. The rules go one per line as
`source -> target` in the order of the table; the text goes as built:

```text
Rewrite the text below by applying the substitution rules.

Rules (replace every occurrence of the left word with the right word):
{rules}

Strict requirements:
- Replace only the words listed in the rules.
- Do not change, add, or remove any other words.
- Keep the exact word order, spaces, and line breaks.
- Output only the rewritten text, nothing else.

Text:
{text}
```

The template is not a parameter: a different wording is a different version of the protocol.

## Before the call: the limits of the model

The description of a measurement names the limits of the model: the context window, the answer
reserve `max_tokens` sent with every request, and the response ceiling of the model. Before the
call the prompt and the reference answer are measured with the tokenizer of the model. The point
gets the status `na`, **without a call**, if the prompt plus the answer reserve exceeds the
context window, or if the reference answer plus 64 tokens for a wrapper exceeds the answer
reserve or the response ceiling. Otherwise the model is called.

## Statuses of a point

| Status | Meaning |
|---|---|
| `ok` | the model answered; an empty answer and silence until the time limit of the call are answers too, and the validator reads them as a collapse from the first line |
| `na` | the task does not fit the limits of the model; the model was not called |
| `api_error` | the provider did not answer; the point is not measured |

The model is called through a reliability layer: repeated attempts on transient failures, an
overall time limit of the call, a circuit breaker. A technical failure becomes the status
`api_error`, never a trial of the model.
