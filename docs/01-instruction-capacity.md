# Instruction capacity: the unit and the experiment

## The problem

Language models follow instructions, but not without limit. When a prompt holds many
independent conditions, a model starts to drop some of them: it skips a rule, mixes two rules
up, breaks the requested format, or keeps producing plausible text that no longer satisfies the
whole task. Described this way, the effect cannot be compared between models or between
prompts. IEC turns it into measured quantities: laws by which the quality depends on each kind
of load, and capacities derived from these laws. The **instruction capacity** of a model is how
many elementary independent instructions it holds and executes at once at a given level of
quality.

## The elementary instruction

The unit of load is the simplest instruction that a program can check: **replace word A with
word B** (`lake -> stone`). It is easy to state, has exactly one right answer, is checked
automatically, scales to hundreds of independent rules, needs no judgement of quality and does
not depend on the meaning of the text.

Both words come from a fixed vocabulary of short English words, each of which is a single token
for the tokenizers of the compared models ([vocabulary](02-vocabulary.md)). With multi-token
words a replacement would also measure tokenization: a word of two tokens can be replaced
half-way, and the length of the prompt would depend on which words were drawn. A single-token
rule maps one input token to one output token, the closest a prompt gets to an atomic operation.
One such rule is **one IEC unit** of instruction load in this protocol.

## The task

A task gives the model

1. a list of `m` rules `A -> B`;
2. a text: a stream of `v = D·n` words, `D` distinct words each occurring `n` times, in lines
   of ten words;
3. the instruction to rewrite the text replacing only the listed words and keeping every other
   word, the word order and the line breaks.

The text is not prose but a controlled stream of vocabulary words: the task measures whether
the model holds and applies the table, not reading or paraphrasing. The reference answer, the
text with every rule applied, is computed by a program, so the check is fully automatic
([task construction](03-task-construction.md)).

## Three quantities

The strict metric, the whole answer equal to the reference, drops to zero quickly as the text
grows and says little. An answer can go wrong in three different ways, and the instrument counts
them apart. The answer is aligned with the reference by places: every reference word and every
answer word gets one place, and each kind of failure is a Bernoulli trial of its own
([answer validation](04-answer-validation.md)).

| Quantity | Trial | Success |
|---|---|---|
| `P_sync` | a place of the alignment | the answer word stands against its reference word, not a word lost and not a word inserted |
| `H` | a rule word standing on its place | the right replacement |
| `F` | a background word standing on its place | the event is a misfire: the right-hand word of a rule stands on the place of the word |

The three quantities combine into the expected share of right places of a text,

$$P_{\mathrm{correct}} = P_{\mathrm{sync}} \cdot \big[\pi H + (1 - \pi)(1 - F)\big], \qquad \pi = \frac{m}{D}.$$

This formula is the way to combine them: it has no law and no capacity of its own.

## What the probabilities depend on

The conditions of a trial are the axes of the experiment:

| Axis | Symbol | Meaning |
|---|---|---|
| number of rules | `m` | size of the table the model holds |
| rule number | `r` | position of the trial's rule in the table, `1..m`; only a rule word has one |
| text length | `v = D·n` | words of the text and of the answer |
| word number | `s` | number of the trial's word by the reference, `1..v` |
| density | `p = m/D` | share of the distinct words of the text that the rules replace |

The seeds of the rule list, of the text and of the order of the rules vary the material without
changing the conditions, so that a measured effect is not the luck of one word pair or one text.

Two kinds of limit appear on these axes. The **static capacity**, how many rules the model holds
at once, shows along `m` and `r`. The **dynamic degradation**, how long the model keeps its place
and keeps applying the rules while it writes, shows along `s` and `v`: a model can hold thirty
rules at the start of an answer and lose them after a few hundred words. The passport separates
the two.

## Capacity at a quality level

A capacity is read from the laws at a quality level $q$, the share of successes that is still
acceptable. The passport reads every capacity at $q = 0.5$, the point where half of the trials
fail: there the share of successes over all requests does not depend on the spread of request
difficulty. It is computed at fixed reference conditions, the start of the answer to a short text with
sparse replacements, so that the capacities of different models are read at the same place. The
**instruction capacity** counts rules and the **fatigue capacity** counts words of the answer
([passport model](05-passport-model.md)). Next to them the **standard stress test** gives the size of a
task of typical proportions, the density 0.5 and one rule per 50 words, at which the last word fails in
half of the cases, and what its errors consist of ([stress test](08-stress-test.md)).

## Why this measures instruction load

The task needs no reasoning, knowledge, arithmetic or understanding of the text. When the model
errs, the cause is, with high probability, the limit of holding and applying many independent
correspondences, not a misunderstanding of the task. This is not a measure of general ability;
it is a calibrated scale of one component of it, in atomic units.

The same rules can serve as a measuring load next to another task: if a model holds 60 rules at
95% alone and 40 rules while it also solves a problem, the problem weighs about 20 IEC units in
this protocol. This version of the instrument measures the rules alone.
