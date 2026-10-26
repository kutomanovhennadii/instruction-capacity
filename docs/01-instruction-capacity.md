# Instruction capacity: the unit and the experiment

## The problem

Language models follow instructions, but not without limit. When a prompt holds many
independent conditions, a model starts to drop some of them: it skips a rule, mixes two rules
up, breaks the requested format, or keeps producing plausible text that no longer satisfies the
whole task. Described this way, the effect cannot be compared between models or between
prompts. IEC turns it into a measured quantity: the **instruction capacity** of a model is how
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

## Trials and the measured probability

Every occurrence of a word to replace is one **trial** with the outcome $z = 1$ if the answer
has the right replacement at that position and $z = 0$ otherwise: the word left as is, replaced
by a wrong word, or lost together with the structure of the text
([answer validation](04-answer-validation.md)). A task with `m` rules over a text in which each
word occurs `n` times gives $m \cdot n$ trials.

The strict metric, the whole answer equal to the reference, drops to zero quickly as the text
grows and says little. The measured quantity is the probability of a right replacement in one
trial as a function of the conditions of the trial.

## What the probability depends on

The conditions of a trial are the axes of the experiment:

| Axis | Symbol | Meaning |
|---|---|---|
| number of rules | `m` | size of the table the model holds |
| rule number | `r` | position of the trial's rule in the table, `1..m` |
| text length | `v = D·n` | words of the text and of the answer |
| word position | `s` | position of the trial's word in the answer, `1..v` |
| density | `p = m/D` | share of the distinct words of the text that the rules replace |

The seeds of the rule list, of the text and of the order of the rules vary the material without
changing the conditions, so that a measured effect is not the luck of one word pair or one text.

Two kinds of limit appear on these axes. The **static capacity**, how many rules the model holds
at once, shows along `m` and `r`. The **dynamic degradation**, how long the model keeps applying
the rules while it writes, shows along `s` and `v`: a model can hold thirty rules at the start of
an answer and lose them after a few hundred words. The passport separates the two.

## Capacity at a quality level

The capacity is read at a quality level $q$, the probability of a right replacement that is still
acceptable. $\mathrm{IEC}_q$ is the number of rules the model holds with at least that
probability: at $q = 0.95$ the practical capacity, at $q = 0.5$ the point where half of the
replacements fail. The passport computes it from a fitted statistical model at fixed reference
conditions, the start of the answer to a short text with sparse replacements, so that the
capacities of different models are read at the same place
([passport model](05-passport-model.md)).

## Why this measures instruction load

The task needs no reasoning, knowledge, arithmetic or understanding of the text. When the model
errs, the cause is, with high probability, the limit of holding and applying many independent
correspondences, not a misunderstanding of the task. This is not a measure of general ability;
it is a calibrated scale of one component of it, in atomic units.

The same rules can serve as a measuring load next to another task: if a model holds 60 rules at
95% alone and 40 rules while it also solves a problem, the problem weighs about 20 IEC units in
this protocol. This version of the instrument measures the rules alone.
