# The standard stress test

The standard stress test describes how a model behaves as the complexity of a task with fixed proportions of the load
increases step by step. The complexity increases until the probability of an error on the last word of the text reaches
50%.

The density of replacements in the test is $p = 0.5$, and the text has one rule per 50 words. With $m$ rules the text
has $T = 50m$ words and $D = 2m$ distinct words, each of them repeated 25 times. The length of the text and the number
of rules increase together, keeping these proportions.

The point of 50% is set by the combined error of the last word of the median request:

$$E = 1 - P_{\mathrm{sync}} \cdot H = 0.5,$$

where $P_{\mathrm{sync}}$ is the probability that the word keeps its place, and $H$ is the conditional probability that
a rule word on its place is replaced correctly. The median request is the request with a zero random shift of
difficulty. False replacements of background words do not enter the definition of the point, but they are measured at
the point found.

## In the data and in the code

Since release 0.2.4 the results of the test are written to the section `stress_test` of `passport_data.json`
([data format](06-data-format.md)). The package `iec.passport.stress_test` computes them from the three published
sections of the passport ([passport model](05-passport-model.md)), so the test can be recomputed from the passport file
alone, without the points of the measurement.

The test gives seven numbers.

1. The point of 50%: the length of the text $T$ and the number of rules $m$.
2. The combined error at the point: 50% by definition.
3. The structure of the error at the point: the errors of every kind per 100 combined errors — a lost place,
   $1 - P_{\mathrm{sync}}$, and a wrong replacement, $P_{\mathrm{sync}}(1 - H)$. Over these 100 errors the false
   replacements of background words are given, $P_{\mathrm{sync}} F$, where $F$ is the probability of a false
   replacement of a background word that kept its place.
4. Synchronization: the mean share of synchronization errors over the text of the median request, and the influence
   of length, density and rules per 100 synchronization errors at the point.
5. Replacements: the same numbers for the errors of executing the rules.
6. Background: the same numbers for the false replacements of background words.
7. The combined error over the text: the mean share of combined errors over all the words of the median request.

The mean share of errors over the text is taken over all its words for the request with a zero shift of difficulty.

The influence of a load is set by its price in the logit of the probability of the correct outcome of the last word.
The total change is shared among the loads by the Aumann–Shapley value of one word, and the result is written per 100
errors of the kind.

The price of a load is the fall of the logit from the zero point that its law gives. For the length of the text the
zero point is 20 words and the start of the answer; for the density, the value of its law at $p \to 0$, or at
$p = 2^{-10}$ when the law has no horizontal asymptote (for the background errors, the top of the law of the density);
for the rules, one rule in the first place of the list. A word belongs to any rule with the same chance, so the price of
the rules is the mean over all the places of the list.

The influence keeps its sign: a negative value means that the load lowers the probability of an error of that kind. The
three shares of the influence sum to 100. When the prices of the loads sum to zero or less, that is, the probability of
an error at the point is not above its value at the zero point, the influence is not defined.

Every number has a 95% interval by the delta method. Every section of the passport publishes the covariance of the
parameters of its laws in the field `law_covariance`. The laws of synchronization, replacements and background errors
are fitted separately, and the passport has no covariance between the sections, so the parameters of the three
quantities are taken independent.

The point of 50% has a status by its place in the measured area:

- `measured` — the point lies inside the measured area, set by the longest text and the largest number of rules of the
  bins of $H$;
- `extrapolated` — the point lies past this area and continues the laws.

Without a point the status gives the reason:

- `below_one_rule` — the error reaches 50% already at one rule;
- `beyond_range` — the error does not reach 50% up to $2^{20}$ rules;
- `no_laws` — one of the three quantities has no laws.

Example: GPT-4o mini, passport 0.2.4 (`artefacts/pilot_gpt4o_mini_v55`). The point of 50% lies at $T = 910 \pm 41$
words and $m = 18.2 \pm 0.8$ rules. Per 100 combined errors there are $95.7 \pm 0.8$ lost places and $4.3 \pm 0.8$
wrong replacements, and over the same 100 errors $0.8 \pm 0.1$ false replacements of background words.

- Synchronization: the mean share of errors over the text is $11.0 \pm 0.3$%; the influence of length, density and
  rules is $63.2 \pm 3.0$, $36.1 \pm 2.8$ and $0.7 \pm 0.8$ per 100 errors.
- Replacements: $1.7 \pm 0.3$%; the influence of the three loads in the same order is $83.6 \pm 3.3$, $3.7 \pm 3.5$ and
  $12.7 \pm 2.0$.
- Background: $0.5 \pm 0.1$%; the influence is $56.6 \pm 2.5$, $17.1 \pm 1.7$ and $26.2 \pm 2.6$.

The combined error over the text is $12.4 \pm 0.4$%.

## Reading the test

The usual case is a point of 50% inside the measured area. The structure of the error there shows which mechanism of
degradation chiefly limits the task: for GPT-4o mini about 96 of 100 combined errors come from a lost place and only
about four from wrong replacements.

A negative influence of a load is not by itself a fault of the calculation: for some models the quality grows with a
load. For example, for Gemini 3.5 Flash Lite the synchronization improves with the length of the text.

The choice of the combined error as the criterion matters. A point of 50% set by the errors of replacement alone could
be misleading: the execution of the rules is counted only on the words that kept their place. Under a heavy load the
share of such words can become negligible, and the point would no longer describe how the model performs the task. The
combined error takes both mechanisms of degradation into account.

## What it is not

Not the measure of the density. In release 0.2.4 the standard stress test replaced the measure of the density, which
described one point of the curve of quality over the density of replacements. The new test studies how the complexity
of the whole task increases: the length of the text and the number of rules increase together, keeping the given
proportions.

Not a capacity. The point of the test is not a capacity of the model. A capacity is the limiting load along one axis
with the other loads at the zero point of the passport: a text of 20 words, the start of the answer, the density at the
shift of its law, one rule. The standard test, on the contrary, increases the length of the text and the number of
rules together.

Not the formula of correctness. The combined error describes the execution of the rules on the rule words together with
the loss of the place. The formula of correctness gives the share of correct words of the whole answer, background
words included. The two have different domains and do not replace each other.

Not the share over all requests. The test is read for the median request, not by the share of errors over all the
requests. For one quantity at the level of 50% the two readings coincide at any spread of difficulty; the combined error
is a product of two quantities, and for it the coincidence is approximate.

## Where to read more

- [passport model](05-passport-model.md) — the laws, the zero point and the capacities;
- [data format](06-data-format.md) — the fields of `stress_test` and `law_covariance`;
- [the package of the test](../src/iec/passport/stress_test/README.md) — the calculation and the statuses.
