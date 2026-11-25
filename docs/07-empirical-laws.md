# Empirical laws

The [passport model](05-passport-model.md) says how the instrument computes. This document says
what the measurement showed: the laws drawn through the bins, read as empirical regularities. Every
number comes from the published passport in `artefacts/`; intervals are half-widths of 95%, and a
capacity marked `*` lies beyond the measured area and continues its law.

## Established on gpt-4o-mini

`pilot_gpt4o_mini_v55`: `gpt-4o-mini-2024-07-18`, 5,183 answered points; 11 of them are answers without a word, and the
passport skips them. The article describes one outcome, a right replacement on a rule word, which joins the loss of the
place with a wrong replacement. Here the outcomes are apart.

| Quantity | Trials | Failures | Variance of the request shift `σ²` |
|---|---|---|---|
| `P_sync` | 15,078,260 | 3,752,796 | 10.47 ± 0.64 |
| `H` | 1,152,598 | 119,630 | 3.04 ± 0.39 |
| `F` | 10,212,207 | 76,562 | 6.08 ± 1.00 |

The loss of the place depends on the particular request much more than the execution of the rules and
the misfires do: its variance of the shift is the largest.

For `H` and `F` the table gives the mean over the requests, because their variance of the shift is not one number: it depends
on the load of a request, and the fit, a GLMM with heterogeneous random-intercept variance, finds this dependence together
with the coefficients. Against one variance for all requests, `2·Δ ln L` is 719.3 for `H` and 1315.6 for `F`, with 24 and 22
more parameters. The variances of `H` and `F` fall with the number of rules, from 7.99 to 0.13 and from 6.76 to 0.13. For
`P_sync` the variance by the load turned four bins of the number of rules against the data: the passport put them below the
bin of one rule, while the median request of the data lies above it. So `P_sync` keeps one variance
([passport model](05-passport-model.md), section 5). The passport prints the variances by the groups of the three axes; the
capacities, taken at the level 0.5, do not depend on them.

## Laws of synchronization: `P_sync`

### Length and word number: a corner between a plateau and fatigue

$$VS(v, s) = \tfrac{1}{2}\Big[(\lambda_S + \kappa_S\ell) + \mu_v - \sqrt{(\lambda_S + \kappa_S\ell - \mu_v)^2 + \kappa_S^2\omega_S^2}\Big], \qquad \ell = \log_2 s.$$

At the start of the answer the logit stays on a plateau of its length slice; past a corner it falls along
a straight line in $\log_2 s$, with a slope shared by all slices. Of the three forms of a slice the data
chose this hyperbola on two asymptotes. The plateau falls with the text length along a straight line in
$\log_2 v$:

$$\mu(v) = \mu_0 + \kappa_V \log_2\tfrac{v}{20}.$$

| Parameter | Value | Meaning |
|---|---|---|
| `kappa_S` | −3.64 ± 0.36 | slope of the fatigue, logit per doubling of the word number |
| `lambda_S` | 45.28 ± 5.93 | the slanted asymptote at the first word, in the coding of the fit |
| `omega_S` | 4.39 ± 0.83 | width of the corner, doublings of the word number |
| `mu_0` | 26.20 ± 2.02 | the plateau of a text of 20 words, the shift of the grid |
| `kappa_V` | −1.81 ± 0.24 | slope of the plateau, logit per doubling of the text |

With the weights $1/(\sigma_i^2 + \tau^2)$ the slices leave `χ² = 40.3` over 56 cells, and the line of the
plateau leaves `χ² = 5.9` over 8 slices.

### Density

$$P(p) = \mu_P - A \cdot 2^{\gamma x}, \qquad x = \log_2\tfrac{p}{2^{-10}}.$$

The profile stays nearly level at rare replacements and falls ever faster towards full density: the data
chose the binary exponential over the slanted hyperbola. `mu_P` = 9.56 ± 0.92, `A` = 0.62 ± 0.27,
`gamma` = 0.48 ± 0.06; `χ² = 7.7` over 10 bins. The loss against rare replacements at full density is
16.81 logits; for the execution of the rules it is 1.73 logits: the density acts first of all on the
synchronization.

The dependence agrees with the copying mechanism proposed in the article: the replacement words do not occur in
the source text, so each of them interrupts the match of the end of the written answer with a fragment of the
input, and at full density no answer word occurs in the input. The instrument measures the form of the
dependence, not its cause, and this reading needs a separate check.

The number of rules is an axis of the model of `P_sync`: its law is a line in `log₂ m`, and its numbers are in the passport.

## Laws of rule execution: `H`

### Length and word number

$$VS(v, s) = \tfrac{1}{2}\Big[(\lambda_S + \kappa_S\ell) + \mu_v - \sqrt{(\lambda_S + \kappa_S\ell - \mu_v)^2 + \kappa_S^2\omega_S^2}\Big], \qquad \mu_v = \mu_0 + \kappa_V \log_2\tfrac{v}{20}.$$

A slice of the grid has the same corner as a slice of `P_sync`: it keeps its level at the start of the
answer and past the corner falls along a straight line in $\log_2 s$. The data chose this hyperbola on two
asymptotes over the binary exponential, which dies out to its level without a corner. The shared
`kappa_S` = −1.29 ± 0.10 logit per doubling of the word number, `lambda_S` = 20.37 ± 1.37 in the coding of
the fit and `omega_S` = 3.91 ± 0.68, and the levels of the slices lie on a straight line,
`mu_0` = 20.06 ± 1.77, `kappa_V` = −1.42 ± 0.23. The law leaves `χ² = 43.5` over 56 cells and `χ² = 5.9`
over 8 slices.

### Density

A two-slope hyperbola, a hyperbola on two slanted asymptotes:

$$P(p) = \mu_P + \beta (x - x_*) + \tfrac{\kappa_P}{2}\Big[(x - x_*) + \sqrt{(x - x_*)^2 + \omega_P^2}\Big], \qquad x = \log_2\tfrac{p}{2^{-10}}.$$

The profile is a hill: at the rarest replacements the rules are executed worse than at a moderate density. The left
asymptote rises by `beta` = 0.42 ± 0.25 logit per doubling of the density, the right one falls by `beta + kappa_P` = −1.33
logit per doubling, `kappa_P` = −1.75 ± 0.78; the asymptotes cross at `p* = 0.11`, the width of the corner is
`omega_P` = 2.11 ± 1.74, and the top stands near `p = 0.04`, 1.54 logits above the rarest density. `χ² = 4.7` over 10 bins,
and no bin is an outlier. The law has no horizontal asymptote, so its shift `mu_P` = −0.20 ± 0.52 is the law at `p = 2⁻¹⁰`.

### Number of rules

$$M(m) = \lambda_M + \kappa_M \log_2 m.$$

`lambda_M` = −0.75 ± 0.48, `kappa_M` = −0.438 ± 0.045 per doubling;
`χ² = 6.5` over 8 bins.

### Rule number

A slanted hyperbola in $\log_2 r$: the rules at the head of the list cost nearly nothing, and past a corner every
doubling of the rule number costs `kappa_R` = −0.51 ± 0.04 logit. The corner lies near
`r* = 10`; `χ² = 4.5` over 8 bins.

## Laws of misfires: `F`

The laws are read on the logit of no misfire: lower means more misfires. They go through the well determined bins with
the weights $1/(\sigma_i^2 + \tau^2)$, as the laws of the other quantities; no bin of `F` is an outlier.

| Axis | Law | Parameters |
|---|---|---|
| text length | a line in $\log_2 v$ | `kappa_V` = −0.705 ± 0.088 per doubling |
| share of the answer `q = s/v` | $c_q + A_q e^{-q/\tau}$ | `A_q` = 1.127 ± 0.188, `tau` = 0.618 ± 0.228 |
| density | $a + b\log_2(p/2^{-10}) + cp$ | `b` = 0.023 ± 0.044, `c` = −3.22 ± 0.49; the law rises a little at rare replacements, `b` within its error of zero, and its maximum stands at `p = 0.010` |
| number of rules | a line in $\log_2 m$ | `kappa_M` = −0.519 ± 0.031 per doubling |

Part of the growth of the misfires with the number of rules is a matter of spelling: the more rules a task has, the
closer each background word stands to the left-hand word of some rule, and a word spelled close to a left-hand word
misfires much more often. The passport law keeps this growth on purpose, because it is part of what adding a rule does
to a text: the law describes the real mix of the words of a task, not a word far from every rule. The price of this
choice is that the capacities of `F` are directly comparable only between models measured on the same vocabulary.

## The passport

The capacities are taken at the level 0.5. There the threshold of the logit of the median request is zero whatever the spread of
request difficulty, so the reserve of a quantity is its whole level: 35.78 ± 2.03 logits for `P_sync`,
19.19 ± 1.79 for `H` and 13.32 ± 0.71 for `F`. A capacity is given with its 95% interval.

| Quantity | Capacity | q | By the law | By the bins |
|---|---|---|---|---|
| `H` | instruction, rules | 0.5 | 36.6 (33.0–40.7) | 35.5 |
| `H` | fatigue, words | 0.5 | 18,537* (14,047–24,462) | |
| `P_sync` | fatigue, words | 0.5 | 17,376* (13,949–21,646) | |
| `F` | instruction, rules | 0.5 | 5.4·10⁷* (1.3·10⁷–2.3·10⁸) | at least 256 |
| `F` | fatigue, words | 0.5 | 4.0·10⁶* (1.7·10⁶–9.6·10⁶) | |

The standard stress test ([stress test](08-stress-test.md)): at the density 0.5 and one rule per 50 words the last word
of the text fails in half of the cases for the median request at T = 1187 ± 53 words and 23.7 ± 1.1 rules. Per 100
combined errors of the last word there, 90.4 ± 1.8 are a lost place and 9.6 ± 1.8 a wrong replacement, and over them
1.3 ± 0.2 are misfires on background words. The errors of the synchronization over the text are 12.7 ± 0.5%, and per
100 of them at the point the length gives 65.3 ± 2.7, the density 34.0 ± 2.6 and the rules 0.8 ± 0.8; the replacements
3.8 ± 0.6%, with 83.0 ± 3.1, 3.5 ± 3.3 and 13.5 ± 2.0; the background 0.9 ± 0.1%, with 57.0 ± 2.5, 16.3 ± 1.7 and
26.7 ± 2.6. The combined error over the text is 15.7 ± 0.5%. At this point GPT-4o mini fails first of all by losing the
place in the text, and the place is lost more to the length of the text than to the density.

## How to read these laws

One model is measured here, so the universality of the laws is not established. The laws describe the median request, and
they hold inside the measured area; a capacity marked `*` shows that the limit of the answer comes before the reserve is
spent. The bins are published next to every law: the gap between a law and its bins, and the status of a bin outside its law,
are part of the result.
