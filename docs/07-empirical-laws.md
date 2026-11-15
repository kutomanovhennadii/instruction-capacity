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
| `P_sync` | 22,598,814 | 11,269,394 | 14.23 ± 2.32 |
| `H` | 1,155,179 | 119,727 | 3.03 ± 0.40 |
| `F` | 10,213,582 | 77,048 | 6.18 ± 1.05 |

The loss of the place depends on the particular request much more than the execution of the rules and
the misfires do: its variance of the shift is the largest.

The table gives the mean over the requests, because the variance of the shift is not one number: it depends on the load of a
request, and the fit, a GLMM with heterogeneous random-intercept variance, finds this dependence together with the
coefficients. Against one variance for all requests, `2·Δ ln L` is
592.3 for `P_sync`, 719.1 for `H` and 1324.9
for `F`, with 24, 24 and 22 more
parameters. The variance of `P_sync` falls with the density, from 23–25 at rare replacements to
6.74 at full density. The variances of `H` and `F` fall with the number of rules, from 8.01
to 0.13 and from 7.02 to 0.13. The number of rules is not an axis of the law of
`P_sync`, and yet it acts on its spread. The passport prints these variances by the groups of the three axes; the capacities,
taken at the level 0.5, do not depend on them.

## Laws of synchronization: `P_sync`

### Length and word number: a corner between a plateau and fatigue

$$VS(v, s) = \tfrac{1}{2}\Big[(\lambda_S + \kappa_S\ell) + \mu_v - \sqrt{(\lambda_S + \kappa_S\ell - \mu_v)^2 + \kappa_S^2\omega_S^2}\Big], \qquad \ell = \log_2 s.$$

At the start of the answer the logit stays on a plateau of its length slice; past a corner it falls along
a straight line in $\log_2 s$, with a slope shared by all slices. The plateau depends on the text length by
the same kind of curve: it stays at one level on short texts, and longer texts lower it along a straight
line in $\log_2 v$.

$$\mu(v) = \mu_0 + \tfrac{\kappa_V}{2}\Big[\log_2\tfrac{v}{v_{*}} + \sqrt{\log_2^2\tfrac{v}{v_{*}} + \omega_V^2}\Big].$$

| Parameter | Value | Meaning |
|---|---|---|
| `kappa_S` | −3.59 ± 0.21 | slope of the fatigue, logit per doubling of the word number |
| `lambda_S` | 20.23 ± 2.46 | the asymptote at the first word, after the correction |
| `omega_S` | 2.80 ± 0.49 | width of the corner, doublings of the word number |
| `mu_0` | 20.01 ± 1.94 | the plateau on short texts, the shift of the grid |
| `kappa_V` | −2.59 ± 0.69 | slope of the plateau, logit per doubling of a long text |
| `v_star` | 565 words | corner of the plateau, `log₂(v*/20) = 4.82 ± 0.97` |

The slices leave `χ² = 186.0` over 56 cells: the curves
go along the cells, but the law does not pass formally. The plateau hyperbola leaves
`χ² = 10.1` over 8 slices.

### Density

$$P(p) = \mu_P + \tfrac{\kappa_P}{2}\Big[\log_2\tfrac{p}{p_{*}} + \sqrt{\log_2^2\tfrac{p}{p_{*}} + \omega_P^2}\Big].$$

The profile stays nearly level at rare replacements and falls steeply towards full density.
`mu_P` = 16.44 ± 1.28, `kappa_P` = −11.59 ± 1.13,
`p_star` = 1.00 (held on the bound of its range), `omega_P` = 4.22 ± 0.46;
`χ² = 22.2` over 10 bins. The loss against rare
replacements at full density is 24.44 logits: full density multiplies the odds of losing the place by `e^24.44`, about
4·10¹⁰. For the execution of the rules it is 1.72 logits: the density acts first of all on the synchronization.

The dependence agrees with the copying mechanism proposed in the article: the replacement words do not occur in
the source text, so each of them interrupts the match of the end of the written answer with a fragment of the
input, and at full density no answer word occurs in the input. The instrument measures the form of the
dependence, not its cause, and this reading needs a separate check.

The number of rules is an axis of the model of `P_sync`: its law is a line in `log₂ m`, and its numbers are in the passport.

## Laws of rule execution: `H`

### Length and word number

$$VS(v, s) = \mu_v - A \cdot 2^{\gamma\ell}, \qquad \mu_v = \mu_0 + \kappa_V \log_2\tfrac{v}{20}.$$

Once the loss of the place is a quantity of its own, a slice of the grid has no corner between a plateau and
fatigue: it falls as an exponent in $\log_2 s$ and dies out to its level at $s \to 0$, with the shared
`A` = 1.065 ± 0.399 and `gamma` = 0.224 ± 0.036, and the levels of the slices lie on a straight line,
`mu_0` = 16.14 ± 1.03, `kappa_V` = −0.711 ± 0.094. By the own variances, the law leaves
`χ² = 151.0` over 56 cells and
`χ² = 1.4` over 8 slices.

### Density

A two-slope hyperbola, a hyperbola on two slanted asymptotes:

$$P(p) = \mu_P + \beta (x - x_*) + \tfrac{\kappa_P}{2}\Big[(x - x_*) + \sqrt{(x - x_*)^2 + \omega_P^2}\Big], \qquad x = \log_2\tfrac{p}{2^{-10}}.$$

The profile is a hill: at the rarest replacements the rules are executed worse than at a moderate density. The left
asymptote rises by `beta` = 0.47 ± 0.27 logit per doubling of the density, the right one falls by `beta + kappa_P` = −1.48
logit per doubling, `kappa_P` = −1.95 ± 0.79; the asymptotes cross at `p* = 0.12`, the width of the corner is
`omega_P` = 2.53 ± 1.67, and the top stands near `p = 0.04`, 1.57 logits above the rarest density. `χ² = 5.5` over 10 bins,
and no bin is an outlier. The law has no horizontal asymptote, so its shift `mu_P` = −0.22 ± 0.53 is the law at `p = 2⁻¹⁰`.

### Number of rules

$$M(m) = \lambda_M + \kappa_M \log_2 m.$$

`lambda_M` = −1.01 ± 0.47, `kappa_M` = −0.378 ± 0.040 per doubling;
`χ² = 36.9` over 8 bins.

### Rule number

A slanted hyperbola in $\log_2 r$: the rules at the head of the list cost nearly nothing, and past a corner every
doubling of the rule number costs `kappa_R` = −0.51 ± 0.04 logit. The corner lies near
`r* = 10`; `χ² = 7.6` over 8 bins.

## Laws of misfires: `F`

The laws are read on the logit of no misfire: lower means more misfires. They go through every bin with a coefficient
by generalised least squares, as the laws of the other quantities; no bin of `F` is an outlier.

| Axis | Law | Parameters |
|---|---|---|
| text length | a line in $\log_2 v$ | `kappa_V` = −0.455 ± 0.056 per doubling |
| share of the answer `q = s/v` | $c_q + A_q e^{-q/\tau}$ | `A_q` = 1.119 ± 0.223, `tau` = 0.674 ± 0.286 |
| density | $a + b\log_2(p/2^{-10}) + cp$ | `b` = −0.029 ± 0.039, `c` = −2.64 ± 0.45; the law has no hill, its maximum stands at the rarest density `p = 2⁻¹⁰` |
| number of rules | a line in $\log_2 m$ | `kappa_M` = −0.391 ± 0.026 per doubling |

Part of the growth of the misfires with the number of rules is a matter of spelling: the more rules a task has, the
closer each background word stands to the left-hand word of some rule, and a word spelled close to a left-hand word
misfires much more often. The passport law keeps this growth on purpose, because it is part of what adding a rule does
to a text: the law describes the real mix of the words of a task, not a word far from every rule. The price of this
choice is that the capacities of `F` are directly comparable only between models measured on the same vocabulary.

## The passport

The capacities are taken at the level 0.5. There the threshold of the logit of the median request is zero whatever the spread of
request difficulty, so the reserve of a quantity is its whole level: 36.41 ± 2.23 logits for `P_sync`,
15.00 ± 1.08 for `H` and 11.22 ± 0.54 for `F`. A capacity is given with its 95% interval.

| Quantity | Capacity | q | By the law | By the bins |
|---|---|---|---|---|
| `H` | instruction, rules | 0.5 | 32.3 (29.1–36.0) | 30.7 |
| `H` | fatigue, words | 0.5 | 10,800* (8,912–13,089) | |
| `H` | measure of the density, times | — | 5.6 (3.0–10.6) | |
| `P_sync` | fatigue, words | 0.5 | 44,473* (34,076–58,042) | |
| `P_sync` | measure of the density, times | — | 4.1·10¹⁰ (1.0·10¹⁰–1.7·10¹¹) | |
| `F` | instruction, rules | 0.5 | 4.4·10⁸* (8.5·10⁷–2.3·10⁹) | at least 256 |
| `F` | fatigue, words | 0.5 | 1.4·10⁸* (4.0·10⁷–5.2·10⁸) | |
| `F` | measure of the density, times | — | 18.6 (12.9–26.8) | |

The measure of the density is how many times the full density multiplies the odds of an error against the rarest
replacements, `e^(−P̂(1))`; a factor below one would mean that the full density lowers them. For `F` it is a conditional
quantity: a text has no background words at `p = 1`, so the number continues
the law of the density and has no real meaning of its own.

## How to read these laws

One model is measured here, so the universality of the laws is not established. The laws describe the median request, and
they hold inside the measured area; a capacity marked `*` shows that the limit of the answer comes before the reserve is
spent. The bins are published next to every law: the gap between a law and its bins, and the status of a bin outside its law,
are part of the result.
