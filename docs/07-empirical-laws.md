# Empirical laws

The [passport model](05-passport-model.md) says how the instrument computes. This document says
what the measurements showed: the laws that the second stage draws through the bins, read as
empirical regularities. Every number comes from the published passports in `artefacts/`;
intervals are ±1.96 standard errors.

## Established on gpt-4o-mini

`pilot_gpt4o_mini_v34`: `gpt-4o-mini-2024-07-18`, 2958 points, 642430 Bernoulli trials, of which
2722 points and 517762 trials fall into the area of analysis. It is the only published
measurement.

### Length and position: a corner between a plateau and fatigue

$$VS(v, s) = \tfrac{1}{2}\Big[(\lambda_S + \kappa_S\ell) + \mu(v) - \sqrt{(\lambda_S + \kappa_S\ell - \mu(v))^2 + \kappa_S^2\omega_S^2}\Big], \qquad \ell = \log_2 s, \quad \mu(v) = \mu_0 + \tfrac{\kappa_V}{2}\Big[\log_2\tfrac{v}{v_{*}} + \sqrt{\log_2^2\tfrac{v}{v_{*}} + \omega_V^2}\Big].$$

At the early positions of the answer the logit stays on a plateau $\mu(v)$; past a corner it falls
along a straight line in $\log_2 s$: the model tires as it writes. The plateau depends on the text
length by a hyperbola: up to about 270 words it stays at one level, and longer texts lower it along
a straight line in $\log_2 v$. The law is fitted in two steps: every length slice gets its own
plateau $\mu_v$ with $\kappa_S$, $\lambda_S$ and $\omega_S$ shared, and the hyperbola $\mu(v)$ is
drawn through the plateaus of the eight determined slices.

| Parameter | Value | Meaning |
|---|---|---|
| `kappa_S` | -2.67 ± 0.310 | slope of the fatigue, logit per doubling of the word position |
| `kappa_V` | -1.60 ± 0.434 | slope of the slanted asymptote of the plateau, logit per doubling of a long text |
| `v_star` | 274.18 ± 173.65 | corner of the plateau, the text length in words where its asymptotes cross |
| `omega_V` | 0.00100 (held on its bound) | width of the corner of the plateau, doublings of the text length |
| `lambda_S` | 17.39 ± 3.37 | level of the asymptote at the first word, relative to `delta_VS` |
| `omega_S` | 2.23 ± 0.528 | width of the corner, doublings of the position |

The values are those of the median point: the passport model carries the shift of a point
inside the fit, and $\hat\sigma^2$ = 5.90 ± 0.360 (see the [passport model](05-passport-model.md)).
Every doubling of the position past the corner costs 2.67 ± 0.310 logit, and every doubling of a
long text lowers the plateau by 1.60 ± 0.434 logit. Weighted by the own variances of the cells,
the slices leave χ² = 56.41 over 56 cells with 11 parameters, and the plateau hyperbola leaves
χ² = 1.28 over 8 plateaus.

![Length × position grid of gpt-4o-mini by length slice, after the shift](../artefacts/pilot_gpt4o_mini_v34/figures/vspmr_glmm_v1/vs-slices-after-shift.png)

![Plateau by length slice of gpt-4o-mini and the plateau hyperbola, after the shift](../artefacts/pilot_gpt4o_mini_v34/figures/vspmr_glmm_v1/plateau-after-shift.png)

### Density of replacements

$$P(p) = \mu_P - A_P\big(1 - e^{-p/p_c}\big), \qquad p = m/D.$$

The logit falls as a larger share of the text has to be replaced, and saturates: $A_P$ =
12.07 ± 2.63, $p_c$ = 0.670 ± 0.269, χ² = 14.68 over 9 bins. Replacing every word of the text
($p = 1$) costs 9.35 ± 1.01 logit against sparse replacements.

### The size of the table

$$M(m) = \lambda_M + \kappa_M \log_2 m.$$

Every rule added to the table costs by itself, apart from its own position: $\kappa_M$ =
-0.427 ± 0.100, so every doubling of the table lowers the level by 0.427 logit. The
restored rule-count bins fall from -0.440 ± 0.578 at one rule to -3.13 ± 0.730 at 65–128 rules.

![Density bins of gpt-4o-mini and their law, after the shift](../artefacts/pilot_gpt4o_mini_v34/figures/vspmr_glmm_v1/P-after-shift.png)

![Rule-count bins of gpt-4o-mini and their law, after the shift](../artefacts/pilot_gpt4o_mini_v34/figures/vspmr_glmm_v1/M-after-shift.png)

### The rule number

$$R(r) = \lambda_R - A_R\,(r^{\gamma} - 1).$$

A later rule costs more than an earlier one, and the cost grows faster than the rule number:
$A_R$ = 0.0412 ± 0.0398, $\gamma$ = 0.736 ± 0.202, and every doubling of the rule number multiplies
the degradation by $2^\gamma$ = 1.67 ± 0.234. The restored bins: -0.240 ± 0.103 for the rules
9–16, -0.412 ± 0.119 for the rules 17–32, -0.664 ± 0.127 for the rules 33–64 and -1.05 ± 0.123 for
the rules 65–128.

![Rule-number bins of gpt-4o-mini and their law, after the shift](../artefacts/pilot_gpt4o_mini_v34/figures/vspmr_glmm_v1/R-after-shift.png)

### The collapse of the answer

$$\ln\frac{\psi}{1 - \psi} = \lambda_\psi + \kappa_\psi \log_2\frac{v}{20} + A_\psi\, p^{\gamma_\psi}.$$

995 of the 2958 answered points lost the text: their answer ended in a failure of the third
kind - lines dropped, merged or added in the middle, or an answer cut before the end of the text.
The probability $\psi$ that a point loses the text is measured by the method of the passport model
over all answered points, the boundary cells too: the logit of the collapse is split into the bins
of the text length and of the density, a law is drawn through the bins of each axis, and the
shifts - the length law at $v$ = 20 words, the density law at $p \to 0$ - go into the level
$\lambda_\psi$. The rate law: $\lambda_\psi$ = -20.00 ± 1.58, $\kappa_\psi$ = 2.40 ± 0.192,
$A_\psi$ = 12.25 ± 1.01, $\gamma_\psi$ = 0.706 ± 0.0931. Every doubling of the text multiplies the
odds of the collapse by about 11, and the density term grows steeply only above $p$ = 1/16. The
number of rules is not an axis of the collapse: fitted together with the length and density
bins, its bins stay within ±0.5 logit of zero with no trend in the number of rules (one bin of
eight, 5–8 rules, has an interval just off zero, as chance gives among eight).

![Collapse of gpt-4o-mini by text length, after the shift](../artefacts/pilot_gpt4o_mini_v34/figures/collapse-V-after-shift.png)

![Collapse of gpt-4o-mini by replacement density, after the shift](../artefacts/pilot_gpt4o_mini_v34/figures/collapse-P-after-shift.png)

### The place of the collapse

$$\ln\frac{F}{1 - F} = \lambda_F + \tfrac{\kappa_{SF}}{2}\Big[\log_2\tfrac{s}{s_*} + \sqrt{\log_2^2\tfrac{s}{s_*} + \omega_{SF}^2}\Big] + A_F\, p^{\gamma_F} + \kappa_{QF}\,\tfrac{s - 1}{v}.$$

$F$ is the probability that a point which lost the text has lost it by the word $s$; every
replacement word of the 995 points that lost the text is a trial (365955 words, 49.2% of them on
or after the line where the answer lost the text). By the same method: the bins of the word
number, the density and the share of the text before the word, a law through the bins of each
axis, the shifts - the plateau of the word law, the density law at $p \to 0$, the share line at
the start of the text - in the level $\lambda_F$. Up to the corner $s_*$ = 39.57 ± 22.63 words
(the first four lines) the word law is flat; past it every doubling of the word number adds
$\kappa_{SF}$ = 1.14 ± 0.209 logit; the width of the corner is held on its bound. $A_F$ =
6.95 ± 1.56 and $\gamma_F$ = 0.420 ± 0.155: in a dense text the model loses the text earlier;
$\kappa_{QF}$ = 3.19 ± 0.589 over the whole text, and the share line passes through the start of
the text ($\lambda_{QF}$ = -0.130 ± 0.640). The text length beyond the word number and the share
does not set the place (Wald 13.89 with 7 degrees of freedom, p = 0.0531). The bins of the number
of rules differ from zero (p = 0.000526) only in one way: the tasks with one rule lose the text
earlier, the bins from 2 to 128 rules lie 0.5 to 1.9 logit below with no trend, and the other bins
move by at most 1.06 of their SE.

![Place of the collapse of gpt-4o-mini by word number, after the shift](../artefacts/pilot_gpt4o_mini_v34/figures/place-S-after-shift.png)

### What follows

| Quantity at q = 0.95 | Binned | By law |
|---|---|---|
| IEC, rules | 36.58 ± 5.41 | 38.06 ± 6.13 |
| IGC, rules | 18.69 ± 3.62 | 17.55 ± 4.04 |
| FEC, words | — | 3277.37 ± 571.30 |

At the zero point, the start of the answer to a short text with sparse replacements,
gpt-4o-mini holds about 37 to 38 rules while 95% of the replacements over all points are right;
when every word of the text is replaced (p = 1), it holds about 18 rules (the IGC).
Fatigue alone spends the same reserve after about 3300 words of the answer (the FEC).

## How to read these laws

They are empirical regularities found on published points and open to checking: every number above
can be recomputed with `scripts/build_passport.py`, and the points can be refitted with any other
model. They are reasons to continue the measurements, on more models and in the cells named above,
not a statement that the data are exhausted.
