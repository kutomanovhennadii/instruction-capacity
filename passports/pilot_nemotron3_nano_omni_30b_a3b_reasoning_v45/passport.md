# Passport of `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`

Measurement `pilot_nemotron3_nano_omni_30b_a3b_reasoning_v45`, 3,695 answered points. The method is described in `docs/05-passport-model.md` of the repository.

## The capacities

The zero point of a capacity is a text of 20 words, the start of the answer, the density at the maximum of its law and one rule; every other axis of the capacity is set by its definition.

The capacities are taken at the level 0.5. At it the share of successes over all requests and the success of the median request reach one half at the same logit, zero, whatever the variance of the request shift: a capacity is the load at which the laws bring the logit of the median request to zero, and it does not depend on that variance.

### P_sync: a word of the text stands on its place

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 3.1·10²¹ (1.8·10⁸–5.5·10³⁴), continuation of the law | at least 256, reserve left 18.45 |
| fatigue capacity, words | 0.5 | 26,277 (13,370–51,645), continuation of the law | - |

The instruction capacity is the number of rules at which the law of the number of rules brings the logit to zero, and next to it the walk along the bins of that axis; by the law it is none when the fall of that law over its bins does not differ from zero. The fatigue capacity is the word at which the full law of the grid at the density going to zero and one rule reaches zero, on a text of the same length: by the curve of the law, as for `H`.

### H: a rule word on its place is replaced correctly

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 34.3 (29.7–39.5) | 34.9 |
| fatigue capacity, words | 0.5 | 838 (714–983) | - |

### F: a misfire on a background word on its place

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 30,470 (632–1.5·10⁶), continuation of the law | at least 256, reserve left 1.88 |
| fatigue capacity, words | 0.5 | none: the law gives no root | - |

The instruction capacity is the number of rules at the zero point. The fatigue capacity is the number of the last word of a text of the same length at the zero point.

## The standard stress test

The standard stress test describes how a model behaves as the complexity of a task with fixed proportions of the load increases step by step. The complexity increases until the probability of an error on the last word of the text reaches 50%. The density of replacements in the test is `p = 0.5`, and the text has one rule per 50 words: with `m` rules the text has `T = 50·m` words and `D = 2m` distinct words, each of them repeated 25 times. The length of the text and the number of rules increase together, keeping these proportions.

The point of 50% is set by the combined error of the last word of the median request, `E = 1 − P_sync·H = 0.5`, where `P_sync` is the probability that the word keeps its place in the text, and `H` is the conditional probability that the replacement is made correctly on a kept place. False replacements of background words do not enter the definition of the point, but they are measured at the point found.

The influence of each load is set by its price in the logit of the probability of the correct outcome of the last word. The influence keeps its sign: a negative value means that the load lowers the probability of an error of that kind. The three shares of the influence sum to 100; when the total price of the loads is zero or less, the influence is not defined. The mean share of errors over the text is computed over all its words for the median request. Every number has a 95% confidence interval by the delta method; the parameters of the three quantities are taken independent, since the passport has no covariances between its sections.

1. The point of 50%: T = 164 ± 19 words, 3.3 ± 0.4 rules.
2. The combined error at the point: 50%.
3. The structure of the error at the point, per 100 errors: synchronization 0.3 ± 0.1, replacements 99.7 ± 0.1; background over them, per the same 100: 0.2 ± 0.1.
4. Synchronization: errors over the text for the median request 0.1 ± 0.0%; influence per 100 errors of synchronization at the point: length 1.9 ± 1.8, density 94.6 ± 3.4, rules 3.5 ± 2.4.
5. Replacements: errors over the text for the median request 31.6 ± 0.7%; influence per 100 errors of replacements at the point: length 76.5 ± 4.7, density 23.9 ± 4.9, rules −0.3 ± 0.4.
6. Background: errors over the text for the median request 0.3 ± 0.1%; influence per 100 errors of background at the point: length not defined, density not defined, rules not defined.
7. The combined error over the text for the median request: 31.7 ± 0.7%.

## The formula

`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`. The formula is the way to combine the three quantities: it has no law and no capacity of its own. `π = m/D` is the replacement density.

## P_sync: a word of the text stands on its place

Model VS-PM, request weights by the cube V × P × M. Fit: with the shift of a request; the variance of the request shift `σ² = 30.00`; 3,695 requests, 14,277,800 trials, 5,734,575 failures, 96 parameters; removed bins: none. The variance by the load turned bins against the median request of the data, so the quantity keeps one variance: the variance by the load turns bins against the median request: m_bin_4: passport -1.22, median request +1.01 ± 0.76; m_bin_5: passport -1.61, median request +1.34 ± 0.80.

The law of the grid: a length slice is a two-asymptote hyperbola, `[(κ·ℓ + λ) + μ − √((κ·ℓ + λ − μ)² + κ²·ω²)]/2` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −3.238 ± 0.255`, `lambda = 35.549 ± 2.759`, `omega = 0.596 ± 0.312`; `χ² = 65.7` over 77 cells; chosen by the smallest `QAIC` among 3 forms. The law of the plateau runs through the levels of the slices.

The law of the plateau is a slanted hyperbola, `μ + (κ/2)·[(log₂(v/20) − log₂(v/20)*) + √((log₂(v/20) − log₂(v/20)*)² + ω²)]`: `mu = 8.35 ± 0.81`, `kappa = 0.65 ± 0.53`, `corner = 4.79 ± 2.01`, `omega = 0.0010 ± 0.00`; `χ² = 11.8` over 11 bins; on the bound of the search: omega; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a binary exponential, `μ − A·2^(γ·log₂(p/2⁻¹⁰))`: `mu = 10.99 ± 3.94`, `A = 3.79 ± 2.84`, `gamma = 0.20 ± 0.07`; `χ² = 7.1` over 11 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 1.03 ± 0.72`, `b = −0.29 ± 0.15`; `χ² = 4.8` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = δ_VS + δ_P + δ_M = 20.38 ± 3.90` is the logit at the density going to zero and one rule, and that density is itself a continuation of the law.

![P_sync: S before correction](figures/P_sync/S-before-correction.png)

![P_sync: S after correction](figures/P_sync/S-after-correction.png)

![P_sync: plateau before correction](figures/P_sync/plateau-before-correction.png)

![P_sync: P before correction](figures/P_sync/P-before-correction.png)

![P_sync: P after correction](figures/P_sync/P-after-correction.png)

![P_sync: M before correction](figures/P_sync/M-before-correction.png)

![P_sync: M after correction](figures/P_sync/M-after-correction.png)

## H: a rule word on its place is replaced correctly

Model VS-PMR, request weights by the cube V × P × M. Fit: with the shift of a request; the variance of the request shift `σ² = 1.05`; 3,656 requests, 829,924 trials, 714,453 failures, 103 parameters; removed bins: vs_11_11. The variance by the load turned bins against the median request of the data, so the quantity keeps one variance: the variance by the load turns bins against the median request: m_bin_3: passport -0.20, median request +0.74 ± 0.66; m_bin_4: passport -0.00, median request +0.92 ± 0.76.

The law of the grid: a length slice is a two-asymptote hyperbola, `[(κ·ℓ + λ) + μ − √((κ·ℓ + λ − μ)² + κ²·ω²)]/2` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −19.027 ± 74.680`, `lambda = 353.287 ± 1479.479`, `omega = 4.545 ± 7.478`; `χ² = 78.4` over 76 cells; chosen by the smallest `QAIC` among 3 forms. The law of the levels runs through the levels of the slices.

The law of the levels of the slices is a line, `a + b·log₂(v/20)`: `a = 10.05 ± 1.96`, `b = −0.39 ± 0.07`; `χ² = 0.1` over 11 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a binary exponential, `μ − A·2^(γ·log₂(p/2⁻¹⁰))`: `mu = 2.65 ± 0.42`, `A = 0.20 ± 0.12`, `gamma = 0.44 ± 0.08`; `χ² = 4.8` over 11 bins; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 0.52 ± 0.28`, `b = 0.05 ± 0.03`; `χ² = 10.5` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the rule number is a binary exponential, `μ − A·2^(γ·log₂ r)`: `mu = −0.15 ± 0.19`, `A = 0.08 ± 0.04`, `gamma = 0.64 ± 0.10`; `χ² = 1.0` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = μ₀ + μ_P + λ_M + λ_R = 12.99 ± 1.90`.

![H: S before correction](figures/H/S-before-correction.png)

![H: S after correction](figures/H/S-after-correction.png)

![H: plateau before correction](figures/H/plateau-before-correction.png)

![H: P before correction](figures/H/P-before-correction.png)

![H: P after correction](figures/H/P-after-correction.png)

![H: M before correction](figures/H/M-before-correction.png)

![H: M after correction](figures/H/M-after-correction.png)

![H: R before correction](figures/H/R-before-correction.png)

![H: R after correction](figures/H/R-after-correction.png)

## F: a misfire on a background word on its place

Model V-(s/v)-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 2.27`; 3,146 requests, 7,690,262 trials, 4,193 failures, 38 parameters; removed bins: none.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 23 parameters more than one variance, `2·Δ ln L = 331.8`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 84.96 | 0.09 | 1.29 ± 0.80 |
| text length, words | 280.7 | 0.08 | 0.86 ± 0.48 |
| text length, words | 534.3 | 0.12 | 0.96 ± 0.44 |
| text length, words | 1067 | 0.11 | 1.14 ± 0.50 |
| text length, words | 2007 | 0.15 | 1.06 ± 0.53 |
| text length, words | 3925 | 0.19 | 1.40 ± 0.51 |
| text length, words | 7883 | 0.16 | 0.95 ± 0.43 |
| text length, words | 1.672e+04 | 0.11 | 1.08 ± 0.83 |
| density | 0.001503 | 0.09 | 0.09 ± 0.12 |
| density | 0.005295 | 0.05 | 0.40 ± 0.34 |
| density | 0.01021 | 0.07 | 0.72 ± 0.44 |
| density | 0.02149 | 0.09 | 0.40 ± 0.26 |
| density | 0.03979 | 0.09 | 0.48 ± 0.23 |
| density | 0.08018 | 0.12 | 1.28 ± 0.47 |
| density | 0.1612 | 0.15 | 2.30 ± 0.65 |
| density | 0.3114 | 0.15 | 2.11 ± 1.08 |
| density | 0.643 | 0.19 | 4.11 ± 1.81 |
| number of rules | 1 | 0.16 | 3.21 ± 1.46 |
| number of rules | 2 | 0.14 | 2.57 ± 1.09 |
| number of rules | 3.402 | 0.16 | 2.24 ± 0.75 |
| number of rules | 6.688 | 0.11 | 1.38 ± 0.49 |
| number of rules | 12.95 | 0.12 | 0.66 ± 0.24 |
| number of rules | 24.97 | 0.11 | 0.79 ± 0.61 |
| number of rules | 50.51 | 0.10 | 0.32 ± 0.17 |
| number of rules | 97.35 | 0.07 | 0.17 ± 0.12 |
| number of rules | 199.6 | 0.05 | 0.20 ± 0.26 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `c + A/v` | `c = 1.10`, `A = 3.29` | 3.5 over 8 groups |
| density | `a + b·log₂ p` | `a = 2.34`, `b = 0.243` | 35.2 over 9 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 0.10`, `A = 4.79`, `m₀ = 2.27` | 3.8 over 9 groups |

The laws run through the bins without the poorly determined ones; a bin weighs `1/(σᵢ² + τ²)`, its own variance and the scatter of the bins around the law, and the errors are carried by the full covariance of the bins.

The law of the length is a binary exponential, `μ − A·2^(γ·log₂(v/20))`: `mu = 5.96 ± 2.07`, `A = −0.91 ± 1.29`, `gamma = 0.26 ± 0.15`; `χ² = 21.7` over 11 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the share of the answer is an exponential, `c + A·e^(−q/τ)`: `c = 0.03 ± 0.21`, `A = −3.57 ± 0.41`, `tau = 0.13 ± 0.03`; `χ² = 2.9` over 9 bins; chosen by the smallest `QAIC` among 2 forms. A fall of −3.57 over the whole answer.

The law of the density is a logarithm plus a line, `a + b·log₂(p/2⁻¹⁰) + c·0.0009765625·2^(log₂(p/2⁻¹⁰))`: `a = −0.12 ± 0.53`, `b = 0.10 ± 0.08`, `c = −2.17 ± 0.79`; `χ² = 7.7` over 10 bins; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −0.02 ± 0.39`, `b = −0.25 ± 0.07`; `χ² = 7.7` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α_F = V(20) + q(0) + δ_P + M(1) = 3.68 ± 0.89`, where `δ_P` is the maximum of the law of the density, at `p = 0.0690`; `V(20)` = 6.87 ± 0.86, `q(0)` = −3.54 ± 0.41, `δ_P` = 0.37 ± 0.27, `M(1)` = −0.02 ± 0.39. The edge of the measured area: 256 rules and 20,480 words.

![F: V before correction](figures/F/V-before-correction.png)

![F: V after correction](figures/F/V-after-correction.png)

![F: answer share before correction](figures/F/answer-share-before-correction.png)

![F: answer share after correction](figures/F/answer-share-after-correction.png)

![F: P before correction](figures/F/P-before-correction.png)

![F: P after correction](figures/F/P-after-correction.png)

![F: M before correction](figures/F/M-before-correction.png)

![F: M after correction](figures/F/M-after-correction.png)

![F: spread V](figures/F/spread-V.png)

![F: spread P](figures/F/spread-P.png)

![F: spread M](figures/F/spread-M.png)
