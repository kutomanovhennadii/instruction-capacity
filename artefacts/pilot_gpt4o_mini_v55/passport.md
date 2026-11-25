# Passport of `gpt-4o-mini-2024-07-18`

Measurement `pilot_gpt4o_mini_v55`, 5,172 answered points. The method is described in `docs/05-passport-model.md` of the repository.

## The capacities

The zero point of a capacity is a text of 20 words, the start of the answer, the density at the maximum of its law and one rule; every other axis of the capacity is set by its definition.

The capacities are taken at the level 0.5. At it the share of successes over all requests and the success of the median request reach one half at the same logit, zero, whatever the variance of the request shift: a capacity is the load at which the laws bring the logit of the median request to zero, and it does not depend on that variance.

### P_sync: a word of the text stands on its place

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | none: the law gives no root | at least 256, reserve left 35.17 |
| fatigue capacity, words | 0.5 | 17,376 (13,949–21,646), continuation of the law | - |

The instruction capacity is the number of rules at which the law of the number of rules brings the logit to zero, and next to it the walk along the bins of that axis; by the law it is none when the fall of that law over its bins does not differ from zero. The fatigue capacity is the word at which the full law of the grid at the density going to zero and one rule reaches zero, on a text of the same length: by the curve of the law, as for `H`.

### H: a rule word on its place is replaced correctly

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 36.6 (33.0–40.7) | 35.5 |
| fatigue capacity, words | 0.5 | 18,537 (14,047–24,462), continuation of the law | - |

### F: a misfire on a background word on its place

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 5.4·10⁷ (1.3·10⁷–2.3·10⁸), continuation of the law | at least 256, reserve left 9.94 |
| fatigue capacity, words | 0.5 | 4.0·10⁶ (1.7·10⁶–9.6·10⁶), continuation of the law | - |

The instruction capacity is the number of rules at the zero point. The fatigue capacity is the number of the last word of a text of the same length at the zero point.

## The standard stress test

The standard stress test describes how a model behaves as the complexity of a task with fixed proportions of the load increases step by step. The complexity increases until the probability of an error on the last word of the text reaches 50%. The density of replacements in the test is `p = 0.5`, and the text has one rule per 50 words: with `m` rules the text has `T = 50·m` words and `D = 2m` distinct words, each of them repeated 25 times. The length of the text and the number of rules increase together, keeping these proportions.

The point of 50% is set by the combined error of the last word of the median request, `E = 1 − P_sync·H = 0.5`, where `P_sync` is the probability that the word keeps its place in the text, and `H` is the conditional probability that the replacement is made correctly on a kept place. False replacements of background words do not enter the definition of the point, but they are measured at the point found.

The influence of each load is set by its price in the logit of the probability of the correct outcome of the last word. The influence keeps its sign: a negative value means that the load lowers the probability of an error of that kind. The three shares of the influence sum to 100; when the total price of the loads is zero or less, the influence is not defined. The mean share of errors over the text is computed over all its words for the median request. Every number has a 95% confidence interval by the delta method; the parameters of the three quantities are taken independent, since the passport has no covariances between its sections.

1. The point of 50%: T = 1187 ± 53 words, 23.7 ± 1.1 rules.
2. The combined error at the point: 50%.
3. The structure of the error at the point, per 100 errors: synchronization 90.4 ± 1.8, replacements 9.6 ± 1.8; background over them, per the same 100: 1.3 ± 0.2.
4. Synchronization: errors over the text for the median request 12.7 ± 0.5%; influence per 100 errors of synchronization at the point: length 65.3 ± 2.7, density 34.0 ± 2.6, rules 0.8 ± 0.8.
5. Replacements: errors over the text for the median request 3.8 ± 0.6%; influence per 100 errors of replacements at the point: length 83.0 ± 3.1, density 3.5 ± 3.3, rules 13.5 ± 2.0.
6. Background: errors over the text for the median request 0.9 ± 0.1%; influence per 100 errors of background at the point: length 57.0 ± 2.5, density 16.3 ± 1.7, rules 26.7 ± 2.6.
7. The combined error over the text for the median request: 15.7 ± 0.5%.

## The formula

`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`. The formula is the way to combine the three quantities: it has no law and no capacity of its own. `π = m/D` is the replacement density.

## P_sync: a word of the text stands on its place

Model VS-PM, request weights by the cube V × P × M. Fit: with the shift of a request; the variance of the request shift `σ² = 10.47`; 5,162 requests, 15,078,260 trials, 3,752,796 failures, 75 parameters; removed bins: vs_1_1, vs_1_2, vs_2_2, vs_2_3, vs_3_1, vs_3_2, vs_10_2, vs_10_4. The variance by the load turned bins against the median request of the data, so the quantity keeps one variance: the variance by the load turns bins against the median request: m_bin_4: passport -0.16, median request +0.52 ± 0.49; m_bin_6: passport -0.27, median request +0.79 ± 0.50; m_bin_8: passport -0.00, median request +2.27 ± 0.94; m_bin_9: passport -0.08, median request +1.12 ± 0.64.

The law of the grid: a length slice is a two-asymptote hyperbola, `[(κ·ℓ + λ) + μ − √((κ·ℓ + λ − μ)² + κ²·ω²)]/2` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −3.644 ± 0.360`, `lambda = 45.281 ± 5.926`, `omega = 4.390 ± 0.829`; `χ² = 40.3` over 56 cells; chosen by the smallest `QAIC` among 3 forms. The law of the plateau runs through the levels of the slices.

The law of the plateau is a line, `a + b·log₂(v/20)`: `a = 26.20 ± 2.02`, `b = −1.81 ± 0.24`; `χ² = 5.9` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a binary exponential, `μ − A·2^(γ·log₂(p/2⁻¹⁰))`: `mu = 9.56 ± 0.92`, `A = 0.62 ± 0.27`, `gamma = 0.48 ± 0.06`; `χ² = 7.7` over 10 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 0.02 ± 0.61`, `b = −0.06 ± 0.06`; `χ² = 6.1` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = δ_VS + δ_P + δ_M = 35.78 ± 2.03` is the logit at the density going to zero and one rule, and that density is itself a continuation of the law.

![P_sync: S before correction](figures/P_sync/S-before-correction.png)

![P_sync: S after correction](figures/P_sync/S-after-correction.png)

![P_sync: plateau before correction](figures/P_sync/plateau-before-correction.png)

![P_sync: P before correction](figures/P_sync/P-before-correction.png)

![P_sync: P after correction](figures/P_sync/P-after-correction.png)

![P_sync: M before correction](figures/P_sync/M-before-correction.png)

![P_sync: M after correction](figures/P_sync/M-after-correction.png)

## H: a rule word on its place is replaced correctly

Model VS-PMR, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 3.04`; 5,135 requests, 1,152,598 trials, 119,630 failures, 82 parameters; removed bins: vs_1_1, vs_1_2, vs_2_1, vs_2_2, vs_2_3, vs_3_1, vs_3_2, vs_4_1, vs_10_2.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 24 parameters more than one variance, `2·Δ ln L = 719.3`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 116.7 | 0.10 | 1.90 ± 0.72 |
| text length, words | 267.7 | 0.07 | 1.86 ± 0.51 |
| text length, words | 505.9 | 0.10 | 1.52 ± 0.35 |
| text length, words | 1009 | 0.13 | 1.70 ± 0.35 |
| text length, words | 2025 | 0.20 | 1.67 ± 0.28 |
| text length, words | 4015 | 0.23 | 1.61 ± 0.27 |
| text length, words | 8064 | 0.17 | 1.39 ± 0.24 |
| density | 0.001093 | 0.02 | 0.47 ± 0.36 |
| density | 0.002322 | 0.03 | 0.70 ± 0.42 |
| density | 0.005236 | 0.06 | 1.24 ± 0.41 |
| density | 0.01025 | 0.06 | 1.28 ± 0.39 |
| density | 0.02029 | 0.06 | 1.13 ± 0.35 |
| density | 0.04047 | 0.08 | 1.35 ± 0.37 |
| density | 0.07807 | 0.12 | 1.31 ± 0.29 |
| density | 0.1495 | 0.13 | 1.73 ± 0.34 |
| density | 0.2952 | 0.16 | 1.53 ± 0.27 |
| density | 0.6035 | 0.18 | 2.20 ± 0.47 |
| density | 1 | 0.11 | 3.66 ± 0.96 |
| number of rules | 1 | 0.14 | 7.99 ± 1.72 |
| number of rules | 2 | 0.14 | 5.28 ± 1.03 |
| number of rules | 3.623 | 0.16 | 3.17 ± 0.57 |
| number of rules | 6.855 | 0.17 | 1.63 ± 0.33 |
| number of rules | 13.83 | 0.10 | 0.93 ± 0.23 |
| number of rules | 25.7 | 0.11 | 0.69 ± 0.16 |
| number of rules | 50 | 0.09 | 0.42 ± 0.13 |
| number of rules | 102.4 | 0.07 | 0.16 ± 0.05 |
| number of rules | 192.8 | 0.03 | 0.13 ± 0.05 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = 2.34`, `b = −0.067` | 2.3 over 7 groups |
| density | `a + b·log₂ p` | `a = 2.18`, `b = 0.164` | 28.0 over 11 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 0.06`, `A = 18.68`, `m₀ = 0.73` | 8.0 over 9 groups |

The law of the grid: a length slice is a two-asymptote hyperbola, `[(κ·ℓ + λ) + μ − √((κ·ℓ + λ − μ)² + κ²·ω²)]/2` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −1.289 ± 0.097`, `lambda = 20.368 ± 1.366`, `omega = 3.910 ± 0.680`; `χ² = 43.5` over 56 cells; chosen by the smallest `QAIC` among 3 forms. The law of the levels runs through the levels of the slices.

The law of the levels of the slices is a line, `a + b·log₂(v/20)`: `a = 20.06 ± 1.77`, `b = −1.42 ± 0.23`; `χ² = 5.9` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a two-slope hyperbola, `μ + β·(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `mu = 2.92 ± 2.06`, `kappa = −1.75 ± 0.78`, `corner = 6.77 ± 0.76`, `omega = 2.11 ± 1.74`, `beta = 0.42 ± 0.25`; `χ² = 4.7` over 10 bins; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −0.75 ± 0.48`, `b = −0.44 ± 0.05`; `χ² = 6.5` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the rule number is a slanted hyperbola, `μ + (κ/2)·[(log₂ r − log₂ r*) + √((log₂ r − log₂ r*)² + ω²)]`: `mu = 0.09 ± 0.22`, `kappa = −0.51 ± 0.04`, `corner = 3.29 ± 0.23`, `omega = 0.57 ± 0.65`; `χ² = 4.5` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = μ₀ + μ_P + λ_M + λ_R = 19.19 ± 1.79`.

![H: S before correction](figures/H/S-before-correction.png)

![H: S after correction](figures/H/S-after-correction.png)

![H: plateau before correction](figures/H/plateau-before-correction.png)

![H: P before correction](figures/H/P-before-correction.png)

![H: P after correction](figures/H/P-after-correction.png)

![H: M before correction](figures/H/M-before-correction.png)

![H: M after correction](figures/H/M-after-correction.png)

![H: R before correction](figures/H/R-before-correction.png)

![H: R after correction](figures/H/R-after-correction.png)

![H: spread V](figures/H/spread-V.png)

![H: spread P](figures/H/spread-P.png)

![H: spread M](figures/H/spread-M.png)

## F: a misfire on a background word on its place

Model V-(s/v)-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 6.08`; 4,571 requests, 10,212,207 trials, 76,562 failures, 34 parameters; removed bins: v_bin_1, v_bin_2.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 22 parameters more than one variance, `2·Δ ln L = 1315.6`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 172.1 | 0.11 | 6.62 ± 1.80 |
| text length, words | 501.7 | 0.09 | 3.98 ± 1.09 |
| text length, words | 1007 | 0.13 | 2.60 ± 0.50 |
| text length, words | 2025 | 0.23 | 1.50 ± 0.19 |
| text length, words | 4015 | 0.25 | 1.07 ± 0.13 |
| text length, words | 8064 | 0.19 | 0.96 ± 0.13 |
| density | 0.001093 | 0.02 | 0.44 ± 0.19 |
| density | 0.002322 | 0.04 | 0.67 ± 0.21 |
| density | 0.005236 | 0.06 | 1.07 ± 0.26 |
| density | 0.01025 | 0.06 | 1.50 ± 0.34 |
| density | 0.02029 | 0.06 | 1.63 ± 0.37 |
| density | 0.04047 | 0.09 | 1.61 ± 0.30 |
| density | 0.07807 | 0.13 | 1.51 ± 0.26 |
| density | 0.1495 | 0.15 | 1.77 ± 0.31 |
| density | 0.2952 | 0.18 | 2.14 ± 0.32 |
| density | 0.6035 | 0.21 | 2.90 ± 0.56 |
| number of rules | 1 | 0.16 | 6.76 ± 1.11 |
| number of rules | 2 | 0.15 | 4.73 ± 0.86 |
| number of rules | 3.54 | 0.14 | 3.64 ± 0.54 |
| number of rules | 6.736 | 0.17 | 2.38 ± 0.34 |
| number of rules | 13.78 | 0.10 | 0.95 ± 0.19 |
| number of rules | 25.61 | 0.11 | 0.57 ± 0.14 |
| number of rules | 49.5 | 0.08 | 0.29 ± 0.08 |
| number of rules | 100.3 | 0.06 | 0.18 ± 0.05 |
| number of rules | 194.4 | 0.03 | 0.13 ± 0.05 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `c + A/(1 + v/v₀)` | `c = 0.67`, `A = 14.37`, `v₀ = 133.94` | 3.3 over 6 groups |
| density | `a + b·log₂ p` | `a = 2.58`, `b = 0.209` | 16.4 over 10 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 0.03`, `A = 12.45`, `m₀ = 1.31` | 11.3 over 9 groups |

The laws run through the bins without the poorly determined ones; a bin weighs `1/(σᵢ² + τ²)`, its own variance and the scatter of the bins around the law, and the errors are carried by the full covariance of the bins.

The law of the length is a line, `a + b·log₂(v/20)`: `a = 11.53 ± 0.63`, `b = −0.71 ± 0.09`; `χ² = 31.9` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the share of the answer is an exponential, `c + A·e^(−q/τ)`: `c = −0.23 ± 0.20`, `A = 1.13 ± 0.19`, `tau = 0.62 ± 0.23`; `χ² = 9.7` over 9 bins; chosen by the smallest `QAIC` among 2 forms. A fall of 0.90 over the whole answer.

The law of the density is a logarithm plus a line, `a + b·log₂(p/2⁻¹⁰) + c·0.0009765625·2^(log₂(p/2⁻¹⁰))`: `a = 0.87 ± 0.25`, `b = 0.02 ± 0.04`, `c = −3.22 ± 0.49`; `χ² = 11.4` over 9 bins; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −0.02 ± 0.28`, `b = −0.52 ± 0.03`; `χ² = 7.0` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α_F = V(20) + q(0) + δ_P + M(1) = 13.32 ± 0.71`, where `δ_P` is the maximum of the law of the density, at `p = 0.0104`; `V(20)` = 11.53 ± 0.63, `q(0)` = 0.90 ± 0.10, `δ_P` = 0.92 ± 0.14, `M(1)` = −0.02 ± 0.28. The edge of the measured area: 256 rules and 10,240 words.

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
