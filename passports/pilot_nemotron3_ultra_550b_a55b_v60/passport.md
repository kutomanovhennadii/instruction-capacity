# Passport of `nvidia/nemotron-3-ultra-550b-a55b:free`

Measurement `pilot_nemotron3_ultra_550b_a55b_v60`, 6,702 answered points. The method is described in `docs/05-passport-model.md` of the repository.

## The capacities

The zero point of a capacity is a text of 20 words, the start of the answer, the density at the maximum of its law and one rule; every other axis of the capacity is set by its definition.

The capacities are taken at the level 0.5. At it the share of successes over all requests and the success of the median request reach one half at the same logit, zero, whatever the variance of the request shift: a capacity is the load at which the laws bring the logit of the median request to zero, and it does not depend on that variance.

### P_sync: a word of the text stands on its place

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | none: the law gives no root | at least 256, reserve left 30.56 |
| fatigue capacity, words | 0.5 | 49,982 (39,314–63,543), continuation of the law | - |

The instruction capacity is the number of rules at which the law of the number of rules brings the logit to zero, and next to it the walk along the bins of that axis; by the law it is none when the fall of that law over its bins does not differ from zero. The fatigue capacity is the word at which the full law of the grid at the density going to zero and one rule reaches zero, on a text of the same length: by the curve of the law, as for `H`.

### H: a rule word on its place is replaced correctly

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 17.1 (16.0–18.3) | 16.6 |
| fatigue capacity, words | 0.5 | 39,820 (35,587–44,558) | - |

### F: a misfire on a background word on its place

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 1.4·10¹⁶ (1.3·10¹³–1.5·10¹⁹), continuation of the law | at least 256, reserve left 18.89 |
| fatigue capacity, words | 0.5 | 1.1·10¹⁰ (1.4·10⁹–9.3·10¹⁰), continuation of the law | - |

The instruction capacity is the number of rules at the zero point. The fatigue capacity is the number of the last word of a text of the same length at the zero point.

## The standard stress test

The standard stress test describes how a model behaves as the complexity of a task with fixed proportions of the load increases step by step. The complexity increases until the probability of an error on the last word of the text reaches 50%. The density of replacements in the test is `p = 0.5`, and the text has one rule per 50 words: with `m` rules the text has `T = 50·m` words and `D = 2m` distinct words, each of them repeated 25 times. The length of the text and the number of rules increase together, keeping these proportions.

The point of 50% is set by the combined error of the last word of the median request, `E = 1 − P_sync·H = 0.5`, where `P_sync` is the probability that the word keeps its place in the text, and `H` is the conditional probability that the replacement is made correctly on a kept place. False replacements of background words do not enter the definition of the point, but they are measured at the point found.

The influence of each load is set by its price in the logit of the probability of the correct outcome of the last word. The influence keeps its sign: a negative value means that the load lowers the probability of an error of that kind. The three shares of the influence sum to 100; when the total price of the loads is zero or less, the influence is not defined. The mean share of errors over the text is computed over all its words for the median request. Every number has a 95% confidence interval by the delta method; the parameters of the three quantities are taken independent, since the passport has no covariances between its sections.

1. The point of 50%: T = 3613 ± 164 words, 72.3 ± 3.3 rules.
2. The combined error at the point: 50%.
3. The structure of the error at the point, per 100 errors: synchronization 57.9 ± 6.0, replacements 42.1 ± 6.0; background over them, per the same 100: 1.6 ± 0.4.
4. Synchronization: errors over the text for the median request 8.1 ± 0.8%; influence per 100 errors of synchronization at the point: length 49.8 ± 3.2, density 50.3 ± 3.2, rules −0.1 ± 1.2.
5. Replacements: errors over the text for the median request 12.5 ± 1.4%; influence per 100 errors of replacements at the point: length 62.7 ± 2.3, density −0.5 ± 3.0, rules 37.8 ± 3.0.
6. Background: errors over the text for the median request 0.4 ± 0.1%; influence per 100 errors of background at the point: length 42.6 ± 2.6, density 42.9 ± 2.7, rules 14.5 ± 2.9.
7. The combined error over the text for the median request: 18.8 ± 0.8%.

## The formula

`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`. The formula is the way to combine the three quantities: it has no law and no capacity of its own. `π = m/D` is the replacement density.

## P_sync: a word of the text stands on its place

Model VS-PM, request weights by the cube V × P × M. Fit: with the shift of a request; the variance of the request shift `σ² = 9.90`; 6,647 requests, 58,000,300 trials, 12,543,473 failures, 99 parameters; removed bins: vs_1_1, vs_1_2, vs_2_2, vs_3_1, vs_3_2, vs_3_3, vs_3_4, vs_4_1, vs_5_1, vs_7_1. The variance by the load turned bins against the median request of the data, so the quantity keeps one variance: the variance by the load turns bins against the median request: m_bin_3: passport -0.28, median request +0.68 ± 0.23; m_bin_4: passport -0.41, median request +1.10 ± 0.31; m_bin_5: passport -0.71, median request +1.26 ± 0.36; m_bin_6: passport -0.27, median request +1.62 ± 0.30; m_bin_7: passport -0.45, median request +2.48 ± 0.42; m_bin_8: passport -0.49, median request +2.36 ± 0.60; m_bin_9: passport -0.99, median request +2.58 ± 1.14.

The law of the grid: a length slice is a binary exponential, `μ − A·2^(γ·ℓ)` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `A = 0.796 ± 0.239`, `gamma = 0.318 ± 0.031`; `χ² = 65.7` over 80 cells; chosen by the smallest `QAIC` among 3 forms. The law of the plateau runs through the levels of the slices.

The law of the plateau is a line, `a + b·log₂(v/20)`: `a = 17.24 ± 1.38`, `b = −0.60 ± 0.18`; `χ² = 5.4` over 10 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a binary exponential, `μ − A·2^(γ·log₂(p/2⁻¹⁰))`: `mu = 13.33 ± 1.50`, `A = 3.21 ± 0.96`, `gamma = 0.25 ± 0.03`; `χ² = 9.8` over 11 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 0.99 ± 0.48`, `b = 0.0056 ± 0.06`; `χ² = 6.0` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = δ_VS + δ_P + δ_M = 31.56 ± 1.95` is the logit at the density going to zero and one rule, and that density is itself a continuation of the law.

![P_sync: S before correction](figures/P_sync/S-before-correction.png)

![P_sync: S after correction](figures/P_sync/S-after-correction.png)

![P_sync: plateau before correction](figures/P_sync/plateau-before-correction.png)

![P_sync: P before correction](figures/P_sync/P-before-correction.png)

![P_sync: P after correction](figures/P_sync/P-after-correction.png)

![P_sync: M before correction](figures/P_sync/M-before-correction.png)

![P_sync: M after correction](figures/P_sync/M-after-correction.png)

## H: a rule word on its place is replaced correctly

Model VS-PMR, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 3.37`; 6,632 requests, 3,591,444 trials, 958,475 failures, 103 parameters; removed bins: vs_1_1, vs_1_2, vs_2_1, vs_2_2, vs_3_1, vs_3_2, vs_3_3, vs_3_4, vs_4_1, vs_4_2, vs_5_1, vs_5_2, vs_6_2, vs_12_2.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 26 parameters more than one variance, `2·Δ ln L = 960.9`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 171.5 | 0.08 | 2.49 ± 1.46 |
| text length, words | 534.8 | 0.06 | 1.38 ± 0.47 |
| text length, words | 1027 | 0.10 | 2.09 ± 0.54 |
| text length, words | 2038 | 0.11 | 1.84 ± 0.43 |
| text length, words | 4076 | 0.18 | 2.07 ± 0.31 |
| text length, words | 7779 | 0.20 | 2.70 ± 0.55 |
| text length, words | 1.53e+04 | 0.17 | 2.42 ± 0.59 |
| text length, words | 3.084e+04 | 0.10 | 1.63 ± 0.41 |
| density | 0.0006288 | 0.03 | 2.84 ± 2.13 |
| density | 0.001273 | 0.04 | 1.58 ± 0.84 |
| density | 0.002666 | 0.06 | 2.90 ± 1.39 |
| density | 0.005233 | 0.06 | 1.46 ± 0.67 |
| density | 0.01046 | 0.08 | 1.06 ± 0.32 |
| density | 0.02044 | 0.07 | 1.44 ± 0.78 |
| density | 0.04046 | 0.09 | 1.35 ± 0.39 |
| density | 0.08032 | 0.10 | 1.37 ± 0.31 |
| density | 0.1538 | 0.11 | 1.66 ± 0.35 |
| density | 0.289 | 0.12 | 2.10 ± 0.44 |
| density | 0.6143 | 0.12 | 4.83 ± 1.14 |
| density | 1 | 0.13 | 5.26 ± 1.30 |
| number of rules | 1 | 0.13 | 4.59 ± 1.53 |
| number of rules | 2 | 0.11 | 3.89 ± 1.20 |
| number of rules | 3.52 | 0.14 | 4.49 ± 1.00 |
| number of rules | 6.768 | 0.13 | 4.66 ± 0.97 |
| number of rules | 13.51 | 0.10 | 2.79 ± 0.62 |
| number of rules | 25.64 | 0.18 | 2.22 ± 0.62 |
| number of rules | 51.66 | 0.10 | 0.70 ± 0.22 |
| number of rules | 101.9 | 0.08 | 0.23 ± 0.07 |
| number of rules | 217 | 0.04 | 0.13 ± 0.04 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = 1.48`, `b = 0.049` | 17.4 over 8 groups |
| density | `a + b·log₂ p` | `a = 2.71`, `b = 0.164` | 112.5 over 12 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 1.7·10⁻³³`, `A = 7.23`, `m₀ = 6.13` | 49.5 over 9 groups |

The law of the grid: a length slice is a two-asymptote hyperbola, `[(κ·ℓ + λ) + μ − √((κ·ℓ + λ − μ)² + κ²·ω²)]/2` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −1.286 ± 0.052`, `lambda = 21.642 ± 0.712`, `omega = 2.089 ± 0.406`; `χ² = 69.4` over 76 cells; chosen by the smallest `QAIC` among 3 forms. The law of the levels runs through the levels of the slices.

The law of the levels of the slices is a binary exponential, `μ − A·2^(γ·log₂(v/20))`: `mu = 13.69 ± 0.69`, `A = 0.00073 ± 0.0021`, `gamma = 1.27 ± 0.40`; `χ² = 8.0` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a two-slope hyperbola, `μ + β·(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `mu = 9.56 ± 3.48`, `kappa = −5.01 ± 0.37`, `corner = 10.00 ± 0.00`, `omega = 4.77 ± 1.25`, `beta = 0.78 ± 0.23`; `χ² = 7.5` over 11 bins; on the bound of the search: corner; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −0.18 ± 0.39`, `b = −0.56 ± 0.04`; `χ² = 7.6` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the rule number is a line, `a + b·log₂ r`: `a = 0.31 ± 0.19`, `b = −0.23 ± 0.02`; `χ² = 5.6` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = μ₀ + μ_P + λ_M + λ_R = 12.85 ± 0.71`.

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

Model V-(s/v)-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 4.21`; 5,762 requests, 41,866,494 trials, 16,911 failures, 36 parameters; removed bins: v_bin_1, v_bin_2, v_bin_3.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 22 parameters more than one variance, `2·Δ ln L = 309.4`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 619.1 | 0.16 | 3.10 ± 1.10 |
| text length, words | 2026 | 0.11 | 3.45 ± 1.01 |
| text length, words | 4097 | 0.19 | 3.87 ± 0.74 |
| text length, words | 7751 | 0.22 | 3.79 ± 0.61 |
| text length, words | 1.53e+04 | 0.20 | 2.34 ± 0.48 |
| text length, words | 3.085e+04 | 0.12 | 2.46 ± 0.76 |
| density | 0.0009696 | 0.08 | 5.04 ± 2.78 |
| density | 0.002664 | 0.06 | 5.69 ± 3.48 |
| density | 0.005233 | 0.07 | 2.80 ± 1.15 |
| density | 0.01046 | 0.10 | 4.45 ± 1.68 |
| density | 0.02044 | 0.08 | 2.44 ± 0.77 |
| density | 0.04046 | 0.11 | 1.77 ± 0.51 |
| density | 0.08032 | 0.11 | 2.15 ± 0.50 |
| density | 0.1539 | 0.12 | 2.72 ± 0.60 |
| density | 0.2892 | 0.14 | 3.10 ± 0.69 |
| density | 0.6136 | 0.14 | 4.28 ± 1.15 |
| number of rules | 1 | 0.15 | 4.82 ± 1.36 |
| number of rules | 2 | 0.12 | 5.59 ± 2.02 |
| number of rules | 3.466 | 0.14 | 7.10 ± 2.48 |
| number of rules | 6.673 | 0.13 | 2.72 ± 0.69 |
| number of rules | 13.37 | 0.10 | 3.54 ± 0.89 |
| number of rules | 24.86 | 0.14 | 2.19 ± 0.52 |
| number of rules | 50.68 | 0.09 | 2.13 ± 0.66 |
| number of rules | 101.4 | 0.08 | 1.09 ± 0.33 |
| number of rules | 216.1 | 0.04 | 0.49 ± 0.31 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `c + A/(1 + v/v₀)` | `c = 0.02`, `A = 3.97`, `v₀ = 38635.51` | 9.2 over 6 groups |
| density | `a + b·log₂ p` | `a = 2.84`, `b = −0.008` | 41.0 over 10 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 0.13`, `A = 5.00`, `m₀ = 22.75` | 21.0 over 9 groups |

The laws run through the bins without the poorly determined ones; a bin weighs `1/(σᵢ² + τ²)`, its own variance and the scatter of the bins around the law, and the errors are carried by the full covariance of the bins.

The law of the length is a line, `a + b·log₂(v/20)`: `a = 12.46 ± 0.85`, `b = −0.67 ± 0.11`; `χ² = 3.1` over 9 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the share of the answer is a line, `a + b·q`: `a = 2.44 ± 0.26`, `b = −2.39 ± 0.30`; `χ² = 8.5` over 9 bins; chosen by the smallest `QAIC` among 2 forms. A fall of 2.39 over the whole answer.

The law of the density is a logarithm plus a line, `a + b·log₂(p/2⁻¹⁰) + c·0.0009765625·2^(log₂(p/2⁻¹⁰))`: `a = 6.12 ± 0.78`, `b = −0.61 ± 0.12`, `c = −4.01 ± 1.07`; `χ² = 4.2` over 10 bins; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 0.86 ± 0.62`, `b = −0.41 ± 0.07`; `χ² = 6.0` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α_F = V(20) + q(0) + δ_P + M(1) = 21.87 ± 1.32`, where `δ_P` is the maximum of the law of the density, at `p = 0.0010`; `V(20)` = 12.46 ± 0.85, `q(0)` = 2.44 ± 0.26, `δ_P` = 6.11 ± 0.78, `M(1)` = 0.86 ± 0.62. The edge of the measured area: 256 rules and 40,960 words.

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
