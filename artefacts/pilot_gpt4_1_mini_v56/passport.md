# Passport of `gpt-4.1-mini-2025-04-14`

Measurement `pilot_gpt4_1_mini_v56`, 6,961 answered points. The method is described in `docs/05-passport-model.md` of the repository.

## The capacities

The zero point of a capacity is a text of 20 words, the start of the answer, the density at the maximum of its law and one rule; every other axis of the capacity is set by its definition.

The capacities are taken at the level 0.5. At it the share of successes over all requests and the success of the median request reach one half at the same logit, zero, whatever the variance of the request shift: a capacity is the load at which the laws bring the logit of the median request to zero, and it does not depend on that variance.

### P_sync: a word of the text stands on its place

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 1.0·10²⁹ (2.5·10²³–4.1·10³⁴), continuation of the law | at least 256, reserve left 37.23 |
| fatigue capacity, words | 0.5 | 1.2·10⁵ (85,436–1.7·10⁵), continuation of the law | - |

The instruction capacity is the number of rules at which the law of the number of rules brings the logit to zero, and next to it the walk along the bins of that axis; by the law it is none when the fall of that law over its bins does not differ from zero. The fatigue capacity is the word at which the full law of the grid at the density going to zero and one rule reaches zero, on a text of the same length: by the curve of the law, as for `H`.

### H: a rule word on its place is replaced correctly

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 24.8 (22.8–27.0) | 22.0 |
| fatigue capacity, words | 0.5 | 1.4·10⁵ (83,833–2.5·10⁵), continuation of the law | - |

### F: a misfire on a background word on its place

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 67,558 (25,844–1.8·10⁵), continuation of the law | at least 256, reserve left 8.19 |
| fatigue capacity, words | 0.5 | 3.0·10¹¹ (2.0·10⁹–4.5·10¹³), continuation of the law | - |

The instruction capacity is the number of rules at the zero point. The fatigue capacity is the number of the last word of a text of the same length at the zero point.

## The standard stress test

The standard stress test describes how a model behaves as the complexity of a task with fixed proportions of the load increases step by step. The complexity increases until the probability of an error on the last word of the text reaches 50%. The density of replacements in the test is `p = 0.5`, and the text has one rule per 50 words: with `m` rules the text has `T = 50·m` words and `D = 2m` distinct words, each of them repeated 25 times. The length of the text and the number of rules increase together, keeping these proportions.

The point of 50% is set by the combined error of the last word of the median request, `E = 1 − P_sync·H = 0.5`, where `P_sync` is the probability that the word keeps its place in the text, and `H` is the conditional probability that the replacement is made correctly on a kept place. False replacements of background words do not enter the definition of the point, but they are measured at the point found.

The influence of each load is set by its price in the logit of the probability of the correct outcome of the last word. The influence keeps its sign: a negative value means that the load lowers the probability of an error of that kind. The three shares of the influence sum to 100; when the total price of the loads is zero or less, the influence is not defined. The mean share of errors over the text is computed over all its words for the median request. Every number has a 95% confidence interval by the delta method; the parameters of the three quantities are taken independent, since the passport has no covariances between its sections.

1. The point of 50%: T = 2624 ± 190 words, 52.5 ± 3.8 rules.
2. The combined error at the point: 50%.
3. The structure of the error at the point, per 100 errors: synchronization 98.3 ± 0.5, replacements 1.7 ± 0.5; background over them, per the same 100: 2.0 ± 0.5.
4. Synchronization: errors over the text for the median request 11.4 ± 0.4%; influence per 100 errors of synchronization at the point: length 50.3 ± 2.1, density 43.8 ± 2.0, rules 5.9 ± 1.2.
5. Replacements: errors over the text for the median request 1.3 ± 0.4%; influence per 100 errors of replacements at the point: length 66.6 ± 6.3, density −1.8 ± 8.4, rules 35.2 ± 5.2.
6. Background: errors over the text for the median request 1.7 ± 0.4%; influence per 100 errors of background at the point: length 30.7 ± 5.2, density 20.7 ± 1.8, rules 48.6 ± 5.4.
7. The combined error over the text for the median request: 12.5 ± 0.5%.

## The formula

`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`. The formula is the way to combine the three quantities: it has no law and no capacity of its own. `π = m/D` is the replacement density.

## P_sync: a word of the text stands on its place

Model VS-PM, request weights by the cube V × P × M. Fit: with the shift of a request; the variance of the request shift `σ² = 18.46`; 6,908 requests, 48,872,410 trials, 8,018,035 failures, 82 parameters; removed bins: vs_1_1, vs_1_2, vs_2_2, vs_2_3, vs_3_1, vs_3_2, vs_3_3, vs_3_4, vs_4_1, vs_4_3, vs_5_2, vs_6_1, vs_6_2, vs_7_2.

The law of the grid: a length slice is a two-asymptote hyperbola, `[(κ·ℓ + λ) + μ − √((κ·ℓ + λ − μ)² + κ²·ω²)]/2` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −3.756 ± 0.222`, `lambda = 49.964 ± 3.121`, `omega = 2.524 ± 0.375`; `χ² = 58.7` over 62 cells; chosen by the smallest `QAIC` among 3 forms. The law of the plateau runs through the levels of the slices.

The law of the plateau is a slanted hyperbola, `μ + (κ/2)·[(log₂(v/20) − log₂(v/20)*) + √((log₂(v/20) − log₂(v/20)*)² + ω²)]`: `mu = 26.01 ± 1.60`, `kappa = −2.61 ± 0.36`, `corner = 4.43 ± 0.69`, `omega = 0.0010 ± 0.00`; `χ² = 4.6` over 8 bins; on the bound of the search: omega; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a slanted hyperbola, `μ + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `mu = 13.66 ± 1.10`, `kappa = −6.45 ± 0.76`, `corner = 6.61 ± 0.42`, `omega = 1.98 ± 0.52`; `χ² = 5.6` over 10 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 0.85 ± 1.21`, `b = −0.42 ± 0.09`; `χ² = 5.3` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = δ_VS + δ_P + δ_M = 40.52 ± 2.24` is the logit at the density going to zero and one rule, and that density is itself a continuation of the law.

![P_sync: S before correction](figures/P_sync/S-before-correction.png)

![P_sync: S after correction](figures/P_sync/S-after-correction.png)

![P_sync: plateau before correction](figures/P_sync/plateau-before-correction.png)

![P_sync: P before correction](figures/P_sync/P-before-correction.png)

![P_sync: P after correction](figures/P_sync/P-after-correction.png)

![P_sync: M before correction](figures/P_sync/M-before-correction.png)

![P_sync: M after correction](figures/P_sync/M-after-correction.png)

## H: a rule word on its place is replaced correctly

Model VS-PMR, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 5.64`; 6,849 requests, 3,834,716 trials, 263,563 failures, 92 parameters; removed bins: vs_1_1, vs_1_2, vs_2_1, vs_2_2, vs_2_3, vs_3_1, vs_3_2, vs_3_3, vs_3_4, vs_4_1, vs_4_2, vs_4_3.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 24 parameters more than one variance, `2·Δ ln L = 1080.6`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 234.1 | 0.08 | 2.59 ± 0.85 |
| text length, words | 531.8 | 0.08 | 2.87 ± 0.67 |
| text length, words | 1021 | 0.10 | 3.00 ± 0.68 |
| text length, words | 2059 | 0.11 | 3.13 ± 0.57 |
| text length, words | 3932 | 0.16 | 3.99 ± 0.71 |
| text length, words | 8018 | 0.21 | 3.37 ± 0.53 |
| text length, words | 1.635e+04 | 0.26 | 2.80 ± 0.49 |
| density | 0.0008488 | 0.06 | 2.17 ± 0.94 |
| density | 0.002711 | 0.05 | 2.03 ± 1.25 |
| density | 0.005546 | 0.06 | 2.23 ± 1.00 |
| density | 0.01068 | 0.04 | 1.71 ± 0.77 |
| density | 0.02017 | 0.05 | 1.92 ± 0.50 |
| density | 0.03973 | 0.05 | 1.65 ± 0.43 |
| density | 0.07611 | 0.07 | 2.00 ± 0.52 |
| density | 0.1538 | 0.12 | 1.99 ± 0.35 |
| density | 0.3054 | 0.17 | 3.41 ± 0.57 |
| density | 0.6159 | 0.19 | 4.96 ± 0.88 |
| density | 1 | 0.13 | 8.03 ± 1.73 |
| number of rules | 1 | 0.13 | 20.34 ± 11.95 |
| number of rules | 2 | 0.11 | 6.65 ± 1.44 |
| number of rules | 3.635 | 0.14 | 6.35 ± 1.15 |
| number of rules | 6.801 | 0.13 | 6.42 ± 1.23 |
| number of rules | 12.87 | 0.10 | 4.81 ± 1.03 |
| number of rules | 26.74 | 0.12 | 2.86 ± 0.60 |
| number of rules | 53.09 | 0.10 | 0.82 ± 0.18 |
| number of rules | 106.1 | 0.11 | 0.27 ± 0.07 |
| number of rules | 195.8 | 0.05 | 0.15 ± 0.05 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = 2.51`, `b = 0.057` | 10.7 over 7 groups |
| density | `a + b·log₂ p` | `a = 3.86`, `b = 0.269` | 113.0 over 11 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 1.0·10⁻³³`, `A = 14.34`, `m₀ = 3.66` | 92.9 over 9 groups |

The law of the grid: a length slice is a binary exponential, `μ − A·2^(γ·ℓ)` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `A = 0.726 ± 0.376`, `gamma = 0.199 ± 0.044`; `χ² = 8.6` over 65 cells; chosen by the smallest `QAIC` among 3 forms. The law of the levels runs through the levels of the slices.

The law of the levels of the slices is a line, `a + b·log₂(v/20)`: `a = 20.50 ± 2.48`, `b = −0.70 ± 0.15`; `χ² = 5.3` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a two-slope hyperbola, `μ + β·(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `mu = 4.42 ± 9.69`, `kappa = −1.70 ± 2.99`, `corner = 2.81 ± 6.02`, `omega = 3.98 ± 6.19`, `beta = 1.11 ± 2.61`; `χ² = 5.4` over 11 bins; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −3.37 ± 2.13`, `b = −0.56 ± 0.06`; `χ² = 5.0` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the rule number is a slanted hyperbola, `μ + (κ/2)·[(log₂ r − log₂ r*) + √((log₂ r − log₂ r*)² + ω²)]`: `mu = 5.4·10⁻⁵ ± 0.18`, `kappa = −0.76 ± 0.04`, `corner = 2.86 ± 0.19`, `omega = 0.0010 ± 0.00`; `χ² = 3.6` over 8 bins; on the bound of the search: omega; chosen by the smallest `QAIC` among 3 forms.

The level `α = μ₀ + μ_P + λ_M + λ_R = 16.68 ± 1.70`.

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

Model V-(s/v)-PM, request weights by the cube V × P × M. Fit: with the shift of a request; the variance of the request shift `σ² = 4.07`; 5,860 requests, 37,034,621 trials, 165,634 failures, 34 parameters; removed bins: v_bin_1, v_bin_2, v_bin_3, v_bin_4. The variance by the load turned bins against the median request of the data, so the quantity keeps one variance: the variance by the load turns bins against the median request: p_bin_2: passport -2.41, median request +0.73 ± 0.35; p_bin_3: passport -1.88, median request +0.55 ± 0.27; p_bin_4: passport -1.31, median request +0.34 ± 0.17; p_bin_5: passport -0.92, median request +0.25 ± 0.13; p_bin_6: passport -0.29, median request +0.20 ± 0.12.

The laws run through the bins without the poorly determined ones; a bin weighs `1/(σᵢ² + τ²)`, its own variance and the scatter of the bins around the law, and the errors are carried by the full covariance of the bins.

The law of the length is a line, `a + b·log₂(v/20)`: `a = 13.72 ± 1.21`, `b = −0.42 ± 0.12`; `χ² = 5.7` over 7 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the share of the answer is a line, `a + b·q`: `a = 0.31 ± 0.06`, `b = −0.32 ± 0.07`; `χ² = 7.2` over 9 bins; chosen by the smallest `QAIC` among 2 forms. A fall of 0.32 over the whole answer.

The law of the density is a logarithm plus a line, `a + b·log₂(p/2⁻¹⁰) + c·0.0009765625·2^(log₂(p/2⁻¹⁰))`: `a = −0.88 ± 0.36`, `b = 0.34 ± 0.07`, `c = −7.44 ± 1.01`; `χ² = 4.2` over 10 bins; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −0.08 ± 0.62`, `b = −0.91 ± 0.04`; `χ² = 4.4` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α_F = V(20) + q(0) + δ_P + M(1) = 14.67 ± 1.05`, where `δ_P` is the maximum of the law of the density, at `p = 0.0665`; `V(20)` = 13.72 ± 1.21, `q(0)` = 0.31 ± 0.06, `δ_P` = 0.72 ± 0.25, `M(1)` = −0.08 ± 0.62. The edge of the measured area: 256 rules and 20,480 words.

![F: V before correction](figures/F/V-before-correction.png)

![F: V after correction](figures/F/V-after-correction.png)

![F: answer share before correction](figures/F/answer-share-before-correction.png)

![F: answer share after correction](figures/F/answer-share-after-correction.png)

![F: P before correction](figures/F/P-before-correction.png)

![F: P after correction](figures/F/P-after-correction.png)

![F: M before correction](figures/F/M-before-correction.png)

![F: M after correction](figures/F/M-after-correction.png)
