# Passport of `gemini-3.1-flash-lite`

Measurement `pilot_gemini3_1_flash_lite_v57`, 5,336 answered points. The method is described in `docs/05-passport-model.md` of the repository.

## The capacities

The zero point of a capacity is a text of 20 words, the start of the answer, the density at the maximum of its law and one rule; every other axis of the capacity is set by its definition.

The capacities are taken at the level 0.5. At it the share of successes over all requests and the success of the median request reach one half at the same logit, zero, whatever the variance of the request shift: a capacity is the load at which the laws bring the logit of the median request to zero, and it does not depend on that variance.

### P_sync: a word of the text stands on its place

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | none: the law gives no root | at least 512, reserve left 40.26 |
| fatigue capacity, words | 0.5 | 99,376 (53,764–1.8·10⁵), continuation of the law | - |

The instruction capacity is the number of rules at which the law of the number of rules brings the logit to zero, and next to it the walk along the bins of that axis; by the law it is none when the fall of that law over its bins does not differ from zero. The fatigue capacity is the word at which the full law of the grid at the density going to zero and one rule reaches zero, on a text of the same length: by the curve of the law, as for `H`.

### H: a rule word on its place is replaced correctly

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 20.2 (16.6–24.5) | 16.4 |
| fatigue capacity, words | 0.5 | 4,772 (4,374–5,205) | - |

### F: a misfire on a background word on its place

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 26,685 (7,035–1.0·10⁵), continuation of the law | at least 512, reserve left 4.87 |
| fatigue capacity, words | 0.5 | 6.2·10¹⁹ (1.0·10⁹–3.6·10³⁰), continuation of the law | - |

The instruction capacity is the number of rules at the zero point. The fatigue capacity is the number of the last word of a text of the same length at the zero point.

## The standard stress test

The standard stress test describes how a model behaves as the complexity of a task with fixed proportions of the load increases step by step. The complexity increases until the probability of an error on the last word of the text reaches 50%. The density of replacements in the test is `p = 0.5`, and the text has one rule per 50 words: with `m` rules the text has `T = 50·m` words and `D = 2m` distinct words, each of them repeated 25 times. The length of the text and the number of rules increase together, keeping these proportions.

The point of 50% is set by the combined error of the last word of the median request, `E = 1 − P_sync·H = 0.5`, where `P_sync` is the probability that the word keeps its place in the text, and `H` is the conditional probability that the replacement is made correctly on a kept place. False replacements of background words do not enter the definition of the point, but they are measured at the point found.

The influence of each load is set by its price in the logit of the probability of the correct outcome of the last word. The influence keeps its sign: a negative value means that the load lowers the probability of an error of that kind. The three shares of the influence sum to 100; when the total price of the loads is zero or less, the influence is not defined. The mean share of errors over the text is computed over all its words for the median request. Every number has a 95% confidence interval by the delta method; the parameters of the three quantities are taken independent, since the passport has no covariances between its sections.

1. The point of 50%: T = 3083 ± 207 words, 61.7 ± 4.1 rules.
2. The combined error at the point: 50%.
3. The structure of the error at the point, per 100 errors: synchronization 82.5 ± 6.1, replacements 17.5 ± 6.1; background over them, per the same 100: 0.3 ± 0.1.
4. Synchronization: errors over the text for the median request 8.2 ± 0.7%; influence per 100 errors of synchronization at the point: length 39.1 ± 4.5, density 62.7 ± 4.6, rules −1.9 ± 1.3.
5. Replacements: errors over the text for the median request 4.2 ± 1.3%; influence per 100 errors of replacements at the point: length 106.2 ± 4.4, density −79.6 ± 22.4, rules 73.4 ± 19.6.
6. Background: errors over the text for the median request 0.5 ± 0.1%; influence per 100 errors of background at the point: length −14.5 ± 27.4, density 16.4 ± 4.2, rules 98.1 ± 24.9.
7. The combined error over the text for the median request: 11.6 ± 0.7%.

## The formula

`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`. The formula is the way to combine the three quantities: it has no law and no capacity of its own. `π = m/D` is the replacement density.

## P_sync: a word of the text stands on its place

Model VS-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 29.30`; 5,272 requests, 49,493,180 trials, 19,475,791 failures, 87 parameters; removed bins: vs_1_1, vs_1_2, vs_2_1, vs_2_2, vs_2_3, vs_3_1, vs_3_2, vs_3_3, vs_3_4, vs_4_1, vs_4_2, vs_4_4, vs_4_5, vs_5_1, vs_5_2, vs_5_4, vs_6_1, vs_6_2, vs_6_3, vs_6_4, vs_7_1, vs_7_2, vs_7_3.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 26 parameters more than one variance, `2·Δ ln L = 554.2`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 369.6 | 0.07 | 21.94 ± 10.38 |
| text length, words | 1126 | 0.07 | 15.12 ± 5.62 |
| text length, words | 2044 | 0.16 | 14.52 ± 3.21 |
| text length, words | 4026 | 0.21 | 22.06 ± 3.52 |
| text length, words | 7794 | 0.21 | 23.90 ± 3.48 |
| text length, words | 1.565e+04 | 0.17 | 45.53 ± 7.87 |
| text length, words | 3.068e+04 | 0.12 | 73.72 ± 15.93 |
| density | 0.0006355 | 0.02 | 30.49 ± 12.73 |
| density | 0.001312 | 0.04 | 31.69 ± 13.82 |
| density | 0.002646 | 0.04 | 40.23 ± 13.41 |
| density | 0.005214 | 0.06 | 36.81 ± 12.95 |
| density | 0.0101 | 0.06 | 40.42 ± 11.72 |
| density | 0.02021 | 0.06 | 31.60 ± 8.19 |
| density | 0.04013 | 0.08 | 28.79 ± 6.71 |
| density | 0.07865 | 0.11 | 28.19 ± 5.57 |
| density | 0.1563 | 0.11 | 32.08 ± 5.96 |
| density | 0.2995 | 0.11 | 20.38 ± 4.65 |
| density | 0.6067 | 0.16 | 22.95 ± 4.10 |
| density | 1 | 0.15 | 17.23 ± 3.32 |
| number of rules | 1 | 0.10 | 23.10 ± 5.06 |
| number of rules | 2 | 0.17 | 20.55 ± 3.38 |
| number of rules | 3.555 | 0.12 | 20.17 ± 4.16 |
| number of rules | 6.901 | 0.10 | 17.47 ± 3.77 |
| number of rules | 13.65 | 0.09 | 24.23 ± 5.62 |
| number of rules | 27.03 | 0.08 | 28.05 ± 5.28 |
| number of rules | 51.09 | 0.11 | 44.47 ± 9.02 |
| number of rules | 100.5 | 0.07 | 31.53 ± 6.21 |
| number of rules | 195.2 | 0.09 | 42.03 ± 13.17 |
| number of rules | 405.2 | 0.06 | 40.69 ± 10.64 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = −32.40`, `b = 4.855` | 77.8 over 7 groups |
| density | `a + b·log₂ p` | `a = 19.76`, `b = −2.228` | 15.1 over 12 groups |
| number of rules | `c + A/m` | `c = 26.54`, `A = 1.4·10⁻²³` | 79.3 over 10 groups |

The law of the grid: a length slice is a slanted hyperbola, `μ + (κ/2)·[(ℓ − ℓ*) + √((ℓ − ℓ*)² + ω²)]` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −3.730 ± 0.319`, `corner = 8.559 ± 0.201`, `omega = 0.479 ± 0.414`; `χ² = 57.6` over 66 cells; chosen by the smallest `QAIC` among 3 forms. The law of the plateau runs through the levels of the slices.

The law of the plateau is a slanted hyperbola, `μ + (κ/2)·[(log₂(v/20) − log₂(v/20)*) + √((log₂(v/20) − log₂(v/20)*)² + ω²)]`: `mu = 19.54 ± 3.86`, `kappa = −1.90 ± 0.90`, `corner = 10.61 ± 0.00`, `omega = 10.61 ± 0.00`; `χ² = 4.3` over 8 bins; on the bound of the search: corner, omega; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a slanted hyperbola, `μ + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `mu = 20.71 ± 4.54`, `kappa = −2.53 ± 0.34`, `corner = 0.00 ± 0.00`, `omega = 3.14 ± 3.41`; `χ² = 7.3` over 11 bins; on the bound of the search: corner; chosen by the smallest `QAIC` among 3 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 1.57 ± 0.91`, `b = 0.12 ± 0.08`; `χ² = 7.3` over 9 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = δ_VS + δ_P + δ_M = 41.83 ± 6.79` is the logit at the density going to zero and one rule, and that density is itself a continuation of the law.

![P_sync: S before correction](figures/P_sync/S-before-correction.png)

![P_sync: S after correction](figures/P_sync/S-after-correction.png)

![P_sync: plateau before correction](figures/P_sync/plateau-before-correction.png)

![P_sync: P before correction](figures/P_sync/P-before-correction.png)

![P_sync: P after correction](figures/P_sync/P-after-correction.png)

![P_sync: M before correction](figures/P_sync/M-before-correction.png)

![P_sync: M after correction](figures/P_sync/M-after-correction.png)

![P_sync: spread V](figures/P_sync/spread-V.png)

![P_sync: spread P](figures/P_sync/spread-P.png)

![P_sync: spread M](figures/P_sync/spread-M.png)

## H: a rule word on its place is replaced correctly

Model VS-PMR, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 5.99`; 5,176 requests, 4,320,726 trials, 1,901,662 failures, 97 parameters; removed bins: vs_1_1, vs_1_2, vs_2_1, vs_2_2, vs_2_3, vs_3_1, vs_3_2, vs_3_3, vs_3_4, vs_4_1, vs_4_2, vs_4_3, vs_4_4, vs_4_5, vs_5_1, vs_5_2, vs_5_3, vs_5_4, vs_5_6, vs_6_2, vs_6_3, vs_6_4.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 26 parameters more than one variance, `2·Δ ln L = 1223.3`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 450.1 | 0.06 | 2.58 ± 1.18 |
| text length, words | 1126 | 0.07 | 2.43 ± 0.90 |
| text length, words | 2044 | 0.16 | 2.19 ± 0.50 |
| text length, words | 4026 | 0.21 | 2.67 ± 0.42 |
| text length, words | 7794 | 0.21 | 3.68 ± 0.53 |
| text length, words | 1.566e+04 | 0.17 | 4.87 ± 1.04 |
| text length, words | 3.073e+04 | 0.12 | 3.83 ± 0.68 |
| density | 0.0006355 | 0.02 | 0.25 ± 0.12 |
| density | 0.001311 | 0.04 | 0.61 ± 0.25 |
| density | 0.002646 | 0.04 | 0.76 ± 0.28 |
| density | 0.005209 | 0.06 | 0.75 ± 0.21 |
| density | 0.01011 | 0.06 | 0.86 ± 0.24 |
| density | 0.0202 | 0.07 | 1.31 ± 0.38 |
| density | 0.0402 | 0.08 | 1.45 ± 0.61 |
| density | 0.07845 | 0.10 | 1.63 ± 0.49 |
| density | 0.1561 | 0.11 | 3.16 ± 0.90 |
| density | 0.3006 | 0.11 | 6.30 ± 1.57 |
| density | 0.6086 | 0.16 | 12.67 ± 2.12 |
| density | 1 | 0.15 | 16.19 ± 2.64 |
| number of rules | 1 | 0.10 | 29.84 ± 9.38 |
| number of rules | 2 | 0.17 | 9.38 ± 2.18 |
| number of rules | 3.545 | 0.12 | 6.17 ± 1.33 |
| number of rules | 6.881 | 0.10 | 4.43 ± 0.93 |
| number of rules | 13.62 | 0.09 | 3.45 ± 0.76 |
| number of rules | 27 | 0.09 | 2.36 ± 0.53 |
| number of rules | 51.13 | 0.12 | 1.59 ± 0.35 |
| number of rules | 100.4 | 0.07 | 0.73 ± 0.30 |
| number of rules | 194.1 | 0.09 | 0.30 ± 0.12 |
| number of rules | 406.5 | 0.06 | 0.44 ± 0.18 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = −1.64`, `b = 0.389` | 15.0 over 7 groups |
| density | `a + b·log₂ p` | `a = 6.56`, `b = 0.603` | 418.4 over 12 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 0.29`, `A = 20.37`, `m₀ = 2.36` | 53.6 over 10 groups |

The law of the grid: a length slice is a two-asymptote hyperbola, `[(κ·ℓ + λ) + μ − √((κ·ℓ + λ − μ)² + κ²·ω²)]/2` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −1.984 ± 0.081`, `lambda = 33.097 ± 1.086`, `omega = 0.127 ± 0.328`; `χ² = 74.5` over 68 cells; chosen by the smallest `QAIC` among 3 forms. The law of the levels runs through the levels of the slices.

The law of the levels of the slices is a line, `a + b·log₂(v/20)`: `a = 18.56 ± 2.12`, `b = −0.11 ± 0.27`; `χ² = 6.0` over 7 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a two-slope hyperbola, `μ + β·(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `mu = 5.14 ± 1.06`, `kappa = −7.74 ± 1.20`, `corner = 10.00 ± 0.00`, `omega = 1.88 ± 0.47`, `beta = 1.12 ± 0.10`; `χ² = 4.8` over 11 bins; on the bound of the search: corner; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −1.90 ± 0.51`, `b = −0.88 ± 0.05`; `χ² = 7.6` over 9 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the rule number is a binary exponential, `μ − A·2^(γ·log₂ r)`: `mu = −0.11 ± 0.15`, `A = 0.15 ± 0.04`, `gamma = 0.50 ± 0.05`; `χ² = 6.0` over 9 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = μ₀ + μ_P + λ_M + λ_R = 9.71 ± 2.06`.

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

Model V-(s/v)-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 5.19`; 4,469 requests, 25,717,845 trials, 91,715 failures, 37 parameters; removed bins: v_bin_1, v_bin_2, v_bin_3.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 24 parameters more than one variance, `2·Δ ln L = 1037.3`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 620.8 | 0.10 | 2.88 ± 1.12 |
| text length, words | 2067 | 0.13 | 1.82 ± 0.42 |
| text length, words | 4063 | 0.22 | 1.46 ± 0.22 |
| text length, words | 7792 | 0.23 | 2.08 ± 0.28 |
| text length, words | 1.559e+04 | 0.19 | 2.65 ± 0.42 |
| text length, words | 3.057e+04 | 0.12 | 2.56 ± 0.45 |
| density | 0.0006355 | 0.02 | 0.35 ± 0.18 |
| density | 0.001311 | 0.04 | 0.59 ± 0.22 |
| density | 0.002646 | 0.05 | 0.58 ± 0.19 |
| density | 0.005209 | 0.07 | 0.75 ± 0.24 |
| density | 0.01011 | 0.07 | 1.06 ± 0.24 |
| density | 0.0202 | 0.08 | 1.18 ± 0.26 |
| density | 0.04013 | 0.09 | 1.66 ± 0.38 |
| density | 0.07828 | 0.12 | 1.61 ± 0.30 |
| density | 0.1556 | 0.13 | 2.69 ± 0.50 |
| density | 0.2996 | 0.13 | 3.76 ± 0.80 |
| density | 0.6062 | 0.19 | 8.97 ± 1.95 |
| number of rules | 1 | 0.12 | 7.31 ± 2.10 |
| number of rules | 2 | 0.20 | 5.93 ± 1.25 |
| number of rules | 3.493 | 0.13 | 6.97 ± 1.54 |
| number of rules | 6.841 | 0.11 | 3.83 ± 0.76 |
| number of rules | 13.49 | 0.09 | 2.66 ± 0.51 |
| number of rules | 26.76 | 0.09 | 1.02 ± 0.23 |
| number of rules | 49.2 | 0.09 | 0.58 ± 0.13 |
| number of rules | 98.4 | 0.07 | 0.27 ± 0.09 |
| number of rules | 191.9 | 0.06 | 0.17 ± 0.04 |
| number of rules | 404.8 | 0.05 | 0.13 ± 0.04 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `c + A/v` | `c = 2.08`, `A = 7.9·10⁻¹⁶` | 39.0 over 6 groups |
| density | `a + b·log₂ p` | `a = 3.36`, `b = 0.297` | 141.0 over 11 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 0.02`, `A = 11.77`, `m₀ = 2.88` | 25.4 over 10 groups |

The laws run through the bins without the poorly determined ones; a bin weighs `1/(σᵢ² + τ²)`, its own variance and the scatter of the bins around the law, and the errors are carried by the full covariance of the bins.

The law of the length is a line, `a + b·log₂(v/20)`: `a = 12.08 ± 1.23`, `b = −0.20 ± 0.13`; `χ² = 39.8` over 9 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the share of the answer is an exponential, `c + A·e^(−q/τ)`: `c = 0.13 ± 0.09`, `A = −2.05 ± 0.15`, `tau = 0.23 ± 0.03`; `χ² = 5.5` over 9 bins; chosen by the smallest `QAIC` among 2 forms. A fall of −2.02 over the whole answer.

The law of the density is a logarithm plus a line, `a + b·log₂(p/2⁻¹⁰) + c·0.0009765625·2^(log₂(p/2⁻¹⁰))`: `a = −0.21 ± 0.33`, `b = 0.16 ± 0.05`, `c = −2.64 ± 0.55`; `χ² = 8.3` over 10 bins; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −0.78 ± 0.40`, `b = −0.68 ± 0.04`; `χ² = 10.3` over 9 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α_F = V(20) + q(0) + δ_P + M(1) = 10.01 ± 1.19`, where `δ_P` is the maximum of the law of the density, at `p = 0.0897`; `V(20)` = 12.08 ± 1.23, `q(0)` = −1.91 ± 0.16, `δ_P` = 0.62 ± 0.14, `M(1)` = −0.78 ± 0.40. The edge of the measured area: 512 rules and 40,960 words.

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
