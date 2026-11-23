# Passport of `gemini-3.1-flash-lite`

Measurement `pilot_gemini3_1_flash_lite_v57`, 5,336 answered points. The method is described in `docs/05-passport-model.md` of the repository.

## The capacities

The zero point of a capacity is a text of 20 words, the start of the answer, the density at the maximum of its law and one rule; every other axis of the capacity is set by its definition.

The capacities are taken at the level 0.5. At it the share of successes over all requests and the success of the median request reach one half at the same logit, zero, whatever the variance of the request shift: a capacity is the load at which the laws bring the logit of the median request to zero, and it does not depend on that variance.

### P_sync: the answer word stands against its reference word

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | none: the law gives no root | at least 512, reserve left 43.95 |
| fatigue capacity, words | 0.5 | 61,997 (33,785–1.1·10⁵), continuation of the law | - |

The instruction capacity is the number of rules at which the law of the number of rules brings the logit to zero, and next to it the walk along the bins of that axis; by the law it is none when the fall of that law over its bins does not differ from zero. The fatigue capacity is the word at which the full law of the grid at the density going to zero and one rule reaches zero, on a text of the same length: by the curve of the law, as for `H`.

### H: a rule word on its place is replaced correctly

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 18.8 (15.3–23.2) | 14.8 |
| fatigue capacity, words | 0.5 | 4,524 (4,150–4,931) | - |

### F: a misfire on a background word on its place

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 27,292 (7,014–1.1·10⁵), continuation of the law | at least 512, reserve left 4.96 |
| fatigue capacity, words | 0.5 | 1.6·10¹⁹ (9.0·10⁸–3.0·10²⁹), continuation of the law | - |

The instruction capacity is the number of rules at the zero point. The fatigue capacity is the number of the last word of a text of the same length at the zero point.

## The standard stress test

The standard stress test describes how a model behaves as the complexity of a task with fixed proportions of the load increases step by step. The complexity increases until the probability of an error on the last word of the text reaches 50%. The density of replacements in the test is `p = 0.5`, and the text has one rule per 50 words: with `m` rules the text has `T = 50·m` words and `D = 2m` distinct words, each of them repeated 25 times. The length of the text and the number of rules increase together, keeping these proportions.

The point of 50% is set by the combined error of the last word of the median request, `E = 1 − P_sync·H = 0.5`, where `P_sync` is the probability that the word keeps its place in the text, and `H` is the conditional probability that the replacement is made correctly on a kept place. False replacements of background words do not enter the definition of the point, but they are measured at the point found.

The influence of each load is set by its price in the logit of the probability of the correct outcome of the last word. The influence keeps its sign: a negative value means that the load lowers the probability of an error of that kind. The three shares of the influence sum to 100; when the total price of the loads is zero or less, the influence is not defined. The mean share of errors over the text is computed over all its words for the median request. Every number has a 95% confidence interval by the delta method; the parameters of the three quantities are taken independent, since the passport has no covariances between its sections.

1. The point of 50%: T = 3189 ± 176 words, 63.8 ± 3.5 rules.
2. The combined error at the point: 50%.
3. The structure of the error at the point, per 100 errors: synchronization 76.3 ± 7.3, replacements 23.7 ± 7.3; background over them, per the same 100: 0.3 ± 0.1.
4. Synchronization: errors over the text for the median request 7.8 ± 1.0%; influence per 100 errors of synchronization at the point: length 48.5 ± 10.1, density 51.5 ± 9.9, rules 0.0 ± 1.2.
5. Replacements: errors over the text for the median request 5.5 ± 1.5%; influence per 100 errors of replacements at the point: length 105.5 ± 4.4, density −81.3 ± 23.8, rules 75.8 ± 20.9.
6. Background: errors over the text for the median request 0.5 ± 0.1%; influence per 100 errors of background at the point: length −12.9 ± 27.7, density 14.0 ± 4.0, rules 98.9 ± 25.6.
7. The combined error over the text for the median request: 12.3 ± 0.7%.

## The formula

`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`. The formula is the way to combine the three quantities: it has no law and no capacity of its own. `π = m/D` is the replacement density.

## P_sync: the answer word stands against its reference word

Model VS-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 29.93`; 5,272 requests, 112,208,407 trials, 82,197,510 failures, 88 parameters; removed bins: vs_1_1, vs_1_2, vs_2_1, vs_2_2, vs_2_3, vs_3_1, vs_3_2, vs_3_3, vs_3_4, vs_4_1, vs_4_2, vs_4_4, vs_4_5, vs_5_1, vs_5_2, vs_5_4, vs_6_1, vs_6_2, vs_6_3, vs_6_4, vs_7_1, vs_7_2.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 26 parameters more than one variance, `2·Δ ln L = 185.7`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 369.6 | 0.07 | 17.24 ± 12.73 |
| text length, words | 1126 | 0.07 | 27.71 ± 11.67 |
| text length, words | 2044 | 0.16 | 25.51 ± 5.95 |
| text length, words | 4026 | 0.21 | 26.59 ± 4.60 |
| text length, words | 7794 | 0.21 | 28.27 ± 4.07 |
| text length, words | 1.565e+04 | 0.17 | 43.35 ± 6.94 |
| text length, words | 3.068e+04 | 0.12 | 50.32 ± 9.68 |
| density | 0.0006355 | 0.02 | 34.61 ± 12.12 |
| density | 0.001312 | 0.04 | 33.75 ± 11.68 |
| density | 0.002646 | 0.04 | 39.82 ± 10.78 |
| density | 0.005214 | 0.06 | 31.95 ± 8.94 |
| density | 0.0101 | 0.06 | 33.55 ± 8.75 |
| density | 0.02021 | 0.06 | 31.01 ± 8.35 |
| density | 0.04013 | 0.08 | 31.73 ± 6.81 |
| density | 0.07865 | 0.11 | 31.71 ± 5.95 |
| density | 0.1563 | 0.11 | 36.34 ± 7.26 |
| density | 0.2995 | 0.11 | 20.01 ± 3.98 |
| density | 0.6067 | 0.16 | 24.09 ± 5.76 |
| density | 1 | 0.15 | 37.50 ± 8.20 |
| number of rules | 1 | 0.10 | 30.06 ± 6.90 |
| number of rules | 2 | 0.17 | 28.11 ± 4.95 |
| number of rules | 3.555 | 0.12 | 29.61 ± 5.52 |
| number of rules | 6.901 | 0.10 | 29.86 ± 7.44 |
| number of rules | 13.65 | 0.09 | 31.95 ± 6.31 |
| number of rules | 27.03 | 0.08 | 34.72 ± 7.79 |
| number of rules | 51.09 | 0.11 | 35.17 ± 7.10 |
| number of rules | 100.5 | 0.07 | 28.46 ± 5.41 |
| number of rules | 195.2 | 0.09 | 31.12 ± 8.75 |
| number of rules | 405.2 | 0.06 | 27.09 ± 7.09 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = −26.47`, `b = 4.656` | 14.2 over 7 groups |
| density | `a + b·log₂ p` | `a = 27.83`, `b = −0.783` | 28.0 over 12 groups |
| number of rules | `c + A/m` | `c = 30.47`, `A = 8.3·10⁻²⁹` | 5.7 over 10 groups |

The law of the grid: a length slice is a two-asymptote hyperbola, `[(κ·ℓ + λ) + μ − √((κ·ℓ + λ − μ)² + κ²·ω²)]/2` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −4.451 ± 0.353`, `lambda = 55.547 ± 4.397`, `omega = 3.106 ± 0.521`; `χ² = 60.3` over 61 cells; chosen by the smallest `QAIC` among 3 forms. The law of the plateau runs through the levels of the slices.

The law of the plateau is a binary exponential, `μ − A·2^(γ·log₂(v/20))`: `mu = 22.82 ± 5.92`, `A = 0.52 ± 0.83`, `gamma = 0.54 ± 0.19`; `χ² = 8.7` over 7 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a slanted hyperbola, `μ + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `mu = 21.13 ± 8.70`, `kappa = −2.60 ± 0.51`, `corner = 0.59 ± 1.20`, `omega = 4.30 ± 5.39`; `χ² = 6.5` over 11 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 2.18 ± 0.74`, `b = −0.00022 ± 0.09`; `χ² = 8.2` over 9 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = δ_VS + δ_P + δ_M = 46.13 ± 11.28` is the logit at the density going to zero and one rule, and that density is itself a continuation of the law.

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

Model VS-PMR, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 5.97`; 5,176 requests, 4,318,887 trials, 1,917,294 failures, 97 parameters; removed bins: vs_1_1, vs_1_2, vs_2_1, vs_2_2, vs_2_3, vs_3_1, vs_3_2, vs_3_3, vs_3_4, vs_4_1, vs_4_2, vs_4_3, vs_4_4, vs_4_5, vs_5_1, vs_5_2, vs_5_3, vs_5_4, vs_5_6, vs_6_2, vs_6_3, vs_6_4.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 26 parameters more than one variance, `2·Δ ln L = 954.6`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 450.1 | 0.06 | 3.28 ± 1.64 |
| text length, words | 1126 | 0.07 | 2.49 ± 0.97 |
| text length, words | 2044 | 0.16 | 2.28 ± 0.55 |
| text length, words | 4026 | 0.21 | 3.04 ± 0.52 |
| text length, words | 7794 | 0.21 | 4.01 ± 0.61 |
| text length, words | 1.566e+04 | 0.17 | 5.85 ± 1.25 |
| text length, words | 3.073e+04 | 0.12 | 4.94 ± 1.19 |
| density | 0.0006355 | 0.02 | 0.33 ± 0.15 |
| density | 0.001311 | 0.04 | 0.76 ± 0.31 |
| density | 0.002646 | 0.04 | 0.96 ± 0.38 |
| density | 0.005209 | 0.06 | 0.92 ± 0.27 |
| density | 0.01011 | 0.06 | 0.97 ± 0.28 |
| density | 0.0202 | 0.07 | 1.36 ± 0.42 |
| density | 0.0402 | 0.08 | 1.46 ± 0.62 |
| density | 0.07845 | 0.10 | 1.90 ± 0.67 |
| density | 0.1561 | 0.11 | 4.78 ± 1.44 |
| density | 0.3006 | 0.11 | 8.83 ± 2.26 |
| density | 0.6086 | 0.16 | 13.19 ± 2.27 |
| density | 1 | 0.15 | 13.78 ± 2.40 |
| number of rules | 1 | 0.10 | 24.38 ± 7.16 |
| number of rules | 2 | 0.17 | 8.90 ± 2.09 |
| number of rules | 3.545 | 0.12 | 6.21 ± 1.38 |
| number of rules | 6.881 | 0.10 | 4.35 ± 0.94 |
| number of rules | 13.62 | 0.09 | 3.46 ± 0.77 |
| number of rules | 27 | 0.09 | 2.44 ± 0.55 |
| number of rules | 51.13 | 0.12 | 1.97 ± 0.47 |
| number of rules | 100.4 | 0.07 | 0.92 ± 0.35 |
| number of rules | 194.1 | 0.09 | 0.86 ± 0.41 |
| number of rules | 406.5 | 0.06 | 0.67 ± 0.30 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = −2.77`, `b = 0.520` | 17.9 over 7 groups |
| density | `a + b·log₂ p` | `a = 7.22`, `b = 0.660` | 305.2 over 12 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 0.98`, `A = 38.17`, `m₀ = 0.74` | 30.2 over 10 groups |

The law of the grid: a length slice is a two-asymptote hyperbola, `[(κ·ℓ + λ) + μ − √((κ·ℓ + λ − μ)² + κ²·ω²)]/2` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −2.002 ± 0.082`, `lambda = 33.247 ± 1.129`, `omega = 0.264 ± 0.267`; `χ² = 80.6` over 68 cells; chosen by the smallest `QAIC` among 3 forms. The law of the levels runs through the levels of the slices.

The law of the levels of the slices is a line, `a + b·log₂(v/20)`: `a = 18.16 ± 2.15`, `b = −0.02 ± 0.27`; `χ² = 6.4` over 7 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a two-slope hyperbola, `μ + β·(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `mu = 5.31 ± 1.16`, `kappa = −8.37 ± 1.56`, `corner = 10.00 ± 0.00`, `omega = 1.83 ± 0.51`, `beta = 1.13 ± 0.10`; `χ² = 6.1` over 11 bins; on the bound of the search: corner; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −1.96 ± 0.54`, `b = −0.88 ± 0.05`; `χ² = 7.5` over 9 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the rule number is a binary exponential, `μ − A·2^(γ·log₂ r)`: `mu = −0.10 ± 0.17`, `A = 0.16 ± 0.05`, `gamma = 0.48 ± 0.05`; `χ² = 5.6` over 9 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = μ₀ + μ_P + λ_M + λ_R = 9.24 ± 2.07`.

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

Model V-(s/v)-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 5.20`; 4,469 requests, 25,706,292 trials, 89,189 failures, 37 parameters; removed bins: v_bin_1, v_bin_2, v_bin_3.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 24 parameters more than one variance, `2·Δ ln L = 961.1`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 620.8 | 0.10 | 2.91 ± 1.09 |
| text length, words | 2067 | 0.13 | 1.80 ± 0.42 |
| text length, words | 4063 | 0.22 | 1.49 ± 0.23 |
| text length, words | 7792 | 0.23 | 2.17 ± 0.30 |
| text length, words | 1.559e+04 | 0.19 | 2.81 ± 0.45 |
| text length, words | 3.057e+04 | 0.12 | 2.74 ± 0.52 |
| density | 0.0006355 | 0.02 | 0.36 ± 0.19 |
| density | 0.001311 | 0.04 | 0.61 ± 0.23 |
| density | 0.002646 | 0.05 | 0.59 ± 0.20 |
| density | 0.005209 | 0.07 | 0.77 ± 0.24 |
| density | 0.01011 | 0.07 | 1.09 ± 0.26 |
| density | 0.0202 | 0.08 | 1.19 ± 0.26 |
| density | 0.04013 | 0.09 | 1.64 ± 0.39 |
| density | 0.07828 | 0.12 | 1.62 ± 0.32 |
| density | 0.1556 | 0.13 | 2.91 ± 0.57 |
| density | 0.2996 | 0.13 | 4.14 ± 0.89 |
| density | 0.6062 | 0.19 | 9.21 ± 2.01 |
| number of rules | 1 | 0.12 | 7.22 ± 2.21 |
| number of rules | 2 | 0.20 | 5.94 ± 1.29 |
| number of rules | 3.493 | 0.13 | 7.16 ± 1.60 |
| number of rules | 6.841 | 0.11 | 3.91 ± 0.80 |
| number of rules | 13.49 | 0.09 | 2.67 ± 0.51 |
| number of rules | 26.76 | 0.09 | 1.03 ± 0.23 |
| number of rules | 49.2 | 0.09 | 0.61 ± 0.14 |
| number of rules | 98.4 | 0.07 | 0.27 ± 0.09 |
| number of rules | 191.9 | 0.06 | 0.21 ± 0.06 |
| number of rules | 404.8 | 0.05 | 0.17 ± 0.06 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `c + A/v` | `c = 2.17`, `A = 2.3·10⁻²³` | 43.6 over 6 groups |
| density | `a + b·log₂ p` | `a = 3.50`, `b = 0.310` | 146.9 over 11 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 0.05`, `A = 11.80`, `m₀ = 2.91` | 26.0 over 10 groups |

The laws run through the bins without the poorly determined ones; a bin weighs `1/(σᵢ² + τ²)`, its own variance and the scatter of the bins around the law, and the errors are carried by the full covariance of the bins.

The law of the length is a line, `a + b·log₂(v/20)`: `a = 12.44 ± 1.27`, `b = −0.21 ± 0.14`; `χ² = 39.6` over 9 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the share of the answer is an exponential, `c + A·e^(−q/τ)`: `c = 0.09 ± 0.09`, `A = −2.08 ± 0.15`, `tau = 0.24 ± 0.03`; `χ² = 5.2` over 9 bins; chosen by the smallest `QAIC` among 2 forms. A fall of −2.05 over the whole answer.

The law of the density is a logarithm plus a line, `a + b·log₂(p/2⁻¹⁰) + c·0.0009765625·2^(log₂(p/2⁻¹⁰))`: `a = −0.39 ± 0.34`, `b = 0.18 ± 0.05`, `c = −2.54 ± 0.57`; `χ² = 8.8` over 10 bins; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −0.80 ± 0.40`, `b = −0.69 ± 0.04`; `χ² = 10.1` over 9 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α_F = V(20) + q(0) + δ_P + M(1) = 10.22 ± 1.23`, where `δ_P` is the maximum of the law of the density, at `p = 0.1047`; `V(20)` = 12.44 ± 1.27, `q(0)` = −2.00 ± 0.17, `δ_P` = 0.58 ± 0.14, `M(1)` = −0.80 ± 0.40. The edge of the measured area: 512 rules and 40,960 words.

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
