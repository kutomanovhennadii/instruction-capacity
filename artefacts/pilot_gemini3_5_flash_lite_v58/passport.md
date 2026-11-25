# Passport of `gemini-3.5-flash-lite`

Measurement `pilot_gemini3_5_flash_lite_v58`, 3,788 answered points. The method is described in `docs/05-passport-model.md` of the repository.

## The capacities

The zero point of a capacity is a text of 20 words, the start of the answer, the density at the maximum of its law and one rule; every other axis of the capacity is set by its definition.

The capacities are taken at the level 0.5. At it the share of successes over all requests and the success of the median request reach one half at the same logit, zero, whatever the variance of the request shift: a capacity is the load at which the laws bring the logit of the median request to zero, and it does not depend on that variance.

### P_sync: a word of the text stands on its place

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | none: the law gives no root | at least 512, reserve left 10.19 |
| fatigue capacity, words | 0.5 | none: the law gives no root | - |

The instruction capacity is the number of rules at which the law of the number of rules brings the logit to zero, and next to it the walk along the bins of that axis; by the law it is none when the fall of that law over its bins does not differ from zero. The fatigue capacity is the word at which the full law of the grid at the density going to zero and one rule reaches zero, on a text of the same length: by the curve of the law, as for `H`.

### H: a rule word on its place is replaced correctly

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 14.3 (13.1–15.6) | 12.7 |
| fatigue capacity, words | 0.5 | 926 (771–1,113) | - |

### F: a misfire on a background word on its place

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | none: the law gives no root | at least 512, reserve left 5.99 |
| fatigue capacity, words | 0.5 | none: the law gives no root | - |

The instruction capacity is the number of rules at the zero point. The fatigue capacity is the number of the last word of a text of the same length at the zero point.

## The standard stress test

The standard stress test describes how a model behaves as the complexity of a task with fixed proportions of the load increases step by step. The complexity increases until the probability of an error on the last word of the text reaches 50%. The density of replacements in the test is `p = 0.5`, and the text has one rule per 50 words: with `m` rules the text has `T = 50·m` words and `D = 2m` distinct words, each of them repeated 25 times. The length of the text and the number of rules increase together, keeping these proportions.

The point of 50% is set by the combined error of the last word of the median request, `E = 1 − P_sync·H = 0.5`, where `P_sync` is the probability that the word keeps its place in the text, and `H` is the conditional probability that the replacement is made correctly on a kept place. False replacements of background words do not enter the definition of the point, but they are measured at the point found.

The influence of each load is set by its price in the logit of the probability of the correct outcome of the last word. The influence keeps its sign: a negative value means that the load lowers the probability of an error of that kind. The three shares of the influence sum to 100; when the total price of the loads is zero or less, the influence is not defined. The mean share of errors over the text is computed over all its words for the median request. Every number has a 95% confidence interval by the delta method; the parameters of the three quantities are taken independent, since the passport has no covariances between its sections.

1. The point of 50%: T = 550 ± 72 words, 11.0 ± 1.4 rules.
2. The combined error at the point: 50%.
3. The structure of the error at the point, per 100 errors: synchronization 3.7 ± 1.0, replacements 96.3 ± 1.0; background over them, per the same 100: 0.9 ± 0.2.
4. Synchronization: errors over the text for the median request 0.8 ± 0.2%; influence per 100 errors of synchronization at the point: length 50.1 ± 7.3, density 54.8 ± 8.3, rules −4.9 ± 4.3.
5. Replacements: errors over the text for the median request 23.4 ± 0.5%; influence per 100 errors of replacements at the point: length 88.2 ± 4.7, density −8.8 ± 6.7, rules 20.5 ± 3.4.
6. Background: errors over the text for the median request 0.5 ± 0.2%; influence per 100 errors of background at the point: length −402.7 ± 2075.5, density 509.3 ± 2101.8, rules −6.6 ± 36.2.
7. The combined error over the text for the median request: 24.0 ± 0.4%.

## The formula

`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`. The formula is the way to combine the three quantities: it has no law and no capacity of its own. `π = m/D` is the replacement density.

## P_sync: a word of the text stands on its place

Model VS-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 13.70`; 3,775 requests, 37,963,300 trials, 5,842,422 failures, 104 parameters; removed bins: vs_1_1, vs_1_2, vs_2_1, vs_2_2, vs_3_1, vs_6_1.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 28 parameters more than one variance, `2·Δ ln L = 1829.7`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 93.16 | 0.08 | 1.23 ± 1.18 |
| text length, words | 283.5 | 0.05 | 0.40 ± 0.36 |
| text length, words | 547.5 | 0.06 | 0.48 ± 0.28 |
| text length, words | 1090 | 0.08 | 2.43 ± 1.35 |
| text length, words | 2153 | 0.12 | 4.52 ± 2.09 |
| text length, words | 4283 | 0.13 | 3.84 ± 1.17 |
| text length, words | 8402 | 0.16 | 8.58 ± 2.87 |
| text length, words | 1.689e+04 | 0.17 | 15.87 ± 5.05 |
| text length, words | 3.146e+04 | 0.15 | 51.02 ± 20.01 |
| density | 0.0006999 | 0.03 | 6.29 ± 6.53 |
| density | 0.001246 | 0.03 | 10.48 ± 8.33 |
| density | 0.002461 | 0.04 | 4.52 ± 4.59 |
| density | 0.004717 | 0.04 | 1.12 ± 0.98 |
| density | 0.009694 | 0.05 | 2.50 ± 2.11 |
| density | 0.01927 | 0.05 | 4.36 ± 4.05 |
| density | 0.0375 | 0.06 | 1.15 ± 0.73 |
| density | 0.07309 | 0.07 | 3.20 ± 1.63 |
| density | 0.1475 | 0.10 | 3.49 ± 1.33 |
| density | 0.3032 | 0.19 | 3.98 ± 1.17 |
| density | 0.6632 | 0.20 | 10.84 ± 3.09 |
| density | 1 | 0.15 | 28.66 ± 7.79 |
| number of rules | 1 | 0.13 | 9.20 ± 3.69 |
| number of rules | 2 | 0.12 | 12.35 ± 5.81 |
| number of rules | 3.624 | 0.15 | 13.91 ± 6.77 |
| number of rules | 6.916 | 0.13 | 7.00 ± 2.83 |
| number of rules | 13.04 | 0.13 | 2.58 ± 0.86 |
| number of rules | 26.02 | 0.10 | 2.61 ± 1.18 |
| number of rules | 52.33 | 0.07 | 2.36 ± 1.29 |
| number of rules | 98.94 | 0.06 | 1.45 ± 0.62 |
| number of rules | 206.3 | 0.06 | 8.27 ± 6.38 |
| number of rules | 381.5 | 0.05 | 4.89 ± 2.82 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = −7.23`, `b = 1.173` | 157.6 over 9 groups |
| density | `a + b·log₂ p` | `a = 8.88`, `b = 0.639` | 165.9 over 12 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 2.19`, `A = 16.70`, `m₀ = 1.58` | 42.5 over 10 groups |

The law of the grid: a length slice is a slanted hyperbola, `μ + (κ/2)·[(ℓ − ℓ*) + √((ℓ − ℓ*)² + ω²)]` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −1.288 ± 0.227`, `corner = 7.291 ± 0.803`, `omega = 2.447 ± 1.356`; `χ² = 85.6` over 83 cells; chosen by the smallest `QAIC` among 3 forms. The law of the plateau runs through the levels of the slices.

The law of the plateau is a slanted hyperbola, `μ + (κ/2)·[(log₂(v/20) − log₂(v/20)*) + √((log₂(v/20) − log₂(v/20)*)² + ω²)]`: `mu = 6.90 ± 0.75`, `kappa = 1.28 ± 0.13`, `corner = 4.77 ± 0.36`, `omega = 0.0010 ± 0.00`; `χ² = 4.0` over 10 bins; on the bound of the search: omega; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a slanted hyperbola, `μ + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `mu = 3.38 ± 0.38`, `kappa = −0.53 ± 0.06`, `corner = 2.63 ± 0.59`, `omega = 0.0010 ± 0.00`; `χ² = 9.0` over 11 bins; on the bound of the search: omega; chosen by the smallest `QAIC` among 3 forms.

The law of the number of rules is a binary exponential, `μ − A·2^(γ·log₂ m)`: `mu = −0.46 ± 1.14`, `A = −0.37 ± 0.93`, `gamma = 0.25 ± 0.33`; `χ² = 5.5` over 9 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = δ_VS + δ_P + δ_M = 10.19 ± 0.79` is the logit at the density going to zero and one rule, and that density is itself a continuation of the law.

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

Model VS-PMR, request weights by the cube V × P × M. Fit: with the shift of a request; the variance of the request shift `σ² = 3.01`; 3,769 requests, 7,895,143 trials, 6,894,664 failures, 116 parameters; removed bins: vs_1_1, vs_2_1, vs_4_1. The variance by the load turned bins against the median request of the data, so the quantity keeps one variance: the variance by the load turns bins against the median request: p_bin_2: passport +0.77, median request -3.27 ± 1.99; p_bin_3: passport +0.70, median request -1.92 ± 1.53; p_bin_4: passport +0.86, median request -1.95 ± 0.75; p_bin_5: passport +0.46, median request -2.46 ± 0.78; p_bin_6: passport +0.62, median request -1.43 ± 0.41; p_bin_7: passport +0.42, median request -1.30 ± 0.43; p_bin_8: passport +0.32, median request -0.97 ± 0.42.

The law of the grid: a length slice is a two-asymptote hyperbola, `[(κ·ℓ + λ) + μ − √((κ·ℓ + λ − μ)² + κ²·ω²)]/2` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −1.402 ± 0.073`, `lambda = 16.078 ± 1.094`, `omega = 3.780 ± 0.680`; `χ² = 91.3` over 87 cells; chosen by the smallest `QAIC` among 3 forms. The law of the levels runs through the levels of the slices.

The law of the levels of the slices is a binary exponential, `μ − A·2^(γ·log₂(v/20))`: `mu = 8.93 ± 0.74`, `A = 0.00016 ± 0.00095`, `gamma = 1.31 ± 0.78`; `χ² = 11.7` over 12 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a two-slope hyperbola, `μ + β·(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `mu = −0.31 ± 0.27`, `kappa = 1.28 ± 0.40`, `corner = 8.44 ± 0.42`, `omega = 0.0010 ± 0.00`, `beta = −0.0059 ± 0.06`; `χ² = 7.3` over 11 bins; on the bound of the search: omega; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −0.82 ± 0.39`, `b = −0.33 ± 0.05`; `χ² = 6.9` over 9 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the rule number is a line, `a + b·log₂ r`: `a = −0.26 ± 0.19`, `b = −0.18 ± 0.01`; `χ² = 6.7` over 9 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = μ₀ + μ_P + λ_M + λ_R = 7.60 ± 0.79`.

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

Model V-(s/v)-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 1.46`; 3,200 requests, 24,229,266 trials, 37,043 failures, 39 parameters; removed bins: v_bin_1.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 26 parameters more than one variance, `2·Δ ln L = 283.6`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 95.32 | 0.07 | 0.10 ± 0.31 |
| text length, words | 282.6 | 0.05 | 0.55 ± 0.63 |
| text length, words | 544.1 | 0.06 | 0.48 ± 0.36 |
| text length, words | 1095 | 0.08 | 0.75 ± 0.45 |
| text length, words | 2172 | 0.11 | 1.29 ± 0.47 |
| text length, words | 4276 | 0.13 | 1.53 ± 0.48 |
| text length, words | 8375 | 0.17 | 1.83 ± 0.53 |
| text length, words | 1.712e+04 | 0.17 | 1.92 ± 0.42 |
| text length, words | 3.156e+04 | 0.15 | 2.53 ± 0.75 |
| density | 0.000957 | 0.07 | 0.38 ± 0.37 |
| density | 0.002461 | 0.04 | 0.50 ± 0.47 |
| density | 0.004723 | 0.05 | 0.79 ± 0.58 |
| density | 0.009694 | 0.06 | 0.89 ± 0.59 |
| density | 0.01936 | 0.06 | 0.46 ± 0.31 |
| density | 0.03753 | 0.07 | 0.59 ± 0.35 |
| density | 0.07311 | 0.08 | 0.62 ± 0.33 |
| density | 0.147 | 0.11 | 1.20 ± 0.48 |
| density | 0.3034 | 0.22 | 1.53 ± 0.52 |
| density | 0.6624 | 0.23 | 3.49 ± 1.28 |
| number of rules | 1 | 0.15 | 1.59 ± 0.91 |
| number of rules | 2 | 0.14 | 1.24 ± 0.63 |
| number of rules | 3.556 | 0.15 | 2.30 ± 0.73 |
| number of rules | 6.768 | 0.13 | 1.31 ± 0.50 |
| number of rules | 12.82 | 0.13 | 1.14 ± 0.49 |
| number of rules | 25.67 | 0.09 | 0.73 ± 0.42 |
| number of rules | 51.68 | 0.07 | 0.80 ± 0.39 |
| number of rules | 97.09 | 0.06 | 0.67 ± 0.36 |
| number of rules | 204.7 | 0.04 | 0.32 ± 0.27 |
| number of rules | 372.5 | 0.04 | 0.93 ± 0.70 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = −1.66`, `b = 0.262` | 3.0 over 9 groups |
| density | `a + b·log₂ p` | `a = 1.79`, `b = 0.152` | 30.6 over 10 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 0.53`, `A = 1.58`, `m₀ = 8.64` | 11.5 over 10 groups |

The laws run through the bins without the poorly determined ones; a bin weighs `1/(σᵢ² + τ²)`, its own variance and the scatter of the bins around the law, and the errors are carried by the full covariance of the bins.

The law of the length is a binary exponential, `μ − A·2^(γ·log₂(v/20))`: `mu = 5.22 ± 0.61`, `A = −0.12 ± 0.14`, `gamma = 0.45 ± 0.14`; `χ² = 6.2` over 10 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the share of the answer is an exponential, `c + A·e^(−q/τ)`: `c = 0.08 ± 0.06`, `A = −2.59 ± 3.01`, `tau = 0.04 ± 0.04`; `χ² = 0.8` over 9 bins; chosen by the smallest `QAIC` among 2 forms. A fall of −2.59 over the whole answer.

The law of the density is a binary exponential, `μ − A·2^(γ·log₂(p/2⁻¹⁰))`: `mu = 4.24 ± 1.44`, `A = 1.09 ± 0.97`, `gamma = 0.24 ± 0.10`; `χ² = 5.6` over 10 bins; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 0.16 ± 0.39`, `b = 0.01 ± 0.05`; `χ² = 6.4` over 9 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α_F = V(20) + q(0) + δ_P + M(1) = 6.15 ± 3.14`, where `δ_P` is the maximum of the law of the density, at `p = 0.0010`; `V(20)` = 5.34 ± 0.52, `q(0)` = −2.50 ± 3.00, `δ_P` = 3.15 ± 0.56, `M(1)` = 0.16 ± 0.39. The edge of the measured area: 512 rules and 40,960 words.

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
