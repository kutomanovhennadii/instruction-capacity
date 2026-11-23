# Passport of `nvidia/nemotron-3.5-lightning:free`

Measurement `pilot_nemotron3_5_lightning_v43`, 4,753 answered points. The method is described in `docs/05-passport-model.md` of the repository.

## The capacities

The zero point of a capacity is a text of 20 words, the start of the answer, the density at the maximum of its law and one rule; every other axis of the capacity is set by its definition.

The capacities are taken at the level 0.5. At it the share of successes over all requests and the success of the median request reach one half at the same logit, zero, whatever the variance of the request shift: a capacity is the load at which the laws bring the logit of the median request to zero, and it does not depend on that variance.

### P_sync: the answer word stands against its reference word

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | none: the law gives no root | at least 32, reserve left 9.65 |
| fatigue capacity, words | 0.5 | 8,758 (6,532–11,743) | - |

The instruction capacity is the number of rules at which the law of the number of rules brings the logit to zero, and next to it the walk along the bins of that axis; by the law it is none when the fall of that law over its bins does not differ from zero. The fatigue capacity is the word at which the full law of the grid at the density going to zero and one rule reaches zero, on a text of the same length: by the curve of the law, as for `H`.

### H: a rule word on its place is replaced correctly

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 10.0 (8.0–12.6) | 7.2 |
| fatigue capacity, words | 0.5 | 197 (145–266) | - |

### F: a misfire on a background word on its place

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | none: the law gives no root | at least 32, reserve left 8.24 |
| fatigue capacity, words | 0.5 | none: the law gives no root | - |

The instruction capacity is the number of rules at the zero point. The fatigue capacity is the number of the last word of a text of the same length at the zero point.

## The standard stress test

The standard stress test describes how a model behaves as the complexity of a task with fixed proportions of the load increases step by step. The complexity increases until the probability of an error on the last word of the text reaches 50%. The density of replacements in the test is `p = 0.5`, and the text has one rule per 50 words: with `m` rules the text has `T = 50·m` words and `D = 2m` distinct words, each of them repeated 25 times. The length of the text and the number of rules increase together, keeping these proportions.

The point of 50% is set by the combined error of the last word of the median request, `E = 1 − P_sync·H = 0.5`, where `P_sync` is the probability that the word keeps its place in the text, and `H` is the conditional probability that the replacement is made correctly on a kept place. False replacements of background words do not enter the definition of the point, but they are measured at the point found.

The influence of each load is set by its price in the logit of the probability of the correct outcome of the last word. The influence keeps its sign: a negative value means that the load lowers the probability of an error of that kind. The three shares of the influence sum to 100; when the total price of the loads is zero or less, the influence is not defined. The mean share of errors over the text is computed over all its words for the median request. Every number has a 95% confidence interval by the delta method; the parameters of the three quantities are taken independent, since the passport has no covariances between its sections.

1. The point of 50%: T = 145 ± 19 words, 2.9 ± 0.4 rules.
2. The combined error at the point: 50%.
3. The structure of the error at the point, per 100 errors: synchronization 50.4 ± 9.4, replacements 49.6 ± 9.4; background over them, per the same 100: 8.2 ± 1.8.
4. Synchronization: errors over the text for the median request 15.1 ± 3.1%; influence per 100 errors of synchronization at the point: length 24.4 ± 4.8, density 78.5 ± 5.4, rules −2.8 ± 1.6.
5. Replacements: errors over the text for the median request 23.0 ± 2.8%; influence per 100 errors of replacements at the point: length 110.1 ± 15.3, density −17.5 ± 16.8, rules 7.5 ± 4.0.
6. Background: errors over the text for the median request 4.5 ± 0.9%; influence per 100 errors of background at the point: length 7.7 ± 2.6, density 99.6 ± 3.4, rules −7.3 ± 2.2.
7. The combined error over the text for the median request: 34.2 ± 1.1%.

## The formula

`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`. The formula is the way to combine the three quantities: it has no law and no capacity of its own. `π = m/D` is the replacement density.

## P_sync: the answer word stands against its reference word

Model VS-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 12.56`; 4,753 requests, 48,574,151 trials, 39,111,772 failures, 92 parameters; removed bins: none.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 25 parameters more than one variance, `2·Δ ln L = 757.5`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 20 | 0.01 | 1.22 ± 0.76 |
| text length, words | 40 | 0.02 | 1.56 ± 0.93 |
| text length, words | 74.78 | 0.04 | 2.07 ± 0.90 |
| text length, words | 138.3 | 0.07 | 3.54 ± 1.16 |
| text length, words | 265.6 | 0.10 | 4.01 ± 1.17 |
| text length, words | 501.8 | 0.12 | 8.58 ± 3.33 |
| text length, words | 979.7 | 0.14 | 12.17 ± 3.61 |
| text length, words | 1925 | 0.15 | 25.15 ± 5.14 |
| text length, words | 3749 | 0.15 | 32.71 ± 6.77 |
| text length, words | 7346 | 0.12 | 26.48 ± 5.59 |
| text length, words | 1.427e+04 | 0.08 | 18.38 ± 5.74 |
| density | 0.001311 | 0.04 | 6.90 ± 3.01 |
| density | 0.002676 | 0.08 | 5.41 ± 2.01 |
| density | 0.005578 | 0.06 | 4.52 ± 1.26 |
| density | 0.01081 | 0.10 | 8.74 ± 2.74 |
| density | 0.0209 | 0.07 | 6.12 ± 1.44 |
| density | 0.04125 | 0.10 | 8.95 ± 3.14 |
| density | 0.07959 | 0.11 | 9.51 ± 1.94 |
| density | 0.1528 | 0.12 | 11.38 ± 2.26 |
| density | 0.3073 | 0.16 | 17.62 ± 3.52 |
| density | 0.6038 | 0.12 | 39.04 ± 11.45 |
| density | 1 | 0.05 | 75.33 ± 29.28 |
| number of rules | 1 | 0.23 | 24.48 ± 4.38 |
| number of rules | 2 | 0.15 | 11.36 ± 1.86 |
| number of rules | 3.518 | 0.17 | 12.18 ± 2.16 |
| number of rules | 6.575 | 0.17 | 9.67 ± 2.28 |
| number of rules | 12.96 | 0.15 | 8.38 ± 1.64 |
| number of rules | 25.23 | 0.13 | 8.29 ± 2.01 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = −9.68`, `b = 2.392` | 134.0 over 11 groups |
| density | `a + b·log₂ p` | `a = 18.42`, `b = 1.581` | 127.6 over 11 groups |
| number of rules | `c + A/m` | `c = 7.34`, `A = 13.77` | 10.8 over 6 groups |

The law of the grid: a length slice is a two-asymptote hyperbola, `[(κ·ℓ + λ) + μ − √((κ·ℓ + λ − μ)² + κ²·ω²)]/2` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −1.669 ± 0.169`, `lambda = 16.553 ± 2.133`, `omega = 2.850 ± 0.708`; `χ² = 67.2` over 77 cells; chosen by the smallest `QAIC` among 3 forms. The law of the plateau runs through the levels of the slices.

The law of the plateau is a slanted hyperbola, `μ + (κ/2)·[(log₂(v/20) − log₂(v/20)*) + √((log₂(v/20) − log₂(v/20)*)² + ω²)]`: `mu = 4.39 ± 0.68`, `kappa = −5.28 ± 2.09`, `corner = 8.18 ± 0.49`, `omega = 0.0010 ± 0.00`; `χ² = 7.9` over 11 bins; on the bound of the search: omega; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a slanted hyperbola, `μ + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `mu = 5.26 ± 1.13`, `kappa = −2.61 ± 0.71`, `corner = 6.57 ± 0.90`, `omega = 2.09 ± 1.10`; `χ² = 6.1` over 10 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 0.82 ± 0.43`, `b = 0.18 ± 0.11`; `χ² = 2.9` over 5 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = δ_VS + δ_P + δ_M = 10.48 ± 1.26` is the logit at the density going to zero and one rule, and that density is itself a continuation of the law.

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

Model VS-PMR, request weights by the cube V × P × M. Fit: with the shift of a request; the variance of the request shift `σ² = 1.38`; 4,680 requests, 590,039 trials, 448,568 failures, 97 parameters; removed bins: none. The variance by the load turned bins against the median request of the data, so the quantity keeps one variance: the variance by the load turns bins against the median request: p_bin_5: passport +0.22, median request -1.22 ± 1.01.

The law of the grid: a length slice is a slanted hyperbola, `μ + (κ/2)·[(ℓ − ℓ*) + √((ℓ − ℓ*)² + ω²)]` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −1.166 ± 0.193`, `corner = 7.422 ± 0.871`, `omega = 4.247 ± 0.783`; `χ² = 106.6` over 77 cells; chosen by the smallest `QAIC` among 3 forms. The law of the levels runs through the levels of the slices.

The law of the levels of the slices is a line, `a + b·log₂(v/20)`: `a = 6.01 ± 0.46`, `b = −0.33 ± 0.04`; `χ² = 6.6` over 11 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a two-slope hyperbola, `μ + β·(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `mu = 0.94 ± 0.40`, `kappa = −0.85 ± 0.19`, `corner = 6.51 ± 0.35`, `omega = 0.67 ± 0.98`, `beta = 0.29 ± 0.10`; `χ² = 4.7` over 10 bins; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −1.26 ± 0.25`, `b = −0.07 ± 0.05`; `χ² = 4.4` over 5 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the rule number is a binary exponential, `μ − A·2^(γ·log₂ r)`: `mu = 0.46 ± 1.16`, `A = 0.54 ± 0.99`, `gamma = 0.33 ± 0.35`; `χ² = 0.2` over 5 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = μ₀ + μ_P + λ_M + λ_R = 3.67 ± 0.61`.

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

Model V-(s/v)-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 2.92`; 4,470 requests, 8,872,340 trials, 29,064 failures, 34 parameters; removed bins: none.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 23 parameters more than one variance, `2·Δ ln L = 222.8`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 32.78 | 0.02 | 1.49 ± 1.10 |
| text length, words | 75.29 | 0.03 | 1.28 ± 0.58 |
| text length, words | 138.2 | 0.06 | 1.57 ± 0.57 |
| text length, words | 267.3 | 0.10 | 1.47 ± 0.40 |
| text length, words | 505.3 | 0.12 | 1.50 ± 0.31 |
| text length, words | 979.7 | 0.15 | 2.34 ± 0.59 |
| text length, words | 1925 | 0.16 | 2.78 ± 0.54 |
| text length, words | 3747 | 0.16 | 3.92 ± 0.74 |
| text length, words | 7328 | 0.12 | 4.95 ± 1.26 |
| text length, words | 1.416e+04 | 0.08 | 3.45 ± 2.11 |
| density | 0.001314 | 0.04 | 0.28 ± 0.19 |
| density | 0.002701 | 0.08 | 1.06 ± 0.42 |
| density | 0.005578 | 0.06 | 2.02 ± 0.85 |
| density | 0.01081 | 0.11 | 2.77 ± 1.16 |
| density | 0.0209 | 0.07 | 2.46 ± 0.71 |
| density | 0.04125 | 0.10 | 2.76 ± 0.59 |
| density | 0.07963 | 0.12 | 2.73 ± 0.55 |
| density | 0.1528 | 0.13 | 3.13 ± 0.54 |
| density | 0.3073 | 0.16 | 3.67 ± 0.70 |
| density | 0.6008 | 0.12 | 4.43 ± 1.31 |
| number of rules | 1 | 0.24 | 5.62 ± 0.99 |
| number of rules | 2 | 0.15 | 2.88 ± 0.54 |
| number of rules | 3.474 | 0.17 | 3.29 ± 0.75 |
| number of rules | 6.497 | 0.17 | 1.86 ± 0.34 |
| number of rules | 12.89 | 0.15 | 1.39 ± 0.33 |
| number of rules | 25.09 | 0.12 | 1.04 ± 0.29 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `c + A/v` | `c = 2.48`, `A = 6.3·10⁻¹⁶` | 105.6 over 10 groups |
| density | `a + b·log₂ p` | `a = 4.56`, `b = 0.422` | 13.5 over 10 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 0.81`, `A = 11.14`, `m₀ = 0.67` | 10.0 over 6 groups |

The laws run through the bins without the poorly determined ones; a bin weighs `1/(σᵢ² + τ²)`, its own variance and the scatter of the bins around the law, and the errors are carried by the full covariance of the bins.

The law of the length is a binary exponential, `μ − A·2^(γ·log₂(v/20))`: `mu = 3.25 ± 0.33`, `A = −0.00081 ± 0.0020`, `gamma = 1.23 ± 0.37`; `χ² = 10.9` over 10 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the share of the answer is a line, `a + b·q`: `a = 0.52 ± 0.12`, `b = −0.44 ± 0.16`; `χ² = 10.4` over 9 bins; chosen by the smallest `QAIC` among 2 forms. A fall of 0.44 over the whole answer.

The law of the density is a logarithm plus a line, `a + b·log₂(p/2⁻¹⁰) + c·0.0009765625·2^(log₂(p/2⁻¹⁰))`: `a = 4.47 ± 0.51`, `b = −0.45 ± 0.08`, `c = −3.11 ± 0.72`; `χ² = 5.9` over 9 bins; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 0.19 ± 0.32`, `b = 0.27 ± 0.07`; `χ² = 2.6` over 5 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α_F = V(20) + q(0) + δ_P + M(1) = 8.44 ± 0.50`, where `δ_P` is the maximum of the law of the density, at `p = 0.0010`; `V(20)` = 3.25 ± 0.33, `q(0)` = 0.52 ± 0.12, `δ_P` = 4.47 ± 0.51, `M(1)` = 0.19 ± 0.32. The edge of the measured area: 32 rules and 20,480 words.

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
