# Passport of `gemini-3.5-flash-lite`

Measurement `pilot_gemini3_5_flash_lite_v58`, 3,788 answered points. The method is described in `docs/05-passport-model.md` of the repository.

## The formula

`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`. The formula is the way to combine the three quantities: it has no law and no capacity of its own. `π = m/D` is the replacement density. The zero point of a capacity is a text of 20 words, the start of the answer, the density at the maximum of its law and one rule; every other axis of the capacity is set by its definition.

The capacities are taken at the level 0.5. At it the share of successes over all requests and the success of the median request reach one half at the same logit, zero, whatever the variance of the request shift: a capacity is the load at which the laws bring the logit of the median request to zero, and it does not depend on that variance.

## P_sync: the answer word stands against its reference word

Model VS-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 15.00`; 3,775 requests, 48,842,597 trials, 16,946,917 failures, 104 parameters; removed bins: vs_1_1, vs_1_2, vs_2_1, vs_2_2, vs_3_1, vs_6_1.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 28 parameters more than one variance, `2·Δ ln L = 1870.8`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 93.16 | 0.08 | 1.32 ± 1.11 |
| text length, words | 283.5 | 0.05 | 0.96 ± 0.54 |
| text length, words | 547.5 | 0.06 | 0.81 ± 0.39 |
| text length, words | 1090 | 0.08 | 2.46 ± 1.23 |
| text length, words | 2153 | 0.12 | 4.96 ± 2.29 |
| text length, words | 4283 | 0.13 | 4.45 ± 1.26 |
| text length, words | 8402 | 0.16 | 9.98 ± 3.45 |
| text length, words | 1.689e+04 | 0.17 | 20.04 ± 6.59 |
| text length, words | 3.146e+04 | 0.15 | 56.24 ± 20.68 |
| density | 0.0006999 | 0.03 | 7.68 ± 7.96 |
| density | 0.001246 | 0.03 | 12.05 ± 9.21 |
| density | 0.002461 | 0.04 | 5.45 ± 5.71 |
| density | 0.004717 | 0.04 | 1.34 ± 1.14 |
| density | 0.009694 | 0.05 | 2.82 ± 2.38 |
| density | 0.01927 | 0.05 | 4.30 ± 3.94 |
| density | 0.0375 | 0.06 | 1.38 ± 0.78 |
| density | 0.07309 | 0.07 | 3.54 ± 1.76 |
| density | 0.1475 | 0.10 | 4.02 ± 1.45 |
| density | 0.3032 | 0.19 | 5.44 ± 1.36 |
| density | 0.6632 | 0.20 | 13.75 ± 3.57 |
| density | 1 | 0.15 | 33.30 ± 8.98 |
| number of rules | 1 | 0.13 | 10.04 ± 3.75 |
| number of rules | 2 | 0.12 | 15.36 ± 6.89 |
| number of rules | 3.624 | 0.15 | 16.35 ± 7.69 |
| number of rules | 6.916 | 0.13 | 8.50 ± 3.20 |
| number of rules | 13.04 | 0.13 | 3.79 ± 1.13 |
| number of rules | 26.02 | 0.10 | 3.17 ± 1.31 |
| number of rules | 52.33 | 0.07 | 2.86 ± 1.34 |
| number of rules | 98.94 | 0.06 | 1.61 ± 0.67 |
| number of rules | 206.3 | 0.06 | 9.29 ± 7.35 |
| number of rules | 381.5 | 0.05 | 5.60 ± 3.17 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = −7.96`, `b = 1.292` | 168.5 over 9 groups |
| density | `a + b·log₂ p` | `a = 10.34`, `b = 0.751` | 176.7 over 12 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 2.49`, `A = 16.61`, `m₀ = 2.31` | 41.0 over 10 groups |

The law of the grid: a length slice is a hyperbola with a shared slope and corner width and its own plateau, and the law of the plateau runs through the plateaus. `κ_S = −4.13 ± 4.30`, `λ̃_S = 77.62 ± 94.55`, `ω_S = 9.62 ± 3.76`; `χ² = 268.1` over 83 cells.

The law of the plateau is a line, `a + b·log₂(v/20)`: `a = 9.46 ± 3.06`, `b = 0.93 ± 0.22`; `χ² = 8.8` over 10 bins; chosen by `χ² + 2k` among 4 forms.

The law of the density is a slanted hyperbola, `μ + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `corner = 2.45 ± 0.69`, `kappa = −0.61 ± 0.07`, `mu = 3.78 ± 0.56`, `omega = 0.09 ± 4.63`; `χ² = 9.0` over 11 bins; chosen by `χ² + 2k` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −0.29 ± 0.26`, `b = 0.21 ± 0.05`; `χ² = 12.0` over 9 bins; chosen by `χ² + 2k` among 4 forms.

The level `α = δ_VS + δ_P + δ_M = 12.95 ± 3.15` is the logit at the density going to zero and one rule, and that density is itself a continuation of the law.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | none: the law gives no root | at least 512, reserve left 12.95 |
| fatigue capacity, words | 0.5 | 3.2·10⁵ (1.2·10⁵–8.5·10⁵), continuation of the law | - |
| how many times full density multiplies the odds of an error | - | 98.3 (51.8–186) | - |

The instruction capacity is the number of rules at which the law of the number of rules brings the logit to zero, and next to it the walk along the bins of that axis; by the law it is none when the fall of that law over its bins does not differ from zero. The fatigue capacity is the word at which the full law of the grid at the density going to zero and one rule reaches zero, on a text of the same length: by the curve of the law, as for `H`. Full density multiplies the odds of an error by `e^(−P̂(1))`, `P̂(1) = P(1) − δ_P` the change of the law of the density from the density going to zero to `p = 1`; a factor below one means that full density lowers the odds of an error. The interval is `e^(−P̂(1) ∓ h)`, `h` the half-width of `P̂(1)`.

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

Model VS-PMR, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 2.98`; 3,769 requests, 7,817,448 trials, 6,848,368 failures, 116 parameters; removed bins: vs_1_1, vs_2_1, vs_4_1.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 29 parameters more than one variance, `2·Δ ln L = 904.6`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 57.28 | 0.05 | 0.96 ± 0.36 |
| text length, words | 145.5 | 0.04 | 0.55 ± 0.27 |
| text length, words | 283.5 | 0.05 | 0.59 ± 0.20 |
| text length, words | 547.5 | 0.06 | 0.67 ± 0.15 |
| text length, words | 1090 | 0.08 | 1.19 ± 0.32 |
| text length, words | 2152 | 0.12 | 1.65 ± 0.37 |
| text length, words | 4283 | 0.13 | 2.33 ± 0.56 |
| text length, words | 8400 | 0.16 | 4.36 ± 1.14 |
| text length, words | 1.686e+04 | 0.17 | 4.40 ± 1.19 |
| text length, words | 3.135e+04 | 0.15 | 5.40 ± 1.46 |
| density | 0.0006999 | 0.03 | 0.17 ± 0.17 |
| density | 0.001249 | 0.03 | 0.57 ± 0.34 |
| density | 0.002461 | 0.04 | 0.56 ± 0.30 |
| density | 0.004723 | 0.04 | 0.83 ± 0.36 |
| density | 0.009694 | 0.05 | 1.10 ± 0.32 |
| density | 0.01936 | 0.05 | 0.88 ± 0.42 |
| density | 0.03753 | 0.06 | 2.23 ± 0.86 |
| density | 0.07312 | 0.07 | 1.49 ± 0.49 |
| density | 0.1469 | 0.10 | 2.96 ± 0.77 |
| density | 0.303 | 0.19 | 3.13 ± 0.50 |
| density | 0.6628 | 0.20 | 4.65 ± 0.64 |
| density | 1 | 0.15 | 4.98 ± 0.89 |
| number of rules | 1 | 0.13 | 6.36 ± 1.66 |
| number of rules | 2 | 0.12 | 3.78 ± 0.75 |
| number of rules | 3.619 | 0.15 | 2.57 ± 0.42 |
| number of rules | 6.919 | 0.13 | 1.93 ± 0.37 |
| number of rules | 13.04 | 0.13 | 1.29 ± 0.28 |
| number of rules | 26.01 | 0.10 | 1.16 ± 0.28 |
| number of rules | 52.33 | 0.07 | 1.56 ± 0.67 |
| number of rules | 98.94 | 0.06 | 1.60 ± 0.71 |
| number of rules | 205.8 | 0.06 | 2.13 ± 1.13 |
| number of rules | 382.1 | 0.05 | 3.09 ± 1.41 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = −1.24`, `b = 0.287` | 126.1 over 10 groups |
| density | `a + b·log₂ p` | `a = 4.02`, `b = 0.373` | 45.8 over 12 groups |
| number of rules | `c + A/m` | `c = 1.29`, `A = 4.62` | 26.8 over 10 groups |

The law of the grid: a length slice is the exponent `μ_v − A·2^(γ·log₂ s)`, which dies out to its level `μ_v` at `s → 0`, and the law of the levels runs through the levels of the slices. `A = 4.649 ± 0.985`, `γ = 0.139 ± 0.016`; by the own variances, `χ² = 370.9` over 87 cells.

The law of the levels of the slices is a line, `a + b·log₂(v/20)`: `a = 12.90 ± 1.29`, `b = −0.04 ± 0.06`; `χ² = 7.0` over 12 bins; chosen by `χ² + 2k` among 4 forms.

The law of the density is a two-slope hyperbola, `μ + β·(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `beta = −0.15 ± 0.05`, `corner = 8.46 ± 0.33`, `kappa = 1.44 ± 0.37`, `mu = −0.17 ± 0.18`, `omega = 0.00 ± 0.00`; `χ² = 20.8` over 11 bins; on the bound of the search: omega; chosen by `χ² + 2k` among 5 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −0.87 ± 0.33`, `b = −0.21 ± 0.05`; `χ² = 25.6` over 9 bins; chosen by `χ² + 2k` among 4 forms.

The law of the rule number is a slanted hyperbola, `μ + (κ/2)·[(log₂ r − log₂ r*) + √((log₂ r − log₂ r*)² + ω²)]`: `corner = 0.00 ± 0.00`, `kappa = −0.17 ± 0.01`, `mu = −0.29 ± 0.21`, `omega = 0.00 ± 0.00`; `χ² = 18.6` over 9 bins; on the bound of the search: corner, omega; chosen by `χ² + 2k` among 4 forms.

The level `α = μ₀ + μ_P + λ_M + λ_R = 12.81 ± 1.32`. The loss of the density at `p = 1` is −0.75 logits.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 22.4 (20.5–24.4) | 20.8 |
| fatigue capacity, words | 0.5 | 1,291 (1,081–1,542) | - |
| how many times full density multiplies the odds of an error | - | 0.5 (0.3–0.7) | - |

Full density multiplies the odds of an error by `e^(−P̂(1))`, `P̂(1) = P(1) − μ_P` the change of the law of the density from its zero point, `p = 2⁻¹⁰` for the two-slope hyperbola, to `p = 1`; a factor below one means that full density lowers the odds of an error. The interval is `e^(−P̂(1) ∓ h)`, `h` the half-width of `P̂(1)`.

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

Model V-(s/v)-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 1.33`; 3,200 requests, 24,081,763 trials, 23,524 failures, 39 parameters; removed bins: v_bin_1.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 26 parameters more than one variance, `2·Δ ln L = 204.9`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 95.32 | 0.07 | 0.13 ± 0.39 |
| text length, words | 282.6 | 0.05 | 0.63 ± 0.67 |
| text length, words | 544.1 | 0.06 | 0.56 ± 0.40 |
| text length, words | 1095 | 0.08 | 0.84 ± 0.48 |
| text length, words | 2172 | 0.11 | 1.34 ± 0.47 |
| text length, words | 4276 | 0.13 | 1.54 ± 0.48 |
| text length, words | 8375 | 0.17 | 1.70 ± 0.49 |
| text length, words | 1.712e+04 | 0.17 | 1.51 ± 0.36 |
| text length, words | 3.156e+04 | 0.15 | 2.10 ± 0.66 |
| density | 0.000957 | 0.07 | 0.47 ± 0.45 |
| density | 0.002461 | 0.04 | 0.58 ± 0.54 |
| density | 0.004723 | 0.05 | 0.88 ± 0.62 |
| density | 0.009694 | 0.06 | 1.00 ± 0.64 |
| density | 0.01936 | 0.06 | 0.51 ± 0.33 |
| density | 0.03753 | 0.07 | 0.64 ± 0.36 |
| density | 0.07311 | 0.08 | 0.65 ± 0.33 |
| density | 0.147 | 0.11 | 1.15 ± 0.45 |
| density | 0.3034 | 0.22 | 1.31 ± 0.43 |
| density | 0.6624 | 0.23 | 2.96 ± 1.08 |
| number of rules | 1 | 0.15 | 1.49 ± 0.81 |
| number of rules | 2 | 0.14 | 1.16 ± 0.59 |
| number of rules | 3.556 | 0.15 | 2.05 ± 0.65 |
| number of rules | 6.768 | 0.13 | 1.22 ± 0.46 |
| number of rules | 12.82 | 0.13 | 1.06 ± 0.46 |
| number of rules | 25.67 | 0.09 | 0.73 ± 0.42 |
| number of rules | 51.68 | 0.07 | 0.89 ± 0.41 |
| number of rules | 97.09 | 0.06 | 0.75 ± 0.38 |
| number of rules | 204.7 | 0.04 | 0.39 ± 0.32 |
| number of rules | 372.5 | 0.04 | 1.16 ± 0.89 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `c + A/(1 + v/v₀)` | `c = 1.49`, `A = 0.00`, `v₀ = 9783811.97` | 21.8 over 9 groups |
| density | `a + b·log₂ p` | `a = 1.63`, `b = 0.128` | 24.9 over 10 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 0.65`, `A = 1.24`, `m₀ = 7.83` | 10.7 over 10 groups |

The laws run through the bins without the poorly determined ones, by ordinary least squares; the form of a law is chosen by `χ² + 2k`, the residuals weighed by the covariance of the bins.

The law of the length is a line, `a + b·log₂(v/20)`: `a = 3.76 ± 0.49`, `b = 0.42 ± 0.05`; `χ² = 65.8` over 11 bins; chosen by `χ² + 2k` among 4 forms.

The law of the share of the answer is a hyperbola, `c + A/q`: `A = −0.07 ± 0.02`, `c = 0.18 ± 0.08`; `χ² = 3.0` over 9 bins; chosen by `χ² + 2k` among 3 forms. A fall of −1.39 over the whole answer.

The law of the density is a logarithm plus a line, `a + b·log₂(p/2⁻¹⁰) + c·0.0009765625·2^(log₂(p/2⁻¹⁰))`: `a = 3.27 ± 0.54`, `b = −0.31 ± 0.08`, `c = −2.20 ± 0.83`; `χ² = 8.8` over 10 bins; chosen by `χ² + 2k` among 5 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 0.09 ± 0.38`, `b = 0.02 ± 0.05`; `χ² = 7.5` over 9 bins; chosen by `χ² + 2k` among 4 forms.

The level `α_F = V(20) + q(0) + δ_P + M(1) = 5.84 ± 0.69`, where `δ_P` is the maximum of the law of the density, at `p = 0.0010`; `M(1)` = 0.09 ± 0.38, `V(20)` = 3.76 ± 0.49, `δ_P` = 3.27 ± 0.54, `q(0)` = −1.28 ± 0.28. The edge of the measured area: 512 rules and 40,960 words.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | none: the law gives no root | at least 512, reserve left 5.75 |
| fatigue capacity, words | 0.5 | none: the law gives no root | - |
| how many times full density multiplies the odds of an error | - | 206 (103–415), conditional: a continuation of the law of the density | - |

The instruction capacity is the number of rules at the zero point. The fatigue capacity is the number of the last word of a text of the same length at the zero point. Full density multiplies the odds of a misfire by `e^(−P̂(1))`, `P̂(1) = P(1) − P(p → 0)`; a factor below one would mean that full density lowers them. It is a conditional quantity: a text has no background words at `p = 1`, so it always continues the law of the density, and it allows comparing the effect of the density on `P_sync`, `H` and `F`.

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
