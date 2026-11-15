# Passport of `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`

Measurement `pilot_nemotron3_nano_omni_30b_a3b_reasoning_v45`, 3,695 answered points. The method is described in `docs/05-passport-model.md` of the repository.

## The formula

`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`. The formula is the way to combine the three quantities: it has no law and no capacity of its own. `π = m/D` is the replacement density. The zero point of a capacity is a text of 20 words, the start of the answer, the density at the maximum of its law and one rule; every other axis of the capacity is set by its definition.

The capacities are taken at the level 0.5. At it the share of successes over all requests and the success of the median request reach one half at the same logit, zero, whatever the variance of the request shift: a capacity is the load at which the laws bring the logit of the median request to zero, and it does not depend on that variance.

## P_sync: the answer word stands against its reference word

Model VS-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 35.35`; 3,695 requests, 43,142,085 trials, 34,600,867 failures, 96 parameters; removed bins: none.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 28 parameters more than one variance, `2·Δ ln L = 1469.2`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 31.18 | 0.03 | 1.42 ± 0.85 |
| text length, words | 76.47 | 0.03 | 0.86 ± 0.42 |
| text length, words | 139.3 | 0.05 | 2.42 ± 0.94 |
| text length, words | 274.9 | 0.09 | 2.70 ± 0.82 |
| text length, words | 524 | 0.14 | 5.21 ± 1.60 |
| text length, words | 1023 | 0.15 | 24.56 ± 8.59 |
| text length, words | 2013 | 0.13 | 55.41 ± 16.13 |
| text length, words | 3920 | 0.16 | 76.76 ± 18.66 |
| text length, words | 7870 | 0.13 | 93.55 ± 24.47 |
| text length, words | 1.673e+04 | 0.09 | 52.04 ± 22.63 |
| density | 0.0005799 | 0.02 | 1.84 ± 1.19 |
| density | 0.001503 | 0.04 | 6.15 ± 3.80 |
| density | 0.002587 | 0.03 | 6.93 ± 3.41 |
| density | 0.005295 | 0.04 | 10.60 ± 3.91 |
| density | 0.01021 | 0.06 | 17.33 ± 6.28 |
| density | 0.02152 | 0.08 | 17.52 ± 6.22 |
| density | 0.03966 | 0.08 | 21.07 ± 5.35 |
| density | 0.08027 | 0.11 | 18.14 ± 3.80 |
| density | 0.161 | 0.13 | 17.82 ± 3.63 |
| density | 0.3115 | 0.13 | 18.10 ± 3.54 |
| density | 0.6551 | 0.17 | 39.81 ± 8.67 |
| density | 1 | 0.14 | 49.61 ± 13.84 |
| number of rules | 1 | 0.13 | 40.91 ± 10.24 |
| number of rules | 2 | 0.12 | 21.11 ± 4.92 |
| number of rules | 3.476 | 0.16 | 21.71 ± 4.45 |
| number of rules | 6.841 | 0.11 | 13.66 ± 2.93 |
| number of rules | 13.2 | 0.12 | 14.56 ± 3.39 |
| number of rules | 24.97 | 0.12 | 20.97 ± 4.14 |
| number of rules | 50.1 | 0.10 | 24.32 ± 5.50 |
| number of rules | 95.97 | 0.08 | 19.76 ± 4.10 |
| number of rules | 195.6 | 0.07 | 18.22 ± 4.66 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = −19.15`, `b = 3.999` | 432.1 over 10 groups |
| density | `a + b·log₂ p` | `a = 30.30`, `b = 2.640` | 46.5 over 12 groups |
| number of rules | `c + A/m` | `c = 17.75`, `A = 14.13` | 31.3 over 9 groups |

The law of the grid: a length slice is a hyperbola with a shared slope and corner width and its own plateau, and the law of the plateau runs through the plateaus. `κ_S = −2.64 ± 0.21`, `λ̃_S = 23.89 ± 2.33`, `ω_S = 0.00 ± 16.57`; `χ² = 115.7` over 77 cells.

The law of the plateau is a line, `a + b·log₂(v/20)`: `a = 4.49 ± 0.62`, `b = 0.49 ± 0.13`; `χ² = 189.4` over 11 bins; chosen by `χ² + 2k` among 4 forms.

The law of the density is a slanted hyperbola, `μ + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `mu = 5.49 ± 1.09`, `kappa = −0.97 ± 0.10`, `corner = 2.04 ± 0.99`, `omega = 0.00 ± 0.00`; `χ² = 34.3` over 11 bins; on the bound of the search: omega; chosen by `χ² + 2k` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 0.38 ± 0.50`, `b = −0.42 ± 0.10`; `χ² = 13.3` over 8 bins; chosen by `χ² + 2k` among 4 forms.

The level `α = δ_VS + δ_P + δ_M = 10.36 ± 1.20` is the logit at the density going to zero and one rule, and that density is itself a continuation of the law.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 2.6·10⁷ (2.3·10⁵–3.0·10⁹), continuation of the law | at least 256, reserve left 6.73 |
| fatigue capacity, words | 0.5 | 8,125 (6,601–10,001) | - |
| how many times full density multiplies the odds of an error | - | 2,177 (688–6,883) | - |

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

Model VS-PMR, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 1.74`; 3,656 requests, 829,043 trials, 713,686 failures, 103 parameters; removed bins: vs_11_11.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 28 parameters more than one variance, `2·Δ ln L = 1025.6`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 31.18 | 0.03 | 0.23 ± 0.14 |
| text length, words | 76.47 | 0.03 | 0.22 ± 0.12 |
| text length, words | 139.3 | 0.05 | 0.41 ± 0.18 |
| text length, words | 275 | 0.09 | 0.39 ± 0.15 |
| text length, words | 524.9 | 0.14 | 0.58 ± 0.17 |
| text length, words | 1022 | 0.15 | 0.61 ± 0.18 |
| text length, words | 2014 | 0.13 | 0.85 ± 0.22 |
| text length, words | 3924 | 0.16 | 0.80 ± 0.21 |
| text length, words | 7870 | 0.13 | 1.72 ± 0.56 |
| text length, words | 1.667e+04 | 0.09 | 3.04 ± 1.53 |
| density | 0.0005835 | 0.02 | 0.01 ± 0.08 |
| density | 0.00151 | 0.04 | 0.13 ± 0.13 |
| density | 0.002584 | 0.03 | 0.38 ± 0.28 |
| density | 0.005317 | 0.04 | 0.48 ± 0.29 |
| density | 0.0102 | 0.06 | 0.42 ± 0.18 |
| density | 0.02152 | 0.08 | 0.82 ± 0.35 |
| density | 0.03979 | 0.08 | 0.58 ± 0.18 |
| density | 0.08023 | 0.11 | 0.87 ± 0.20 |
| density | 0.1612 | 0.13 | 0.73 ± 0.13 |
| density | 0.3114 | 0.13 | 0.93 ± 0.21 |
| density | 0.6542 | 0.17 | 1.58 ± 0.51 |
| density | 1 | 0.13 | 1.49 ± 0.56 |
| number of rules | 1 | 0.13 | 8.69 ± 2.77 |
| number of rules | 2 | 0.12 | 2.52 ± 0.69 |
| number of rules | 3.479 | 0.16 | 1.48 ± 0.39 |
| number of rules | 6.841 | 0.11 | 0.59 ± 0.16 |
| number of rules | 13.2 | 0.12 | 0.31 ± 0.09 |
| number of rules | 24.92 | 0.12 | 0.21 ± 0.07 |
| number of rules | 50.02 | 0.09 | 0.21 ± 0.07 |
| number of rules | 96.19 | 0.08 | 0.26 ± 0.13 |
| number of rules | 196.1 | 0.07 | 0.42 ± 0.21 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = −0.45`, `b = 0.119` | 29.4 over 10 groups |
| density | `a + b·log₂ p` | `a = 1.17`, `b = 0.106` | 17.8 over 12 groups |
| number of rules | `c + A/m` | `c = 0.15`, `A = 4.33` | 50.9 over 9 groups |

The law of the grid: a length slice is the exponent `μ_v − A·2^(γ·log₂ s)`, which dies out to its level `μ_v` at `s → 0`, and the law of the levels runs through the levels of the slices. `A = 1.369 ± 0.294`, `γ = 0.239 ± 0.024`; by the own variances, `χ² = 122.3` over 76 cells.

The law of the levels of the slices is a line, `a + b·log₂(v/20)`: `a = 6.06 ± 0.57`, `b = −0.30 ± 0.04`; `χ² = 1.9` over 11 bins; chosen by `χ² + 2k` among 4 forms.

The law of the density is a two-slope hyperbola, `μ + β·(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `mu = 2.00 ± 0.91`, `kappa = −0.87 ± 0.33`, `corner = 6.54 ± 0.55`, `omega = 1.67 ± 1.71`, `beta = −0.02 ± 0.15`; `χ² = 1.6` over 11 bins; chosen by `χ² + 2k` among 5 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −0.08 ± 0.37`, `b = 0.01 ± 0.03`; `χ² = 235.2` over 8 bins; chosen by `χ² + 2k` among 4 forms.

The law of the rule number is a slanted hyperbola, `μ + (κ/2)·[(log₂ r − log₂ r*) + √((log₂ r − log₂ r*)² + ω²)]`: `mu = 0.21 ± 0.46`, `kappa = −1.70 ± 1.87`, `corner = 7.14 ± 2.51`, `omega = 2.68 ± 0.48`; `χ² = 0.8` over 8 bins; chosen by `χ² + 2k` among 4 forms.

The level `α = μ₀ + μ_P + λ_M + λ_R = 7.85 ± 0.61`. The loss of the density at `p = 1` is 3.34 logits.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 24.2 (20.2–28.9) | 23.9 |
| fatigue capacity, words | 0.5 | 623 (512–758) | - |
| how many times full density multiplies the odds of an error | - | 28.2 (18.4–43.4) | - |

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

Model V-(s/v)-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 2.24`; 3,146 requests, 7,689,136 trials, 4,083 failures, 38 parameters; removed bins: none.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 23 parameters more than one variance, `2·Δ ln L = 318.6`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 84.96 | 0.09 | 1.36 ± 0.87 |
| text length, words | 280.7 | 0.08 | 0.86 ± 0.47 |
| text length, words | 534.3 | 0.12 | 0.99 ± 0.44 |
| text length, words | 1067 | 0.11 | 1.13 ± 0.51 |
| text length, words | 2007 | 0.15 | 1.07 ± 0.54 |
| text length, words | 3925 | 0.19 | 1.37 ± 0.51 |
| text length, words | 7883 | 0.16 | 0.91 ± 0.44 |
| text length, words | 1.672e+04 | 0.11 | 1.09 ± 0.83 |
| density | 0.001503 | 0.09 | 0.10 ± 0.12 |
| density | 0.005295 | 0.05 | 0.42 ± 0.35 |
| density | 0.01021 | 0.07 | 0.70 ± 0.43 |
| density | 0.02149 | 0.09 | 0.41 ± 0.26 |
| density | 0.03979 | 0.09 | 0.48 ± 0.24 |
| density | 0.08018 | 0.12 | 1.29 ± 0.48 |
| density | 0.1612 | 0.15 | 2.22 ± 0.65 |
| density | 0.3114 | 0.15 | 2.05 ± 1.04 |
| density | 0.643 | 0.19 | 4.06 ± 1.86 |
| number of rules | 1 | 0.16 | 3.09 ± 1.40 |
| number of rules | 2 | 0.14 | 2.56 ± 1.08 |
| number of rules | 3.402 | 0.16 | 2.12 ± 0.74 |
| number of rules | 6.688 | 0.11 | 1.35 ± 0.48 |
| number of rules | 12.95 | 0.12 | 0.65 ± 0.25 |
| number of rules | 24.97 | 0.11 | 0.83 ± 0.64 |
| number of rules | 50.51 | 0.10 | 0.34 ± 0.18 |
| number of rules | 97.35 | 0.07 | 0.18 ± 0.12 |
| number of rules | 199.6 | 0.05 | 0.21 ± 0.27 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `c + A/v` | `c = 1.08`, `A = 9.76` | 3.3 over 8 groups |
| density | `a + b·log₂ p` | `a = 2.29`, `b = 0.236` | 32.4 over 9 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 0.11`, `A = 4.50`, `m₀ = 2.37` | 3.8 over 9 groups |

The laws run through the bins without the poorly determined ones, by ordinary least squares; the form of a law is chosen by `χ² + 2k`, the residuals weighed by the covariance of the bins.

The law of the length is a line, `a + b·log₂(v/20)`: `a = 5.18 ± 0.56`, `b = 0.60 ± 0.08`; `χ² = 29.1` over 11 bins; chosen by `χ² + 2k` among 4 forms.

The law of the share of the answer is a hyperbola, `c + A/q`: `c = 0.13 ± 0.20`, `A = −0.12 ± 0.01`; `χ² = 7.2` over 9 bins; chosen by `χ² + 2k` among 3 forms. A fall of −2.27 over the whole answer.

The law of the density is a logarithm plus a line, `a + b·log₂(p/2⁻¹⁰) + c·0.0009765625·2^(log₂(p/2⁻¹⁰))`: `a = −0.18 ± 0.51`, `b = 0.09 ± 0.07`, `c = −1.79 ± 0.79`; `χ² = 15.3` over 10 bins; chosen by `χ² + 2k` among 5 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 0.01 ± 0.39`, `b = −0.27 ± 0.06`; `χ² = 52.7` over 8 bins; chosen by `χ² + 2k` among 4 forms.

The level `α_F = V(20) + q(0) + δ_P + M(1) = 3.20 ± 0.50`, where `δ_P` is the maximum of the law of the density, at `p = 0.0757`; `V(20)` = 5.18 ± 0.56, `q(0)` = −2.26 ± 0.24, `δ_P` = 0.28 ± 0.26, `M(1)` = 0.01 ± 0.39. The edge of the measured area: 256 rules and 20,480 words.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 3,574 (308–41,452), continuation of the law | at least 256, reserve left 1.34 |
| fatigue capacity, words | 0.5 | none: the law gives no root | - |
| how many times full density multiplies the odds of an error | - | 2.3 (1.2–4.7), conditional: a continuation of the law of the density | - |

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
