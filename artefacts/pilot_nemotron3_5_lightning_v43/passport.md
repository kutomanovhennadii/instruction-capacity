# Passport of `nvidia/nemotron-3.5-lightning:free`

Measurement `pilot_nemotron3_5_lightning_v43`, 4,753 answered points. The method is described in `docs/05-passport-model.md` of the repository.

## The formula

`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`. The formula is the way to combine the three quantities: it has no law and no capacity of its own. `π = m/D` is the replacement density. The zero point of a capacity is a text of 20 words, the start of the answer, the density at the maximum of its law and one rule; every other axis of the capacity is set by its definition.

The capacities are taken at the level 0.5. At it the share of successes over all requests and the success of the median request reach one half at the same logit, zero, whatever the variance of the request shift: a capacity is the load at which the laws bring the logit of the median request to zero, and it does not depend on that variance.

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

The law of the grid: a length slice is a hyperbola with a shared slope and corner width and its own plateau, and the law of the plateau runs through the plateaus. `κ_S = −1.63 ± 0.17`, `λ̃_S = 11.99 ± 2.18`, `ω_S = 3.36 ± 0.78`; `χ² = 112.5` over 77 cells.

The law of the plateau is a slanted hyperbola, `μ + (κ/2)·[(log₂(v/20) − log₂(v/20)*) + √((log₂(v/20) − log₂(v/20)*)² + ω²)]`: `corner = 8.27 ± 0.44`, `kappa = −5.51 ± 2.10`, `mu = 4.40 ± 0.72`, `omega = 0.00 ± 0.00`; `χ² = 19.6` over 11 bins; on the bound of the search: omega; chosen by `χ² + 2k` among 4 forms.

The law of the density is a slanted hyperbola, `μ + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `corner = 6.51 ± 0.93`, `kappa = −2.59 ± 0.71`, `mu = 5.47 ± 1.15`, `omega = 2.21 ± 1.12`; `χ² = 11.7` over 10 bins; chosen by `χ² + 2k` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 0.85 ± 0.42`, `b = 0.18 ± 0.11`; `χ² = 9.4` over 5 bins; chosen by `χ² + 2k` among 4 forms.

The level `α = δ_VS + δ_P + δ_M = 10.73 ± 1.29` is the logit at the density going to zero and one rule, and that density is itself a continuation of the law.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | none: the law gives no root | at least 32, reserve left 9.87 |
| fatigue capacity, words | 0.5 | 9,735 (7,277–13,022) | - |
| how many times full density multiplies the odds of an error | - | 19,231 (4,719–78,374) | - |

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

Model VS-PMR, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 2.37`; 4,680 requests, 590,039 trials, 448,568 failures, 97 parameters; removed bins: none.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 24 parameters more than one variance, `2·Δ ln L = 1343.3`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 32.14 | 0.03 | 1.57 ± 0.75 |
| text length, words | 74.78 | 0.04 | 0.69 ± 0.26 |
| text length, words | 138.2 | 0.07 | 0.66 ± 0.22 |
| text length, words | 265.6 | 0.11 | 0.90 ± 0.22 |
| text length, words | 502 | 0.12 | 0.84 ± 0.18 |
| text length, words | 980.1 | 0.14 | 1.07 ± 0.24 |
| text length, words | 1924 | 0.15 | 1.38 ± 0.31 |
| text length, words | 3748 | 0.15 | 1.69 ± 0.39 |
| text length, words | 7305 | 0.12 | 2.88 ± 0.77 |
| text length, words | 1.42e+04 | 0.07 | 5.36 ± 2.41 |
| density | 0.001312 | 0.04 | 0.73 ± 0.54 |
| density | 0.00273 | 0.07 | 1.07 ± 0.45 |
| density | 0.005597 | 0.06 | 1.96 ± 0.89 |
| density | 0.0108 | 0.10 | 1.89 ± 0.57 |
| density | 0.02093 | 0.07 | 2.23 ± 0.66 |
| density | 0.04127 | 0.10 | 2.11 ± 0.55 |
| density | 0.07966 | 0.11 | 1.55 ± 0.36 |
| density | 0.1528 | 0.12 | 1.16 ± 0.22 |
| density | 0.3073 | 0.16 | 1.19 ± 0.25 |
| density | 0.6044 | 0.12 | 1.06 ± 0.26 |
| density | 1 | 0.05 | 0.95 ± 0.41 |
| number of rules | 1 | 0.22 | 5.18 ± 1.02 |
| number of rules | 2 | 0.15 | 3.78 ± 0.77 |
| number of rules | 3.523 | 0.17 | 1.39 ± 0.26 |
| number of rules | 6.57 | 0.17 | 0.74 ± 0.13 |
| number of rules | 12.94 | 0.15 | 0.41 ± 0.08 |
| number of rules | 25.22 | 0.13 | 0.44 ± 0.12 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = −0.23`, `b = 0.155` | 68.1 over 10 groups |
| density | `a + b·log₂ p` | `a = 1.01`, `b = −0.116` | 28.6 over 11 groups |
| number of rules | `c + A/m` | `c = 0.12`, `A = 5.04` | 26.2 over 6 groups |

The law of the grid: a length slice is the exponent `μ_v − A·2^(γ·log₂ s)`, which dies out to its level `μ_v` at `s → 0`, and the law of the levels runs through the levels of the slices. `A = 0.565 ± 0.092`, `γ = 0.288 ± 0.018`; by the own variances, `χ² = 262.6` over 77 cells.

The law of the levels of the slices is a line, `a + b·log₂(v/20)`: `a = 6.25 ± 0.30`, `b = −0.26 ± 0.03`; `χ² = 23.3` over 11 bins; chosen by `χ² + 2k` among 4 forms.

The law of the density is a two-slope hyperbola, `μ + β·(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `beta = 0.47 ± 0.10`, `corner = 6.00 ± 0.24`, `kappa = −0.89 ± 0.14`, `mu = 0.94 ± 0.26`, `omega = 0.25 ± 1.19`; `χ² = 9.1` over 10 bins; chosen by `χ² + 2k` among 5 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −1.88 ± 0.26`, `b = −0.07 ± 0.04`; `χ² = 33.4` over 5 bins; chosen by `χ² + 2k` among 4 forms.

The law of the rule number is a slanted hyperbola, `μ + (κ/2)·[(log₂ r − log₂ r*) + √((log₂ r − log₂ r*)² + ω²)]`: `corner = 3.51 ± 6.20`, `kappa = −0.56 ± 0.83`, `mu = 0.50 ± 0.24`, `omega = 4.47 ± 0.00`; `χ² = 0.3` over 5 bins; on the bound of the search: omega; chosen by `χ² + 2k` among 4 forms.

The level `α = μ₀ + μ_P + λ_M + λ_R = 2.36 ± 0.52`. The loss of the density at `p = 1` is −1.17 logits.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 8.0 (6.7–9.7) | 3.3 |
| fatigue capacity, words | 0.5 | 67.3 (40.5–112) | - |
| how many times full density multiplies the odds of an error | - | 0.3 (0.2–0.5) | - |

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
| text length, words | `c + A/v` | `c = 2.48`, `A = 0.00` | 105.6 over 10 groups |
| density | `a + b·log₂ p` | `a = 4.56`, `b = 0.422` | 13.5 over 10 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 0.81`, `A = 11.14`, `m₀ = 0.67` | 10.0 over 6 groups |

The laws run through the bins without the poorly determined ones, by ordinary least squares; the form of a law is chosen by `χ² + 2k`, the residuals weighed by the covariance of the bins.

The law of the length is a line, `a + b·log₂(v/20)`: `a = 2.43 ± 0.36`, `b = 0.21 ± 0.06`; `χ² = 145.7` over 11 bins; chosen by `χ² + 2k` among 4 forms.

The law of the share of the answer is an exponential, `c + A·e^(−q/τ)`: `A = 304.77 ± 121.86`, `c = −304.31 ± 121.76`, `tau = 800.00 ± 0.00`; `χ² = 57.1` over 9 bins; on the bound of the search: tau; chosen by `χ² + 2k` among 3 forms. A fall of 0.38 over the whole answer.

The law of the density is a logarithm plus a line, `a + b·log₂(p/2⁻¹⁰) + c·0.0009765625·2^(log₂(p/2⁻¹⁰))`: `a = 4.41 ± 0.50`, `b = −0.42 ± 0.07`, `c = −3.41 ± 0.69`; `χ² = 17.4` over 9 bins; chosen by `χ² + 2k` among 5 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 0.21 ± 0.32`, `b = 0.26 ± 0.07`; `χ² = 7.3` over 5 bins; chosen by `χ² + 2k` among 4 forms.

The level `α_F = V(20) + q(0) + δ_P + M(1) = 7.51 ± 0.67`, where `δ_P` is the maximum of the law of the density, at `p = 0.0010`; `M(1)` = 0.21 ± 0.32, `V(20)` = 2.43 ± 0.36, `δ_P` = 4.41 ± 0.50, `q(0)` = 0.46 ± 0.12. The edge of the measured area: 32 rules and 20,480 words.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | none: the law gives no root | at least 32, reserve left 7.29 |
| fatigue capacity, words | 0.5 | none: the law gives no root | - |
| how many times full density multiplies the odds of an error | - | 2,103 (1,117–3,961), conditional: a continuation of the law of the density | - |

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
