# Passport of `gpt-4o-mini-2024-07-18`

Measurement `pilot_gpt4o_mini_v55`, 5,172 answered points. The method is described in `docs/05-passport-model.md` of the repository.

## The formula

`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`. The formula is the way to combine the three quantities: it has no law and no capacity of its own. `π = m/D` is the replacement density. The zero point of a capacity is a text of 20 words, the start of the answer, the density at the maximum of its law and one rule; every other axis of the capacity is set by its definition.

The capacities are taken at the level 0.5. At it the share of successes over all requests and the success of the median request reach one half at the same logit, zero, whatever the variance of the request shift: a capacity is the load at which the laws bring the logit of the median request to zero, and it does not depend on that variance.

## P_sync: the answer word stands against its reference word

Model VS-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 14.23`; 5,162 requests, 22,598,814 trials, 11,269,394 failures, 75 parameters; removed bins: vs_1_1, vs_1_2, vs_2_2, vs_2_3, vs_3_1, vs_3_2, vs_10_2, vs_10_4.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 24 parameters more than one variance, `2·Δ ln L = 592.3`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 110.6 | 0.10 | 14.97 ± 12.59 |
| text length, words | 267.7 | 0.07 | 7.88 ± 2.64 |
| text length, words | 505.9 | 0.10 | 7.61 ± 1.73 |
| text length, words | 1009 | 0.13 | 8.46 ± 1.30 |
| text length, words | 2025 | 0.20 | 11.85 ± 1.39 |
| text length, words | 4015 | 0.23 | 12.78 ± 1.85 |
| text length, words | 8064 | 0.16 | 10.69 ± 1.66 |
| density | 0.001093 | 0.02 | 22.84 ± 7.64 |
| density | 0.002322 | 0.03 | 23.87 ± 6.61 |
| density | 0.005236 | 0.06 | 24.54 ± 7.19 |
| density | 0.01025 | 0.06 | 25.17 ± 5.97 |
| density | 0.02029 | 0.06 | 20.88 ± 5.03 |
| density | 0.04047 | 0.08 | 12.02 ± 2.64 |
| density | 0.07807 | 0.12 | 10.24 ± 2.30 |
| density | 0.1494 | 0.13 | 9.02 ± 2.12 |
| density | 0.2949 | 0.16 | 6.59 ± 1.08 |
| density | 0.6029 | 0.19 | 9.69 ± 1.72 |
| density | 1 | 0.11 | 6.74 ± 1.88 |
| number of rules | 1 | 0.14 | 19.72 ± 4.20 |
| number of rules | 2 | 0.14 | 14.00 ± 2.13 |
| number of rules | 3.624 | 0.16 | 11.42 ± 1.64 |
| number of rules | 6.862 | 0.17 | 9.28 ± 1.50 |
| number of rules | 13.83 | 0.10 | 7.58 ± 1.30 |
| number of rules | 25.7 | 0.11 | 7.43 ± 1.22 |
| number of rules | 50 | 0.09 | 8.70 ± 1.98 |
| number of rules | 102.4 | 0.06 | 9.00 ± 2.03 |
| number of rules | 192.8 | 0.03 | 11.45 ± 4.82 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = 0.61`, `b = 0.907` | 16.9 over 7 groups |
| density | `a + b·log₂ p` | `a = 5.94`, `b = −1.838` | 49.4 over 11 groups |
| number of rules | `c + A/m` | `c = 7.72`, `A = 12.12` | 9.0 over 9 groups |

The law of the grid: a length slice is a hyperbola with a shared slope and corner width and its own plateau, and the law of the plateau runs through the plateaus. `κ_S = −3.59 ± 0.21`, `λ̃_S = 20.23 ± 2.46`, `ω_S = 2.80 ± 0.49`; `χ² = 186.0` over 56 cells.

The law of the plateau is a slanted hyperbola, `μ + (κ/2)·[(log₂(v/20) − log₂(v/20)*) + √((log₂(v/20) − log₂(v/20)*)² + ω²)]`: `corner = 4.82 ± 0.97`, `kappa = −2.59 ± 0.69`, `mu = 20.01 ± 1.94`, `omega = 0.03 ± 1.18`; `χ² = 10.1` over 8 bins; chosen by `χ² + 2k` among 4 forms.

The law of the density is a slanted hyperbola, `μ + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `corner = 10.00 ± 0.00`, `kappa = −11.59 ± 1.13`, `mu = 16.44 ± 1.28`, `omega = 4.22 ± 0.46`; `χ² = 22.2` over 10 bins; on the bound of the search: corner; chosen by `χ² + 2k` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −0.05 ± 0.55`, `b = −0.04 ± 0.08`; `χ² = 4.0` over 8 bins; chosen by `χ² + 2k` among 4 forms.

The level `α = δ_VS + δ_P + δ_M = 36.41 ± 2.23` is the logit at the density going to zero and one rule, and that density is itself a continuation of the law.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | none: the law gives no root | at least 256, reserve left 36.02 |
| fatigue capacity, words | 0.5 | 44,473 (34,076–58,042), continuation of the law | - |
| how many times full density multiplies the odds of an error | - | 4.1·10¹⁰ (1.0·10¹⁰–1.7·10¹¹) | - |

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

Model VS-PMR, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 3.03`; 5,135 requests, 1,155,179 trials, 119,727 failures, 82 parameters; removed bins: vs_1_1, vs_1_2, vs_2_1, vs_2_2, vs_2_3, vs_3_1, vs_3_2, vs_4_1, vs_10_2.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 24 parameters more than one variance, `2·Δ ln L = 719.1`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 116.7 | 0.10 | 1.88 ± 0.76 |
| text length, words | 267.7 | 0.07 | 1.79 ± 0.50 |
| text length, words | 505.9 | 0.10 | 1.50 ± 0.35 |
| text length, words | 1009 | 0.13 | 1.69 ± 0.35 |
| text length, words | 2025 | 0.20 | 1.69 ± 0.28 |
| text length, words | 4015 | 0.23 | 1.62 ± 0.27 |
| text length, words | 8064 | 0.17 | 1.41 ± 0.24 |
| density | 0.001093 | 0.02 | 0.46 ± 0.36 |
| density | 0.002322 | 0.03 | 0.69 ± 0.42 |
| density | 0.005236 | 0.06 | 1.22 ± 0.41 |
| density | 0.01025 | 0.06 | 1.28 ± 0.39 |
| density | 0.02029 | 0.06 | 1.12 ± 0.35 |
| density | 0.04047 | 0.08 | 1.34 ± 0.37 |
| density | 0.07807 | 0.12 | 1.32 ± 0.29 |
| density | 0.1495 | 0.13 | 1.77 ± 0.34 |
| density | 0.2952 | 0.16 | 1.55 ± 0.27 |
| density | 0.6035 | 0.18 | 2.21 ± 0.48 |
| density | 1 | 0.11 | 3.54 ± 0.95 |
| number of rules | 1 | 0.14 | 8.01 ± 1.73 |
| number of rules | 2 | 0.14 | 5.38 ± 1.06 |
| number of rules | 3.623 | 0.16 | 3.13 ± 0.57 |
| number of rules | 6.855 | 0.17 | 1.64 ± 0.33 |
| number of rules | 13.83 | 0.10 | 0.91 ± 0.23 |
| number of rules | 25.7 | 0.11 | 0.70 ± 0.16 |
| number of rules | 50 | 0.09 | 0.42 ± 0.13 |
| number of rules | 102.4 | 0.07 | 0.16 ± 0.05 |
| number of rules | 192.8 | 0.03 | 0.13 ± 0.05 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = 2.20`, `b = −0.054` | 2.5 over 7 groups |
| density | `a + b·log₂ p` | `a = 2.18`, `b = 0.164` | 24.8 over 11 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 0.06`, `A = 19.03`, `m₀ = 0.72` | 8.6 over 9 groups |

The law of the grid: a length slice is the exponent `μ_v − A·2^(γ·log₂ s)`, which dies out to its level `μ_v` at `s → 0`, and the law of the levels runs through the levels of the slices. `A = 1.065 ± 0.399`, `γ = 0.224 ± 0.036`; by the own variances, `χ² = 151.0` over 56 cells.

The law of the levels of the slices is a line, `a + b·log₂(v/20)`: `a = 16.14 ± 1.03`, `b = −0.71 ± 0.09`; `χ² = 1.4` over 8 bins; chosen by `χ² + 2k` among 4 forms.

The law of the density is a two-slope hyperbola, `μ + β·(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `beta = 0.47 ± 0.27`, `corner = 6.96 ± 0.75`, `kappa = −1.95 ± 0.79`, `mu = 3.45 ± 2.33`, `omega = 2.53 ± 1.67`; `χ² = 5.5` over 10 bins; chosen by `χ² + 2k` among 5 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −1.01 ± 0.47`, `b = −0.38 ± 0.04`; `χ² = 36.9` over 8 bins; chosen by `χ² + 2k` among 4 forms.

The law of the rule number is a slanted hyperbola, `μ + (κ/2)·[(log₂ r − log₂ r*) + √((log₂ r − log₂ r*)² + ω²)]`: `corner = 3.29 ± 0.23`, `kappa = −0.51 ± 0.04`, `mu = 0.10 ± 0.22`, `omega = 0.59 ± 0.65`; `χ² = 7.6` over 8 bins; chosen by `χ² + 2k` among 4 forms.

The level `α = μ₀ + μ_P + λ_M + λ_R = 15.00 ± 1.08`. The loss of the density at `p = 1` is 1.72 logits.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 32.3 (29.1–36.0) | 30.7 |
| fatigue capacity, words | 0.5 | 10,800 (8,912–13,089), continuation of the law | - |
| how many times full density multiplies the odds of an error | - | 5.6 (3.0–10.6) | - |

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

Model V-(s/v)-PM, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 6.17`; 4,571 requests, 10,213,582 trials, 77,048 failures, 34 parameters; removed bins: v_bin_1, v_bin_2.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 22 parameters more than one variance, `2·Δ ln L = 1324.9`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 172.1 | 0.11 | 6.59 ± 1.84 |
| text length, words | 501.7 | 0.09 | 3.97 ± 1.09 |
| text length, words | 1007 | 0.13 | 2.60 ± 0.50 |
| text length, words | 2025 | 0.23 | 1.53 ± 0.20 |
| text length, words | 4015 | 0.25 | 1.11 ± 0.14 |
| text length, words | 8064 | 0.19 | 0.98 ± 0.14 |
| density | 0.001093 | 0.02 | 0.43 ± 0.19 |
| density | 0.002322 | 0.04 | 0.66 ± 0.21 |
| density | 0.005236 | 0.06 | 1.05 ± 0.26 |
| density | 0.01025 | 0.06 | 1.49 ± 0.34 |
| density | 0.02029 | 0.06 | 1.62 ± 0.37 |
| density | 0.04047 | 0.09 | 1.60 ± 0.30 |
| density | 0.07807 | 0.13 | 1.50 ± 0.26 |
| density | 0.1495 | 0.15 | 1.84 ± 0.32 |
| density | 0.2952 | 0.18 | 2.21 ± 0.33 |
| density | 0.6035 | 0.21 | 3.03 ± 0.59 |
| number of rules | 1 | 0.16 | 7.02 ± 1.14 |
| number of rules | 2 | 0.15 | 4.82 ± 0.87 |
| number of rules | 3.54 | 0.14 | 3.74 ± 0.55 |
| number of rules | 6.736 | 0.17 | 2.43 ± 0.35 |
| number of rules | 13.78 | 0.10 | 0.95 ± 0.19 |
| number of rules | 25.61 | 0.11 | 0.58 ± 0.14 |
| number of rules | 49.5 | 0.08 | 0.29 ± 0.08 |
| number of rules | 100.3 | 0.06 | 0.17 ± 0.05 |
| number of rules | 194.4 | 0.03 | 0.13 ± 0.05 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `c + A/(1 + v/v₀)` | `c = 0.69`, `A = 13.78`, `v₀ = 141.21` | 2.3 over 6 groups |
| density | `a + b·log₂ p` | `a = 2.65`, `b = 0.220` | 16.6 over 10 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 0.02`, `A = 13.13`, `m₀ = 1.26` | 11.7 over 9 groups |

The laws run through the bins without the poorly determined ones, by ordinary least squares; the form of a law is chosen by `χ² + 2k`, the residuals weighed by the covariance of the bins.

The law of the length is a line, `a + b·log₂(v/20)`: `a = 9.98 ± 0.46`, `b = −0.45 ± 0.06`; `χ² = 63.9` over 8 bins; chosen by `χ² + 2k` among 4 forms.

The law of the share of the answer is an exponential, `c + A·e^(−q/τ)`: `A = 1.12 ± 0.22`, `c = −0.28 ± 0.24`, `tau = 0.67 ± 0.29`; `χ² = 110.2` over 9 bins; chosen by `χ² + 2k` among 3 forms. A fall of 0.86 over the whole answer.

The law of the density is a logarithm plus a line, `a + b·log₂(p/2⁻¹⁰) + c·0.0009765625·2^(log₂(p/2⁻¹⁰))`: `a = 1.04 ± 0.24`, `b = −0.03 ± 0.04`, `c = −2.64 ± 0.45`; `χ² = 35.7` over 9 bins; chosen by `χ² + 2k` among 5 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −0.63 ± 0.27`, `b = −0.39 ± 0.03`; `χ² = 308.3` over 8 bins; chosen by `χ² + 2k` among 4 forms.

The level `α_F = V(20) + q(0) + δ_P + M(1) = 11.22 ± 0.54`, where `δ_P` is the maximum of the law of the density, at `p = 0.0010`; `M(1)` = −0.63 ± 0.27, `V(20)` = 9.98 ± 0.46, `δ_P` = 1.04 ± 0.24, `q(0)` = 0.84 ± 0.10. The edge of the measured area: 256 rules and 10,240 words.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 4.4·10⁸ (8.5·10⁷–2.3·10⁹), continuation of the law | at least 256, reserve left 8.42 |
| fatigue capacity, words | 0.5 | 1.4·10⁸ (4.0·10⁷–5.2·10⁸), continuation of the law | - |
| how many times full density multiplies the odds of an error | - | 18.6 (12.9–26.8), conditional: a continuation of the law of the density | - |

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
