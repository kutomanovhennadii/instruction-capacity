# Passport of `gpt-4.1-mini-2025-04-14`

Measurement `pilot_gpt4_1_mini_v56`, 6,961 answered points. The method is described in `docs/05-passport-model.md` of the repository.

## The formula

`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`. The formula is the way to combine the three quantities: it has no law and no capacity of its own. `π = m/D` is the replacement density. The zero point of a capacity is a text of 20 words, the start of the answer, the density at the maximum of its law and one rule; every other axis of the capacity is set by its definition.

The capacities are taken at the level 0.5. At it the share of successes over all requests and the success of the median request reach one half at the same logit, zero, whatever the variance of the request shift: a capacity is the load at which the laws bring the logit of the median request to zero, and it does not depend on that variance.

## P_sync: the answer word stands against its reference word

Model VS-PM, request weights by the cube V × P × M. Fit: with the shift of a request; the variance of the request shift `σ² = 27.94`; 6,908 requests, 85,962,046 trials, 45,165,631 failures, 88 parameters; removed bins: vs_1_1, vs_1_2, vs_3_1, vs_3_2, vs_3_3, vs_3_4, vs_4_1, vs_4_3.

The law of the grid: a length slice is a two-asymptote hyperbola, `[(κ·ℓ + λ) + μ − √((κ·ℓ + λ − μ)² + κ²·ω²)]/2` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −5.043 ± 0.234`, `lambda = 64.180 ± 3.352`, `omega = 2.668 ± 0.483`; `χ² = 57.1` over 69 cells; chosen by the smallest `QAIC` among 3 forms. The law of the plateau runs through the levels of the slices.

The law of the plateau is a slanted hyperbola, `μ + (κ/2)·[(log₂(v/20) − log₂(v/20)*) + √((log₂(v/20) − log₂(v/20)*)² + ω²)]`: `mu = 32.50 ± 1.71`, `kappa = −4.72 ± 1.23`, `corner = 5.81 ± 0.85`, `omega = 0.00 ± 0.00`; `χ² = 5.1` over 9 bins; on the bound of the search: omega; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a slanted hyperbola, `μ + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `mu = 16.16 ± 0.98`, `kappa = −9.38 ± 1.27`, `corner = 7.22 ± 0.47`, `omega = 2.09 ± 0.35`; `χ² = 6.3` over 10 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 0.84 ± 1.03`, `b = −0.49 ± 0.08`; `χ² = 7.2` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = δ_VS + δ_P + δ_M = 49.51 ± 2.11` is the logit at the density going to zero and one rule, and that density is itself a continuation of the law.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 4.2·10³⁰ (3.0·10²⁵–5.9·10³⁵), continuation of the law | at least 256, reserve left 45.55 |
| fatigue capacity, words | 0.5 | 53,647 (42,518–67,689), continuation of the law | - |
| how many times full density multiplies the odds of an error | - | 5.7·10¹² (1.6·10¹²–2.0·10¹³) | - |

The instruction capacity is the number of rules at which the law of the number of rules brings the logit to zero, and next to it the walk along the bins of that axis; by the law it is none when the fall of that law over its bins does not differ from zero. The fatigue capacity is the word at which the full law of the grid at the density going to zero and one rule reaches zero, on a text of the same length: by the curve of the law, as for `H`. Full density multiplies the odds of an error by `e^(−P̂(1))`, `P̂(1) = P(1) − δ_P` the change of the law of the density from the density going to zero to `p = 1`; a factor below one means that full density lowers the odds of an error. The interval is `e^(−P̂(1) ∓ h)`, `h` the half-width of `P̂(1)`.

![P_sync: S before correction](figures/P_sync/S-before-correction.png)

![P_sync: S after correction](figures/P_sync/S-after-correction.png)

![P_sync: plateau before correction](figures/P_sync/plateau-before-correction.png)

![P_sync: P before correction](figures/P_sync/P-before-correction.png)

![P_sync: P after correction](figures/P_sync/P-after-correction.png)

![P_sync: M before correction](figures/P_sync/M-before-correction.png)

![P_sync: M after correction](figures/P_sync/M-after-correction.png)

## H: a rule word on its place is replaced correctly

Model VS-PMR, request weights by the cube V × P × M. Fit: GLMM with heterogeneous random-intercept variance, that is with the shift of a request whose variance depends on the load; the variance of the request shift `σ² = 5.71`; 6,849 requests, 3,805,051 trials, 253,040 failures, 92 parameters; removed bins: vs_1_1, vs_1_2, vs_2_1, vs_2_2, vs_2_3, vs_3_1, vs_3_2, vs_3_3, vs_3_4, vs_4_1, vs_4_2, vs_4_3.

The variance of the shift depends on the load: `ln σ_j = z_j·γ`, 24 parameters more than one variance, `2·Δ ln L = 1064.4`. `σ²` above is the mean over the requests. A group is taken with the other two axes averaged over the requests.

| Axis | Centre of the group | Share of the requests | σ² |
|---|---|---|---|
| text length, words | 234.1 | 0.08 | 2.54 ± 0.88 |
| text length, words | 531.8 | 0.08 | 2.87 ± 0.67 |
| text length, words | 1021 | 0.10 | 3.03 ± 0.65 |
| text length, words | 2059 | 0.11 | 3.08 ± 0.56 |
| text length, words | 3932 | 0.16 | 3.97 ± 0.70 |
| text length, words | 8018 | 0.21 | 3.28 ± 0.51 |
| text length, words | 1.635e+04 | 0.26 | 2.70 ± 0.46 |
| density | 0.0008488 | 0.06 | 2.40 ± 0.98 |
| density | 0.002711 | 0.05 | 2.10 ± 1.23 |
| density | 0.005546 | 0.06 | 2.30 ± 0.98 |
| density | 0.01068 | 0.04 | 1.79 ± 0.78 |
| density | 0.02017 | 0.05 | 2.01 ± 0.52 |
| density | 0.03973 | 0.05 | 1.68 ± 0.43 |
| density | 0.07611 | 0.07 | 2.03 ± 0.51 |
| density | 0.1538 | 0.12 | 1.93 ± 0.35 |
| density | 0.3054 | 0.17 | 3.22 ± 0.56 |
| density | 0.6159 | 0.19 | 4.70 ± 0.84 |
| density | 1 | 0.13 | 7.48 ± 1.60 |
| number of rules | 1 | 0.13 | 21.87 ± 12.21 |
| number of rules | 2 | 0.11 | 6.66 ± 1.48 |
| number of rules | 3.635 | 0.14 | 5.76 ± 1.06 |
| number of rules | 6.801 | 0.13 | 6.00 ± 1.19 |
| number of rules | 12.87 | 0.10 | 4.53 ± 1.00 |
| number of rules | 26.74 | 0.12 | 2.81 ± 0.58 |
| number of rules | 53.09 | 0.10 | 0.82 ± 0.18 |
| number of rules | 106.1 | 0.11 | 0.27 ± 0.07 |
| number of rules | 195.8 | 0.05 | 0.16 ± 0.05 |

The approximations through the groups, by the residuals in `ln σ²`, serve the eye.

| Axis | Approximation | Parameters | χ² |
|---|---|---|---|
| text length, words | `a + b·log₂ v` | `a = 2.77`, `b = 0.031` | 12.0 over 7 groups |
| density | `a + b·log₂ p` | `a = 3.71`, `b = 0.239` | 105.3 over 11 groups |
| number of rules | `c + A/(1 + m/m₀)` | `c = 0.00`, `A = 13.33`, `m₀ = 3.91` | 83.4 over 9 groups |

The law of the grid: a length slice is a binary exponential, `μ − A·2^(γ·ℓ)` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `A = 0.733 ± 0.399`, `gamma = 0.195 ± 0.046`; `χ² = 7.5` over 65 cells; chosen by the smallest `QAIC` among 3 forms. The law of the levels runs through the levels of the slices.

The law of the levels of the slices is a line, `a + b·log₂(v/20)`: `a = 20.74 ± 2.54`, `b = −0.68 ± 0.15`; `χ² = 4.6` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a two-slope hyperbola, `μ + β·(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + (κ/2)·[(log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*) + √((log₂(p/2⁻¹⁰) − log₂(p/2⁻¹⁰)*)² + ω²)]`: `mu = 3.20 ± 5.70`, `kappa = −1.34 ± 1.97`, `corner = 3.31 ± 4.36`, `omega = 3.18 ± 4.90`, `beta = 0.84 ± 1.71`; `χ² = 5.4` over 11 bins; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −3.80 ± 2.20`, `b = −0.56 ± 0.06`; `χ² = 5.1` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the rule number is a slanted hyperbola, `μ + (κ/2)·[(log₂ r − log₂ r*) + √((log₂ r − log₂ r*)² + ω²)]`: `mu = 0.01 ± 0.23`, `kappa = −0.75 ± 0.04`, `corner = 2.74 ± 0.25`, `omega = 0.17 ± 0.59`; `χ² = 3.6` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = μ₀ + μ_P + λ_M + λ_R = 16.50 ± 1.76`. The loss of the density at `p = 1` is 0.16 logits.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 23.5 (20.9–26.4) | 20.5 |
| fatigue capacity, words | 0.5 | 1.7·10⁵ (93,307–3.1·10⁵), continuation of the law | - |
| how many times full density multiplies the odds of an error | - | 1.2 (0.4–3.6) | - |

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

Model V-(s/v)-PM, request weights by the cube V × P × M. Fit: with the shift of a request; the variance of the request shift `σ² = 3.86`; 5,860 requests, 36,982,696 trials, 158,936 failures, 34 parameters; removed bins: v_bin_1, v_bin_2, v_bin_3, v_bin_4. The variance by the load turned bins against the median request of the data, so the quantity keeps one variance: the variance by the load turns bins against the median request: p_bin_2: passport -2.45, median request +0.23 ± 0.23.

The laws run through the bins without the poorly determined ones; a bin weighs `1/(σᵢ² + τ²)`, its own variance and the scatter of the bins around the law, and the errors are carried by the full covariance of the bins.

The law of the length is a line, `a + b·log₂(v/20)`: `a = 13.55 ± 1.18`, `b = −0.37 ± 0.12`; `χ² = 5.6` over 7 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the share of the answer is a line, `a + b·q`: `a = 0.15 ± 0.05`, `b = −0.17 ± 0.07`; `χ² = 8.2` over 9 bins; chosen by the smallest `QAIC` among 2 forms. A fall of 0.17 over the whole answer.

The law of the density is a logarithm plus a line, `a + b·log₂(p/2⁻¹⁰) + c·0.0009765625·2^(log₂(p/2⁻¹⁰))`: `a = −1.15 ± 0.35`, `b = 0.36 ± 0.07`, `c = −7.01 ± 0.98`; `χ² = 4.3` over 10 bins; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 0.00 ± 0.62`, `b = −0.94 ± 0.04`; `χ² = 4.3` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α_F = V(20) + q(0) + δ_P + M(1) = 14.26 ± 1.01`, where `δ_P` is the maximum of the law of the density, at `p = 0.0732`; `V(20)` = 13.55 ± 1.18, `q(0)` = 0.15 ± 0.05, `δ_P` = 0.55 ± 0.25, `M(1)` = 0.00 ± 0.62. The edge of the measured area: 256 rules and 20,480 words.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 35,094 (14,636–84,144), continuation of the law | at least 256, reserve left 7.54 |
| fatigue capacity, words | 0.5 | 5.1·10¹² (8.5·10⁹–3.1·10¹⁵), continuation of the law | - |
| how many times full density multiplies the odds of an error | - | 31.4 (16.4–60.0), conditional: a continuation of the law of the density | - |

The instruction capacity is the number of rules at the zero point. The fatigue capacity is the number of the last word of a text of the same length at the zero point. Full density multiplies the odds of a misfire by `e^(−P̂(1))`, `P̂(1) = P(1) − P(p → 0)`; a factor below one would mean that full density lowers them. It is a conditional quantity: a text has no background words at `p = 1`, so it always continues the law of the density, and it allows comparing the effect of the density on `P_sync`, `H` and `F`.

![F: V before correction](figures/F/V-before-correction.png)

![F: V after correction](figures/F/V-after-correction.png)

![F: answer share before correction](figures/F/answer-share-before-correction.png)

![F: answer share after correction](figures/F/answer-share-after-correction.png)

![F: P before correction](figures/F/P-before-correction.png)

![F: P after correction](figures/F/P-after-correction.png)

![F: M before correction](figures/F/M-before-correction.png)

![F: M after correction](figures/F/M-after-correction.png)
