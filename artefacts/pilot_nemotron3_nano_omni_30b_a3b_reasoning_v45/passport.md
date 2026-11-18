# Passport of `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`

Measurement `pilot_nemotron3_nano_omni_30b_a3b_reasoning_v45`, 3,695 answered points. The method is described in `docs/05-passport-model.md` of the repository.

## The formula

`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`. The formula is the way to combine the three quantities: it has no law and no capacity of its own. `π = m/D` is the replacement density. The zero point of a capacity is a text of 20 words, the start of the answer, the density at the maximum of its law and one rule; every other axis of the capacity is set by its definition.

The capacities are taken at the level 0.5. At it the share of successes over all requests and the success of the median request reach one half at the same logit, zero, whatever the variance of the request shift: a capacity is the load at which the laws bring the logit of the median request to zero, and it does not depend on that variance.

## P_sync: the answer word stands against its reference word

Model VS-PM, request weights by the cube V × P × M. Fit: with the shift of a request; the variance of the request shift `σ² = 35.80`; 3,695 requests, 43,142,085 trials, 34,600,867 failures, 96 parameters; removed bins: none. The variance by the load turned bins against the median request of the data, so the quantity keeps one variance: the variance by the load turns bins against the median request: m_bin_4: passport -0.65, median request +1.35 ± 0.98; m_bin_5: passport -0.97, median request +1.47 ± 1.07.

The law of the grid: a length slice is a two-asymptote hyperbola, `[(κ·ℓ + λ) + μ − √((κ·ℓ + λ − μ)² + κ²·ω²)]/2` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −2.725 ± 0.212`, `lambda = 26.463 ± 2.015`, `omega = 0.001 ± 0.975`; `χ² = 67.3` over 77 cells; chosen by the smallest `QAIC` among 3 forms. The law of the plateau runs through the levels of the slices.

The law of the plateau is a line, `a + b·log₂(v/20)`: `a = 7.93 ± 0.81`, `b = 0.07 ± 0.18`; `χ² = 10.3` over 11 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a binary exponential, `μ − A·2^(γ·log₂(p/2⁻¹⁰))`: `A = 4.26 ± 2.95`, `gamma = 0.20 ± 0.07`, `mu = 12.83 ± 4.14`; `χ² = 7.0` over 11 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 1.12 ± 0.75`, `b = −0.22 ± 0.17`; `χ² = 4.7` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = δ_VS + δ_P + δ_M = 21.88 ± 4.20` is the logit at the density going to zero and one rule, and that density is itself a continuation of the law.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 1.9·10³⁰ (8,035–4.4·10⁵⁶), continuation of the law | at least 256, reserve left 20.19 |
| fatigue capacity, words | 0.5 | 29,156 (12,520–67,900), continuation of the law | - |
| how many times full density multiplies the odds of an error | - | 2.8·10⁷ (4.3·10⁵–1.8·10⁹) | - |

The instruction capacity is the number of rules at which the law of the number of rules brings the logit to zero, and next to it the walk along the bins of that axis; by the law it is none when the fall of that law over its bins does not differ from zero. The fatigue capacity is the word at which the full law of the grid at the density going to zero and one rule reaches zero, on a text of the same length: by the curve of the law, as for `H`. Full density multiplies the odds of an error by `e^(−P̂(1))`, `P̂(1) = P(1) − δ_P` the change of the law of the density from the density going to zero to `p = 1`; a factor below one means that full density lowers the odds of an error. The interval is `e^(−P̂(1) ∓ h)`, `h` the half-width of `P̂(1)`.

![P_sync: S before correction](figures/P_sync/S-before-correction.png)

![P_sync: S after correction](figures/P_sync/S-after-correction.png)

![P_sync: plateau before correction](figures/P_sync/plateau-before-correction.png)

![P_sync: P before correction](figures/P_sync/P-before-correction.png)

![P_sync: P after correction](figures/P_sync/P-after-correction.png)

![P_sync: M before correction](figures/P_sync/M-before-correction.png)

![P_sync: M after correction](figures/P_sync/M-after-correction.png)

## H: a rule word on its place is replaced correctly

Model VS-PMR, request weights by the cube V × P × M. Fit: with the shift of a request; the variance of the request shift `σ² = 1.05`; 3,656 requests, 829,043 trials, 713,686 failures, 103 parameters; removed bins: vs_11_11. The variance by the load turned bins against the median request of the data, so the quantity keeps one variance: the variance by the load turns bins against the median request: m_bin_3: passport -0.19, median request +0.74 ± 0.66.

The law of the grid: a length slice is a two-asymptote hyperbola, `[(κ·ℓ + λ) + μ − √((κ·ℓ + λ − μ)² + κ²·ω²)]/2` over `ℓ = log₂ s`, with an own level `μ` in every slice and the other parameters shared, in the coding of the fit: `kappa = −14.906 ± 43.771`, `lambda = 270.159 ± 861.547`, `omega = 4.987 ± 5.792`; `χ² = 78.4` over 76 cells; chosen by the smallest `QAIC` among 3 forms. The law of the levels runs through the levels of the slices.

The law of the levels of the slices is a line, `a + b·log₂(v/20)`: `a = 9.90 ± 1.87`, `b = −0.39 ± 0.07`; `χ² = 0.1` over 11 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the density is a binary exponential, `μ − A·2^(γ·log₂(p/2⁻¹⁰))`: `A = 0.22 ± 0.13`, `gamma = 0.43 ± 0.07`, `mu = 2.70 ± 0.43`; `χ² = 5.0` over 11 bins; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = 0.52 ± 0.28`, `b = 0.06 ± 0.03`; `χ² = 10.3` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the rule number is a binary exponential, `μ − A·2^(γ·log₂ r)`: `A = 0.08 ± 0.04`, `gamma = 0.65 ± 0.10`, `mu = −0.16 ± 0.19`; `χ² = 1.2` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α = μ₀ + μ_P + λ_M + λ_R = 12.89 ± 1.81`. The loss of the density at `p = 1` is 4.24 logits.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 34.3 (29.8–39.4) | 35.0 |
| fatigue capacity, words | 0.5 | 842 (716–990) | - |
| how many times full density multiplies the odds of an error | - | 69.4 (45.2–106) | - |

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

The laws run through the bins without the poorly determined ones; a bin weighs `1/(σᵢ² + τ²)`, its own variance and the scatter of the bins around the law, and the errors are carried by the full covariance of the bins.

The law of the length is a binary exponential, `μ − A·2^(γ·log₂(v/20))`: `A = −1.01 ± 1.41`, `gamma = 0.25 ± 0.15`, `mu = 5.90 ± 2.20`; `χ² = 21.6` over 11 bins; chosen by the smallest `QAIC` among 3 forms.

The law of the share of the answer is an exponential, `c + A·e^(−q/τ)`: `A = −3.68 ± 0.40`, `c = 0.03 ± 0.22`, `tau = 0.13 ± 0.03`; `χ² = 3.2` over 9 bins; chosen by the smallest `QAIC` among 2 forms. A fall of −3.68 over the whole answer.

The law of the density is a logarithm plus a line, `a + b·log₂(p/2⁻¹⁰) + c·0.0009765625·2^(log₂(p/2⁻¹⁰))`: `a = −0.18 ± 0.53`, `b = 0.11 ± 0.08`, `c = −2.01 ± 0.80`; `χ² = 7.7` over 10 bins; chosen by the smallest `QAIC` among 4 forms.

The law of the number of rules is a line, `a + b·log₂ m`: `a = −0.01 ± 0.39`, `b = −0.25 ± 0.07`; `χ² = 7.7` over 8 bins; chosen by the smallest `QAIC` among 3 forms.

The level `α_F = V(20) + q(0) + δ_P + M(1) = 3.58 ± 0.89`, where `δ_P` is the maximum of the law of the density, at `p = 0.0760`; `M(1)` = −0.01 ± 0.39, `V(20)` = 6.91 ± 0.88, `δ_P` = 0.34 ± 0.27, `q(0)` = −3.65 ± 0.41. The edge of the measured area: 256 rules and 20,480 words.

| Capacity | q | By the law | By the bins |
|---|---|---|---|
| instruction capacity, rules | 0.5 | 18,377 (474–7.1·10⁵), continuation of the law | at least 256, reserve left 1.74 |
| fatigue capacity, words | 0.5 | none: the law gives no root | - |
| how many times full density multiplies the odds of an error | - | 2.6 (1.3–5.3), conditional: a continuation of the law of the density | - |

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
