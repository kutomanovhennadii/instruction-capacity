# The passport model

The passport of a measurement is computed from its published points alone. This document gives
every step; `scripts/build_passport.py` recomputes a published passport, and
`scripts/verify.py` compares it with the published one.

## 1. Trials

The validator ([answer validation](04-answer-validation.md)) turns the answer of every point with
the status `ok` into trials. A trial carries the conditions of its point (`D`, `n`, `m`), the
number `r` of its rule, the position `s` of its word in the answer and its score `z`.

## 2. The area of analysis

Not every measured point enters the fit: where the model collapses completely, or where the task
does not fit the model, a cell carries no information on capacity and would distort the fit.

The **purchase cube** is a grid of cells `D × m × n` whose axes are cut by doubling from the start
step `(D, n, m) = (4, 5, 1)` (the `m` axis from 1) up to the largest `D`, `m` and `n` among the
executed points (status `ok` or `na`). The **representative** of a cell is its upper corner, a
node `(D, n, m)`. Its state is

- `na` if the provider refused it at least three times with no `ok` or `na` point, or if it has no
  `ok` point;
- no state if it has fewer than three points with the status `ok` or `na`;
- **collapse** if every `ok` point of it has a third-kind collapse;
- **open** otherwise.

A cell is a **boundary** cell if its representative collapsed or is `na`, and **unreachable** if
its representative has no state or its task cannot be built (`m > D`, `D + m` larger than the
vocabulary, the limits of the model). The fit takes the points of the remaining **ordinary**
cells. The passport prints the filling of the cube ("Bin cube filling").

## 3. Bins

The model has a free coefficient per bin. The bins double:

| Axis | Bins |
|---|---|
| text length `v = D·n` | 1–20, 21–40, 41–80, … up to the bin of the longest text |
| word position `s` in the answer | by the answer line of ten words: lines 1–2, 3–4, 5–8, …, that is words 1–20, 21–40, 41–80, …; not beyond the last line of the longest text of the length bin |
| density `p = m/D` | by `k = ⌈D/m⌉` in the bins {1}, {2}, {3–4}, {5–8}, …: `p = 1`, `p ∈ [1/2, 1)`, `[1/4, 1/2)`, … down to the sparsest level |
| rule count `m` | {1}, {2}, {3–4}, {5–8}, … up to the bin of the largest `m` |
| rule number `r` | the same bins as `m` |

A length bin and a position bin form a **cell of the grid** `V·S`.

## 4. The model VS-PMR (`vspmr_glmm_v1`)

The logit of a right replacement in trial $i$ of point $j$ is

$$\eta_{ij} = \log\frac{\pi_{ij}}{1-\pi_{ij}} = VS(v, s) + P(p) + M(m) + R(r) + b_j, \qquad b_j \sim N(0, \sigma^2),$$

where $VS(v, s)$ is the free coefficient of the grid cell of the trial (length bin $V$, position
bin $S$), and $P$, $M$, $R$ are the free coefficients of its density, rule-count and rule-number
bins. In matrix form $\eta_{ij} = x_{ij}\theta + b_j$: $x_{ij}$ is the design row of the trial
(ones in its bins), $\theta$ the coefficients. $b_j$ is the **shift of the point**: one number
for all trials of an answer, the difficulty of that particular request; $\sigma^2$ is the
variance of the shifts. The number of rules $m$ and the length $D$ are the same for all trials
of a point, so the bins $M$ and $P$ are estimated by comparing points with each other, and
without the shift in the model the spread of point difficulty would land in them. The bins
describe the **median point** ($b = 0$). There is no separate intercept: the grid cells carry
the level. Three bins are reference bins with an implicit zero: the density bin of the start
step `(4, 5, 1)` ($p \in [1/4, 1/2)$), the rule-count bin {1} and the rule-number bin {1}.

**Exclusion.** A bin in which every replacement is right, or every replacement is wrong, has no
finite maximum-likelihood estimate. Such a bin is removed together with its trials, and the pass
repeats until nothing changes. Removed bins are published with the status `excluded` and no
number.

## 5. Weights

Points are not spread evenly: some cells were measured densely, others sparsely. Each point gets a
weight inversely proportional to the number of points in its cell of the **run cube**, length bin
× density bin × rule-count bin, normalized to a mean of 1. The passport reports the effective
number of points

$$n_{\mathrm{eff}} = \frac{\left(\sum w\right)^2}{\sum w^2}.$$

## 6. The fit with the shift of a point

The shifts are not estimated as free parameters: the likelihood of every point is averaged over
its shift,

$$L_j(\theta, \sigma^2) = \int \prod_i \pi_{ij}(b)^{z_{ij}}\big(1 - \pi_{ij}(b)\big)^{1 - z_{ij}}\,\varphi(b; 0, \sigma^2)\,db,$$

and $\theta$ and $\sigma^2$ maximize the weighted marginal likelihood $\sum_j w_j \ln L_j$ jointly
($w_j$ is the weight of the point, section 5). The integral is computed by adaptive Gauss-Hermite
quadrature: for every point the mode $\hat b_j$ of its shift and the curvature $s_j$ of the log
posterior are found, and the integral becomes a weighted sum over 15 nodes
$b = \hat b_j + \sqrt{2}\,s_j z_k$, placed anew at every evaluation. The maximum is found by
Newton's method, the second derivatives by the Louis identity; the start is the plain weighted
logit of the same trials (Newton's method on collapsed rows: trials with equal design rows are
merged with summed counts) and $\sigma^2 = 1$. The covariance is the cluster sandwich with one
point per cluster around the Hessian of the marginal likelihood; it covers $\theta$ and
$\sigma^2$, so $\hat\sigma^2$ comes with its standard error and its covariance with the
coefficients. When the data show no spread between points, $\sigma^2$ goes to its floor
$10^{-8}$ and stays there, and the coefficients are those of the plain logit. If a fit does not
converge, there is no passport: the command reports the reason and stops; there is no fallback
fit.

## 7. The second stage: laws through the bins

The bins of every block are summarized by a law of its axis:

$$VS(v, s) = \tfrac{1}{2}\Big[(\lambda_S + \kappa_S\ell) + \mu(v) - \sqrt{(\lambda_S + \kappa_S\ell - \mu(v))^2 + \kappa_S^2\omega_S^2}\Big], \qquad \ell = \log_2 s,$$

$$\mu(v) = \mu_0 + \tfrac{\kappa_V}{2}\Big[\log_2\tfrac{v}{v_{*}} + \sqrt{\log_2^2\tfrac{v}{v_{*}} + \omega_V^2}\Big],$$

$$P(p) = \mu_P - A_P\big(1 - e^{-p/p_c}\big), \qquad M(m) = \lambda_M + \kappa_M \log_2 m, \qquad R(r) = \lambda_R - A_R\,(r^{\gamma} - 1).$$

The grid law is a smooth corner between two regimes: a **plateau** $\mu(v)$ at the early positions
of the answer and a slanted **asymptote** $\lambda_S + \kappa_S\ell$, the fatigue along the answer;
$\omega_S$ is the width of the corner. The plateau itself is a hyperbola in the text length: a
horizontal asymptote $\mu_0$ for short texts and a slanted one of slope $\kappa_V$ per doubling for
long texts, which cross at the length $v_*$, with the width $\omega_V$ of the corner.

**The grid law in two steps.** First, every length slice gets its own plateau level $\mu_v$, with
$\kappa_S$, $\lambda_S$ and $\omega_S$ shared. This fit is weighted by the own variances of the
cells: the cells of a slice are strongly correlated, and weights from the full covariance drove the
free plateau of a slice above all its cells. A slice enters only if its plateau is computable from
its points: at least two measured cells, at least one of them on the plateau. Second, the
hyperbola $\mu(v)$ is drawn through the plateaus of all determined slices (at least five), weighted
by their own variances. The errors of both steps are carried by the full covariance of the fit. A
corner parameter that the data push to the edge of its range ($v_*$ from 20 words up to the
longest determined slice; $\omega_V$ from 0.001 up to the doublings from 20 words to that slice)
is **held** there: its value is the bound, it has no error, and the law names it in `held`. The
density, rule-count and rule-number laws are fitted by generalized least squares, weighted by the
covariance of their bins. The abscissa of a bin is the centre of mass of its trials with the fit
weights.

**Zero point and shifts.** Every law has a value at the zero of its axis, the **shift** of its
block. At vanishing load the grid tends to the horizontal asymptote of its plateau - at the start
of the answer ($s \to 0$) of short texts ($v \to 0$) - so its shift is $\delta_{VS} = \mu_0$;
$\lambda_S$ and $\mu_v$ are given relative to it, and $\mu_0$ itself is not published. For the
density the zero is $p \to 0$, for the rule count $m = 1$, for the rule number $r = 1$. The shifts
are $\delta_{VS}$, $\mu_P$, $\lambda_M$ and $\lambda_R$. They are moved into one level,

$$\alpha_{\mathrm{phys}} = \delta_{VS} + \mu_P + \lambda_M + \lambda_R,$$

and the bins are published **restored**: each minus the shift of its block. The prediction of a
trial does not change. The restored bins and laws read as costs relative to the zero point, and
$\alpha_{\mathrm{phys}}$ is the logit of a right replacement at the zero point.

## 8. Public quantities

They are computed from $\alpha_{\mathrm{phys}}$, the restored bins and the laws, at the levels
$q = 0.95$ and $q = 0.5$, with standard errors by the delta method on the full covariance of the
fit, extended by $\sigma^2$.

**The threshold of a level.** A capacity of the level $q$ is the load at which the **share of
right replacements over all points** is still not below $q$. The bins describe the median point;
the share over all points at the logit $\eta$ of the median point is

$$\bar\pi(\eta) = E_b\Big[\frac{1}{1 + e^{-(\eta + b)}}\Big], \qquad b \sim N(0, \sigma^2).$$

The shift does not depend on the load, so $\bar\pi = q$ holds exactly at $\eta = \eta^*$, where the
**threshold** $\eta^*(q, \sigma^2)$ is the root of $\bar\pi(\eta^*) = q$ (computed with 64
Gauss-Hermite nodes; at $\sigma^2 = 0$ it is $\operatorname{logit} q$). The error of $\hat\sigma^2$
enters a capacity through $d\eta^*/d\sigma^2$. The reserve of the level $q$ is
$\Delta_q = \alpha_{\mathrm{phys}} - \eta^*(q, \sigma^2)$, the margin of the zero point above the
threshold. All capacities except the IGC are therefore read at the zero point: the start of the
answer to a short text with sparse replacements; longer texts, later positions and denser tables
spend the same reserve. The IGC takes the density at p = 1 and the other axes at the zero point.

| Quantity | Formula | Unit |
|---|---|---|
| IEC, binned | the largest `m` with `C(m) = Σ_{r=2..m} a(r) ≤ Δ_q`, plus the fraction of the next rule; the price of a rule is `a(r) = −ΔM(r) − R(r)`, where `ΔM(r)` is the step of the restored rule-count profile per rule | rules |
| IEC, by law | the root of `C(m) = Δ_q` with `C(m) = −κ_M log₂ m + A_R((m^(γ+1) − 1)/(γ + 1) − (m − 1))` | rules |
| IGC, binned | the IEC walk with the restored bin of full density in the cumulative price: the largest `m` with `C(m) − P(1) ≤ Δ_q`, plus the fraction of the next rule; `P(1)` is the restored coefficient of the density bin `p = 1` | rules |
| IGC, by law | the root of `C(m) + A_P(1 − e^(−1/p_c)) = Δ_q`, with `C(m)` of the IEC by law | rules |
| price of a position doubling | `−κ_S` | logit per doubling |
| FEC, by law | `s` where the asymptote reaches the level: `log₂s = (η*(q, σ²) − α_phys − λ_S)/κ_S` | words |
| price of a volume doubling | `−κ_V` | logit per doubling |
| loss at full density | `A_P(1 − e^(−1/p_c))` | logit |
| late-rule degradation factor | `2^γ` | times per doubling |
| late-rule degradation scale | `A_R` | logit |
| rule-set discount per doubling | `κ_M` | logit per doubling |

The IEC is computed only inside the measured support of the rule bins, up to the upper edge of
the last rule-count bin; beyond it the passport gives the support as a lower bound. A quantity
that lies beyond the measured texts is marked `extrapolated`; the late-rule terms and the
rule-set discount are marked `not_detected` when they are within 1.96 standard errors of zero.
The binned and the law calculations of the IEC and the IGC are published side by side: their
gap measures the approximation. The IGC (Instructional Guaranteed Capacity) is the IEC at
full density: at p = 1 every rule brings its own word (D = m), the density stays at one as
rules are added, and the loss of full density enters the cumulative price once, as a term
constant in `m`. The FEC (Fatigue Execution Capacity) has the law calculation only: the word
position has no bins of its own, every cell of the grid belongs to its length slice, and a
crossing read from the cells would carry the error of the slices it is read from.

**Diagnostics.** The passport also reports $\hat\sigma^2$ with its standard error and the
within-point correlation $\hat\rho = \hat\sigma^2/(\hat\sigma^2 + \pi^2/3)$, both from the fit,
the plateau by length slice, the deviation of every restored bin from its law in units of its standard error, and the
canonical correlation of the increments of neighbouring bins of every two components: a
correlation near one means that the fit cannot tell which of the two components explains an
observation.

## 9. Verification

`scripts/verify.py` recomputes the passport and compares it with the published one: the
composition exactly (keys, list lengths, strings, integers, statuses); floating-point numbers
outside the second stage within $10^{-6}$ relative; numbers of the second stage that have their
own standard error within $0.01\,\mathrm{SE}$; other numbers of the second stage within 10%
relative. The tolerances keep a margin of 3 to 600 times over the spread that a mere reordering
of the summation produced on the published measurements: the second stage contains nonlinear
fits that stop at the tolerance of their optimizer and, for a degenerate law, anywhere along a
flat valley.
