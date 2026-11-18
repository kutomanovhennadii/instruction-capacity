# The passport model

The passport of a measurement is computed from its published points alone. This document gives
every step; `scripts/build_passport.py` recomputes a published passport, and
`scripts/verify.py` compares it with the published one.

## 1. Trials

The validator ([answer validation](04-answer-validation.md)) aligns the answer of every point with
the status `ok` with its reference and gives the trials of the three quantities: `P_sync` on the
places of the alignment, `H` on the rule words standing on their places, `F` on the background
words standing on their places. All answered points enter the fit: the alignment gives an outcome
at every place of an answer, so no point is taken out of the analysis because it lost its text. The
trials of a point with the same conditions are folded into one row with summed counts.

## 2. Bins

A model has a free coefficient per bin. The bins double:

| Axis | Bins |
|---|---|
| text length `v = D·n` | 1–20, 21–40, 41–80, … up to the bin of the longest text |
| word number `s` | by the answer line of ten words: lines 1, 2, 3–4, 5–8, …, that is words 1–10, 11–20, 21–40, 41–80, …; not beyond the last line of the longest text of the length bin |
| density `p = m/D` | by `k = ⌈D/m⌉` in the bins {1}, {2}, {3–4}, {5–8}, …: `p = 1`, `p ∈ [1/2, 1)`, `[1/4, 1/2)`, … down to the sparsest level |
| rule count `m` | {1}, {2}, {3–4}, {5–8}, … up to the bin of the largest `m` |
| rule number `r` | the same bins as `m` |
| share of the answer `q = s/v` | ten equal bins, for `F` only |

A length bin and a word-number bin form a **cell of the grid** `V·S`.

## 3. The three models

The logit of a success in trial $i$ of point $j$ is

$$\begin{aligned}
P_{\mathrm{sync}}:&\quad \eta_{ij} = VS(v, s) + P(p) + M(m) + b_j,\\
H:&\quad \eta_{ij} = VS(v, s) + P(p) + M(m) + R(r) + b_j,\\
F:&\quad \eta_{ij} = V(v) + Q(s/v) + P(p) + M(m) + b_j,
\end{aligned}
\qquad b_j \sim N(0, \sigma_j^2).$$

For `F` a success is the absence of a misfire, so $\eta$ is the logit of $1 - F$. In every model a
more negative coefficient means more errors. $VS$ is the free coefficient of a cell of the grid: the
contributions of the text length and of the word number do not add up, because doubling the text
costs much at the start of the answer and almost nothing deep in it. $P$, $M$, $R$, $V$ and $Q$ are
the free coefficients of the bins of their axes. Only a rule word has a rule number, so only `H` has
`R`.
For `F` the deviation of a grid cell from the sum of its axes is small: the slices of the length do not
differ in shape along the answer. So `F` has the length and the share of the answer as two separate axes.

$b_j$ is the **shift of the request**: one number for all trials of an answer, the difficulty of that
particular request; $\sigma_j^2$ is its variance, which depends on the load of the request: the model is a
GLMM with heterogeneous random-intercept variance (section 5).
The conditions of a request are the same for all its trials, so the bins of $M$ and $P$ are estimated by comparing requests with each
other, and without the shift in the model the spread of request difficulty would land in them. The
bins describe the **median request** ($b = 0$). There is no separate intercept: the cells of the grid,
and the length bins in `F`, carry the level. The other axes have reference bins with an implicit
zero: the density bin of the start step `(D, n, m) = (4, 5, 1)` ($p \in [1/4, 1/2)$), the rule-count
bin {1}, the rule-number bin {1} and the last share bin.

**Exclusion.** A bin in which every trial has one outcome has no finite maximum-likelihood estimate.
Such a bin is removed together with its trials, and the pass repeats until nothing changes. Removed
bins are named in the section of the quantity.

## 4. Weights

Points are not spread evenly: some cells were measured densely, others sparsely. Each point gets a
weight inversely proportional to the number of points in its cell of the **run cube**, length bin
× density bin × rule-count bin, normalized to a mean of 1. All three quantities use the same weights.
The passport reports the effective number of points

$$n_{\mathrm{eff}} = \frac{\left(\sum w\right)^2}{\sum w^2}.$$

## 5. The fit with the shift of a request

The shifts are not estimated as free parameters: the likelihood of every point is averaged over
its shift,

$$L_j(\theta, \sigma^2) = \int \prod_i \pi_{ij}(b)^{z_{ij}}\big(1 - \pi_{ij}(b)\big)^{1 - z_{ij}}\,\varphi(b; 0, \sigma^2)\,db,$$

and $\theta$ and $\sigma^2$ maximize the weighted marginal likelihood $\sum_j w_j \ln L_j$ jointly
($w_j$ is the weight of the point, section 4). The integral is computed by adaptive Gauss-Hermite
quadrature: for every point the mode $\hat b_j$ of its shift and the curvature $s_j$ of the log
posterior are found, and the integral becomes a weighted sum over 15 nodes
$b = \hat b_j + \sqrt{2}\,s_j z_k$, placed anew at every evaluation. The maximum is found by
Newton's method on the total derivative of this sum: the nodes move with the parameters, and their
motion enters the gradient and the second derivatives; where the matrix of the second derivatives is
not negative definite, the step takes the moduli of its eigenvalues. The start is the plain weighted
logit of the same rows and $\sigma^2 = 1$. A full step from a variance far below the maximum can
overshoot it many times over and leave the fit crawling along a flat slope, so a fit that does not
converge is started again from $\sigma^2 = 4$, $16$ and $64$, and the first that converges is taken.
On the published measurement these starts reach the same maximum. The covariance is the cluster
sandwich with one point per cluster around the Hessian of the marginal likelihood; it covers
$\theta$ and $\sigma^2$, so $\hat\sigma^2$ comes with its error and its covariance with the
coefficients. When the data show no spread between points, $\sigma^2$ goes to its floor $10^{-8}$
and stays there.

**The spread by the load.** The fit above takes one variance for all requests. It is the first step: the data
reject one variance, so the second step fits the GLMM with heterogeneous random-intercept variance: the log of the
standard deviation of the shift of a request is linear in the groups of its load, the log-linear model that
[Hedeker, Mermelstein, Demirtas 2008] give the between-subject variance:

$$b_j \sim N(0, \sigma_j^2), \qquad \ln\sigma_j = z_j \gamma.$$

The row $z_j$ holds a one for the level and the free bins of the text length, the density and the number of rules of
the request. A rare edge of an axis is merged with its neighbour until the group has 30 requests with failures, and the
most populated group of an axis is its reference and has no column. $\theta$ and $\gamma$ maximize the same weighted
marginal likelihood from the estimates of the first step, by the same quadrature and Newton steps, and the covariance is
the same sandwich over $\theta$ and $\gamma$. A group whose standard deviation the data push to the lower bound
$\ln\sigma = \ln 10^{-4}$ has no measured spread: it is merged with its neighbour on the axis and the fit is made again,
and the passport notes the merge (`SPREAD_MERGED`). The bins of the logit stay as they were. The variance $\sigma^2$ of the
section of a quantity is then the mean of $\sigma_j^2$ over the requests with their weights. The section prints the
variance by the groups of every axis, with the other two axes averaged over the requests, and the gain $2\Delta\ln L$
against one variance.

Through the groups of every axis the section draws an approximation, fitted by the residuals in $\ln\sigma^2$: the residual of
a group is $(\ln\sigma^2 - \ln a(x))/s$ with $s = h/(1.96\,\sigma^2)$, where $a(x)$ is the approximation and $h$ the half-width
of the group, so every group weighs by its relative precision. The form is chosen for every axis of every quantity: a line in
$\log_2$ of the axis, $a + b\log_2 x$, or the fractional hyperbola $c + A/(1 + x/x_0)$.

| Quantity | Text length | Density | Number of rules |
|---|---|---|---|
| `P_sync` | line | line | hyperbola |
| `H` | line | line | hyperbola |
| `F` | hyperbola | line | hyperbola |

The approximations serve the eye. An axis with no more groups
than the parameters of its form has no approximation. A hyperbola whose $x_0$ stops at the lower bound of its search has
degenerated into the inverse law $c + A/x$: its $A$ and $x_0$ mean nothing apart, so the section fits the inverse law anew and
gives it in place of the hyperbola.

**The check against the median request.** A bin is the logit of the median request of the model, and where the
variance changes along an axis, the bin can lie on the other side of the reference bin than the requests themselves.
Every bin of an axis constant within a request — the density, the number of rules and, for `F`, the text length — is
therefore compared with the median request of the data. The requests with at least 50 trials of the quantity are grouped
into the cells of the other two axes. In every cell the median shares of failures $s$ of the requests of the bin and of
the reference bin go to the logit of the odds of a success, $\ln\big((1 - s)/s\big)$ with $s$ kept inside
$[0.005, 0.995]$, and their difference is averaged over the cells with the weight of the number of the requests of the
reference bin. Its error comes from 200 resamplings of the requests with a fixed seed. When a bin of the passport and
its comparison lie on the two sides of the reference bin and the comparison lies beyond the half-width of its 95%
interval, the variance by the load has turned the bin against the data: the quantity is fitted with one variance, and
the passport names the bins (`SPREAD_AGAINST_DATA`).

The section of a quantity names its fit method. `glmm_heterogeneous` is the
GLMM with heterogeneous random-intercept variance, the fit with the spread by the load. `glmm` means that it did
not converge or turned bins against the median request, and the section gives the fit with one variance. `logit` means that the shift did not converge and the plain
logit is given: its numbers are not comparable with other measurements. `none` means that nothing converged: the quantity
has no laws and no capacities.

**The own error of a bin.** The whiskers of a bin are its quasi-variance [Firth 2000]: numbers $q_k$
such that $q_k + q_l$ gives the variance of the difference of the bins $k$ and $l$. A reference bin has
an own error too: its zero comes from the coding alone.

## 6. The laws through the bins

The bins of every axis are summarized by a law. Intervals are half-widths of 95%, by the delta method
over the covariance of the coefficients and of $\sigma^2$. All laws below use $\ell = \log_2 s$ and
$u = \log_2(v/20)$. The formulas are those of the established forms; the data can give a block one of its
alternatives (*Which form a law takes*, below).

**`P_sync`.** A length slice of the grid is a curve over $\ell$ whose parameters are shared by all the slices, while
every slice has its own level $\mu_v$, the horizontal asymptote of the slice at $s \to 0$. The established curve is the
hyperbola on two asymptotes, the horizontal $\mu_v$ and the slanted $\lambda_S + \kappa_S\ell$, with the corner width
$\omega_S$; the slanted hyperbola and the binary exponential are its alternatives. A slanted hyperbola runs through the
levels of the slices:

$$VS(v, s) = \tfrac{1}{2}\Big[(\lambda_S + \kappa_S\ell) + \mu(u) - \sqrt{(\lambda_S + \kappa_S\ell - \mu(u))^2 + \kappa_S^2\omega_S^2}\Big],$$

$$\mu(u) = \mu_0 + \tfrac{\kappa_V}{2}\Big[(u - u_*) + \sqrt{(u - u_*)^2 + \omega_V^2}\Big].$$

The law of the density is the same slanted hyperbola over $x = \log_2(p/2^{-10})$:

$$P(p) = \mu_P + \tfrac{\kappa_P}{2}\Big[(x - x_*) + \sqrt{(x - x_*)^2 + \omega_P^2}\Big].$$

The number of rules is the line $M(m) = \lambda_M + \kappa_M \log_2 m$, as for `H` and `F`.

**`H`.** A slice of the grid is a curve with shared parameters and an own level $\mu_v$. The established curve is
the binary exponential with shared $A$ and $\gamma$; the two-asymptote hyperbola and the slanted hyperbola are its
alternatives. The levels of the slices lie on a line:

$$VS(v, s) = \mu_v - A \cdot 2^{\gamma\ell}, \qquad \mu_v = \mu_0 + \kappa_V u.$$

At $s \to 0$ a slice dies out to its level $\mu_v$. The density is a two-slope hyperbola over
$x = \log_2(p/2^{-10})$, a hyperbola on two slanted asymptotes with the slopes $\beta$ on the left and $\beta + \kappa_P$ on
the right, of any sign:

$$P(p) = \mu_P + \beta (x - x_*) + \tfrac{\kappa_P}{2}\Big[(x - x_*) + \sqrt{(x - x_*)^2 + \omega_P^2}\Big].$$

It falls from a top to both sides, as a hill, or rises at one end, and at $\beta = 0$ it is the slanted hyperbola of
`P_sync`. The rule number is a slanted hyperbola over $\log_2 r$, the number of rules is the line
$M(m) = \lambda_M + \kappa_M \log_2 m$.

**`F`.** The length and the number of rules are lines over $u$ and over $\log_2 m$, the share of the answer
is an exponent, and the density is a logarithm plus a line:

$$V(v) = V(20) + \kappa_V u, \quad Q(q) = c_q + A_q e^{-q/\tau}, \quad P(p) = a + b\log_2(p/2^{-10}) + cp, \quad M(m) = \lambda_M + \kappa_M \log_2 m.$$

**Which bins enter and with which weights.** The grids of `P_sync` and `H` go in two steps, as in section 5.3.1
of the article: the curve of the slices through all the cells at once, with the parameters of the curve shared and
the level of every slice its own, then the law of the levels through the levels of the slices. Every law weighs a
point by $1/(\sigma_i^2 + \tau^2)$. Here $\sigma_i^2$ is the own variance of the point — a cell, a level or a bin — and
$\tau^2$ the scatter of the points around their law beyond their own variances, by the estimate of DerSimonian and
Laird for a regression [DerSimonian, Laird 1986; Viechtbauer, Cheung 2010]:

$$\tau^2 = \max\Big(0,\ \frac{\chi^2 - (k - p)}{\operatorname{tr} P}\Big), \qquad P = W - WJ\,(J^\top W J)^{+} J^\top W, \qquad W = \operatorname{diag}(1/\sigma_i^2),$$

where $J$ holds the derivatives of the law by its free parameters in place of the design of a linear model, $k$ is
the number of the points, $p$ the rank of $J$ and $\chi^2$ the weighted sum of the squared residuals with the weights
$W$. $\tau^2$ is measured around the candidate form with the smallest $\chi^2$ per degree of freedom, so the curvature
of a law does not count as scatter, and every candidate of the block is then fitted with the same weights. The mutual
dependence of the points stays out of the weights, because with it the curve of a slice runs away from its cells;
the errors are still carried by the full covariance, through the sensitivity of the estimates to the coefficients.
Every bin with a coefficient enters its law; a bin leaves it only as an outlier, by the procedure of astropy
(`FittingWithOutlierRemoval` with `sigma_clip`): the residuals of the bins not yet clipped give their median and
their standard deviation, a bin farther than three standard deviations from the median is clipped, the law is
fitted again through the rest, and the procedure repeats until the number of the clipped bins stops changing.
The scale is one for all the bins, so a poorly determined bin gets no wider tolerance. A parameter that the data
push to a bound of its range stays there, and the law names it in `held`.

**Which form a law takes.** Every block has its established form, the first of the block in
`config/passport_laws.yaml`, and alternatives after it, taken from the line, the slanted hyperbola and the binary
exponential. The density of `H` has the two-slope hyperbola established and the slanted hyperbola among its
alternatives, the density of `F` the logarithm plus a line, and the share of the answer of `F` the exponential with
the line as its only alternative; the slices of the grids choose among the forms named above. The simple hyperbola
$c + A/x$ is no candidate: every law defines the level of its block at the zero of its abscissa — one rule, the first
rule of the list, 20 words, the density $2^{-10}$, the start of the answer — and that hyperbola has its pole there.
The amplitude of the exponential and of the binary exponential and the slope of the slanted hyperbola take either
sign, so a law can rise along its axis as well as fall. The established form is fitted first; every form is then
fitted through the bins it kept with the same weights, and the form with the smallest criterion
$\mathrm{QAIC} = \chi^2/\hat c + 2k$ is taken [Burnham, Anderson 2002]. Here $k$ is the number of free parameters and
$\hat c$ the overdispersion: the smallest $\chi^2$ per degree of freedom among the converged forms, not below one, so
that a flexible form that lies badly does not inflate it. The bins scatter around any law more than their own errors
allow, and $\hat c$ brings the residuals to their real scatter. The established form wins a tie. An alternative with a
parameter held on a bound of its range does not replace the established form: the data push it to the edge of its
family, and its criterion says little of its shape. The law publishes its form, the established form, $\hat c$, the
record of every candidate and the clipped bins.

## 7. The level

The shift of a block is the maximum of its law. For a law that falls along its axis the maximum stands at the
zero of the axis: for the grid of `P_sync` it is $\mu_0$, the horizontal asymptote of the plateau; for the grid of
`H` it is $\mu_0$, the level of a text of 20 words at $s \to 0$; for the density of `P_sync` it is the law
at $p \to 0$; for the rules it is the law at $m = 1$ and $r = 1$. The law of the density of `F` can rise at rare
replacements and have a hill: then its shift $\delta_P$ is the top of the hill, at $p^* = b/(\lvert c\rvert\ln 2)$;
without a hill it is the law at $p = 2^{-10}$. So after the correction every contribution of these loads is a price
and never lies above zero. The density of `H` is the exception: its two-slope hyperbola has no horizontal asymptote,
and its shift $\mu_P$ is the law at the zero of its abscissa, $p = 2^{-10}$, the edge of the measured zone. Where the
law of the density of `H` rises from rare replacements to a hill or towards full density, its contribution after the
correction lies above zero there. The level of a quantity is the sum of its shifts:

$$\alpha_{P_{\mathrm{sync}}} = \mu_0 + \mu_P + \lambda_M, \qquad \alpha_H = \mu_0 + \mu_P + \lambda_M + \lambda_R, \qquad \alpha_F = V(20) + Q(0) + \delta_P + M(1).$$

The level answers the **zero point**: a text of 20 words, the start of the answer, the density at the maximum of
its law and one rule. For a law that falls from rare replacements this density goes to zero; for the law of the
density of `F` with a hill it is the best density of the model; for the density of `H` it is $p = 2^{-10}$.
Moving the shifts into the level
is a calibration, not a change of the model: the prediction of every trial stays the same. The figures
of a passport show every axis before this correction, as the coefficients of the fit, and after it,
minus the shift of the axis.

## 8. Capacities

**The level of the capacities.** A capacity of the level $q$ is the load at which the **share of successes
over all requests** is still not below $q$. The bins describe the median request; the share over all
requests at the logit $\eta$ of the median request is

$$\bar\pi(\eta) = E_b\Big[\frac{1}{1 + e^{-(\eta + b)}}\Big], \qquad b \sim N(0, \sigma^2),$$

and the **threshold** $\eta^*(q, \sigma^2)$ is the root of $\bar\pi(\eta^*) = q$. Above one half the threshold
grows with the variance, and the variance depends on the load (section 5), so a capacity of such a level
would depend on the load at which its variance is read, while the axes a capacity does not name have no
natural value for it. At $q = 0.5$ the threshold is zero at any variance and for any symmetric
distribution of the shift, because $1/(1 + e^{-b}) + 1/(1 + e^{b}) = 1$. The passport therefore gives every
capacity at the level 0.5: a capacity is the load at which the laws bring the logit of the median request
to zero, the **reserve** of a quantity is its whole level $\alpha$, and the variance of the shift enters the
capacities only through the laws, which it keeps unbiased.

| Quantity | Capacity | Definition | Unit |
|---|---|---|---|
| `H` | instruction | the root of $-\kappa_M\log_2 m + \int_1^m [R(1) - R(r)]\,dr = \alpha$ | rules |
| `H` | fatigue | the word number at which the full law of the grid reaches zero at the shortest text holding the word, $v = s$ | words |
| `P_sync` | instruction | $-\kappa_M\log_2 m = \alpha$, as for `F`; none when the 95% interval of $\kappa_M$ reaches zero | rules |
| `P_sync` | fatigue | the word at which the full law of the grid at $p \to 0$ reaches zero, $v = s$: the root of the same curve as the other capacities of the grid, not its slanted asymptote; none when the law stays above zero up to $2^{40}$ words, as when the plateau rises with the length faster than the slices fall | words |
| `F` | instruction | $-\kappa_M\log_2 m = \alpha$; a background word has no rule number | rules |
| `F` | fatigue | the number of the last word of a text of the same length: the root of $\alpha - [Q(0) - Q(1)] + [V(s) - V(20)] = 0$ | words |
| every quantity | measure of the density | $e^{-\hat P(1)}$, where $\hat P(1)$ is the change of the law of the density from its zero point to $p = 1$: how many times the full density multiplies the odds of an error, written as the factor with its interval. A factor below one means that the full density lowers the odds of an error, as the rising law of the density of `H` can give. For `F` a conditional quantity: a text has no background words at $p = 1$, so it continues the law of the density | times |

By the law, `P_sync` has an instruction capacity only when its law of the number of rules falls beyond its errors: when the 95% interval of $\kappa_M$ reaches zero, the data do not bound the capacity from above, and the passport gives none.

**Intervals and the measured area.** The intervals of a capacity are computed by the delta method on the
scale of its logarithm, so they are not symmetric. A capacity that lies beyond the measured rules or words
is marked as a continuation of the law (the status `extrapolated`). The instruction capacities of `H`,
`F` and `P_sync` are also given by the bins: the walk along the bins that spends the reserve; when the reserve is not
spent at the edge of the measured area, the passport gives the edge as a lower bound and what is left of the
reserve. The gap between the law and the bins measures the approximation.

## 9. Verification

`scripts/verify.py` runs four checks: the SHA-256 of the files listed in `measurement.json`; the vocabulary
against its reference; the rule list and the source text of every point rebuilt from its seeds; the passport
recomputed from the points against the published one. The composition of the passport is compared exactly
(keys, list lengths, strings, integers, statuses). Floating-point numbers of the fits and the weights are
compared within $10^{-6}$ relative, all the others within $10^{-3}$ relative; every number also has
an absolute floor of $10^{-9}$. Reordering the summation of the same points moved the passport of the published
measurement by at most $7.6 \cdot 10^{-9}$ relative in the first group and $9.5 \cdot 10^{-6}$ in the second, so
the tolerances keep a margin of about 100 times and more. `verify` checks integrity and reproducibility only; it
does not prove the method right.
