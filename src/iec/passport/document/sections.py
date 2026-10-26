"""Sections of the passport document: totals, capacities, figures, model, diagnostics, cube.

A section reads only the passport dictionary and returns its lines; an empty list means that
the section does not apply.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from iec.passport.document.figures import (
    COLLAPSE_FIGURE_NAMES,
    FIGURE_NAMES,
    PLACE_FIGURE_NAMES,
)
from iec.passport.document.formatting import (
    APPROXIMATION_BLOCKS,
    approx_number,
    approx_signed,
    approx_value,
    format_number,
    optional_number,
    weighting_line,
)
from iec.passport.document.labels import METHODS, WORDS, format_note, quantity_label
from iec.passport.measurement_files import FIGURES_DIRNAME
from iec.passport.model_fitting import HALFWIDTH_QUANTILE
from iec.passport.passport_data import INCOMPLETE_TOKEN_COUNTERS
from iec.passport.public_quantities import Quantity

FIGURE_CAPTIONS: dict[str, str] = {
    "vs-slices-before-shift.png": "Grid cells by length slice and the slice curves, before the "
    "shift (raw coefficients)",
    "vs-slices-after-shift.png": "Grid cells by length slice and the slice curves, after the "
    "shift δ_VS",
    "plateau-before-shift.png": "Plateau μ_v by length slice and the plateau hyperbola, before "
    "the shift",
    "plateau-after-shift.png": "Plateau μ_v by length slice and the plateau hyperbola, after the "
    "shift δ_VS",
    "P-before-shift.png": "Density bins and their law, before the shift",
    "P-after-shift.png": "Density bins and their law, after the shift μ_P",
    "M-before-shift.png": "Rule-count bins and their law, before the shift",
    "M-after-shift.png": "Rule-count bins and their law, after the shift λ_M",
    "R-before-shift.png": "Rule-number bins and their law, before the shift",
    "R-after-shift.png": "Rule-number bins and their law, after the shift λ_R",
    "collapse-V-before-shift.png": "Collapse of the answer: text-length bins and their law, "
    "before the shift",
    "collapse-V-after-shift.png": "Collapse of the answer: text-length bins and their law, after "
    "the shift λ_Vψ",
    "collapse-P-before-shift.png": "Collapse of the answer: density bins and their law, before "
    "the shift",
    "collapse-P-after-shift.png": "Collapse of the answer: density bins and their law, after the "
    "shift μ_Pψ",
    "place-S-before-shift.png": "Place of the collapse: word-number bins and their law, before "
    "the shift",
    "place-S-after-shift.png": "Place of the collapse: word-number bins and their law, after the "
    "shift μ_SF",
    "place-P-before-shift.png": "Place of the collapse: density bins and their law, before the "
    "shift",
    "place-P-after-shift.png": "Place of the collapse: density bins and their law, after the "
    "shift μ_PF",
    "place-Q-before-shift.png": "Place of the collapse: bins of the share of the text and their "
    "line, before the shift",
    "place-Q-after-shift.png": "Place of the collapse: bins of the share of the text and their "
    "line, after the shift λ_QF",
}

#: The second-stage laws and the meaning of the symbols, the same for every VS-PMR model, in
#: LaTeX: GitHub renders ``$$…$$`` as display math.
_LAW_FORMULAS: tuple[str, ...] = (
    "The second stage draws a law of its axis through the bins of every block:",
    "",
    r"$$VS = \tfrac{1}{2}\Big[(\lambda_S + \kappa_S\ell) + \mu(v)"
    r" - \sqrt{(\lambda_S + \kappa_S\ell - \mu(v))^2 + \kappa_S^2\omega_S^2}\Big],"
    r" \qquad \ell = \log_2 s$$",
    "",
    r"$$\mu(v) = \mu_0 + \tfrac{\kappa_V}{2}\Big[\log_2\tfrac{v}{v_{*}}"
    r" + \sqrt{\log_2^2\tfrac{v}{v_{*}} + \omega_V^2}\Big]$$",
    "",
    r"$$P(p) = \mu_P - A_P\big(1 - e^{-p/p_c}\big), \qquad M(m) = \lambda_M + \kappa_M \log_2 m,"
    r" \qquad R(r) = \lambda_R - A_R\,(r^{\gamma} - 1)$$",
    "",
    r"Here $s$ is the word position in the answer, $v = D \cdot n$ the text length in words, "
    r"$p = m/D$ the replacement density, $m$ the number of rules and $r$ the rule number. "
    r"The surface law is fitted in two steps: every length slice has its own plateau $\mu_v$ "
    r"with shared $\kappa_S$, $\lambda_S$, $\omega_S$, and the plateau $\mu(v)$ is a hyperbola "
    r"in the text length drawn through the $\mu_v$: the horizontal asymptote $\mu_0$ for short "
    r"texts and the slope $\kappa_V$ per doubling for long texts, with the corner at the length "
    r"$v_*$ and the width $\omega_V$. The shift of a block is its law at the zero of its axis - "
    r"for the surface the horizontal asymptote $\delta_{VS} = \mu_0$, approached at the start of "
    r"the answer ($s \to 0$) of short texts ($v \to 0$); $p \to 0$, $m = 1$, $r = 1$ - and it "
    r"goes into the level $\alpha_{phys}$.",
    "",
)

#: The plateau hyperbola of the grid law.
PLATEAU_LAW_FORMULA = (
    r"$\mu(v) = \mu_0 + \tfrac{\kappa_V}{2}\big[\log_2\tfrac{v}{v_{*}}"
    r" + \sqrt{\log_2^2\tfrac{v}{v_{*}} + \omega_V^2}\big]$"
)

#: Meaning of the law parameters - the column of the law table.
LAW_MEANINGS: dict[str, str] = {
    "delta_VS": "shift of the grid: the horizontal asymptote μ_0 of the plateau (the start of "
    "the answer of short texts)",
    "kappa_S": "fatigue slope, logit per doubling of the word position",
    "lambda_S": "level of the slanted fatigue asymptote at the first word (s = 1), relative to "
    "the shift delta_VS",
    "omega_S": "transition width, doublings of the word position",
    "kappa_V": "slope of the slanted asymptote of the plateau: logit per doubling of a long text",
    "v_star": "corner of the plateau: the text length in words where its asymptotes cross",
    "omega_V": "width of the corner of the plateau, doublings of the text length",
    "mu_P": "density shift (the law at p → 0)",
    "A_P": "loss at saturation of the density",
    "p_c": "characteristic density",
    "lambda_M": "rule-count shift (the line at m = 1)",
    "kappa_M": "slope per doubling of the number of rules",
    "lambda_R": "rule-number shift (the curve at r = 1)",
    "A_R": "scale of the late-rule degradation",
    "gamma": "exponent of the late-rule degradation",
}


def model_ids(passport: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the models of the passport in its order - the keys of its sections."""
    return tuple(passport.get("passport") or {})


def model_formulas(model: str, shift_in_fit: bool) -> list[str]:
    """Return the formula of a model and the second-stage laws in LaTeX.

    With the shift of a point inside the fit the logit carries ``b`` and the level of a capacity
    is read at the threshold ``η*(q, σ²)``.
    """
    if shift_in_fit:
        logit = r"$$\eta = VS(v, s) + P(p) + M(m) + R(r) + b, \qquad b \sim N(0, \sigma^2)$$"
        shift = [
            r"$b$ is the shift of a point, shared by all its trials and integrated out of the "
            r"likelihood; the bins and laws describe the median point ($b = 0$). A capacity "
            r"keeps its meaning - the share of right replacements over all points - so the "
            r"level $q$ is read at the threshold $\eta^*(q, \sigma^2)$ at which the median "
            r"point gives the share $q$.",
            "",
        ]
    else:
        logit = r"$$\eta = VS(v, s) + P(p) + M(m) + R(r)$$"
        shift = []
    return [
        f"The passport model VS-PMR (`{model}`) gives the logit of a right replacement as",
        "",
        logit,
        "",
        *shift,
        *_LAW_FORMULAS,
    ]


def _approximation(passport: Mapping[str, Any], model: str) -> dict[str, Any] | None:
    """Return the approximation block of the section of a model, or ``None``."""
    block = ((passport.get("passport") or {}).get(model) or {}).get("approximation")
    return block if isinstance(block, dict) else None


# --- Totals ------------------------------------------------------------------------------


def header_lines(passport: Mapping[str, Any]) -> list[str]:
    """Totals of the published points and the alarms of the passport."""
    resources = passport["resources"]
    total = resources.get("total_cost_usd")
    money = (
        "- money: not computed, the measurement has no tariff"
        if total is None
        else f"- money: **${float(total):.4f}** - cost of the published points (provider "
        "tokens × the tariff of the measurement)"
    )
    lines = [
        f"- points: {resources['unique_point_keys']} published, {resources['ok_runs']} answered",
        f"- tokens: prompt {resources['prompt_tokens_billed']}, response "
        f"{resources['completion_tokens_billed']} (provider counters)",
        money,
        f"- Bernoulli trials: {resources['trials']} in {resources['ok_runs']} answered points - "
        "these are not model requests",
    ]
    alarms = passport.get("alarms") or []
    incomplete = next(
        (alarm for alarm in alarms if alarm.get("code") == INCOMPLETE_TOKEN_COUNTERS), None
    )
    if incomplete is not None:
        lines.append(
            f"- **provider counters are incomplete**: {incomplete['evidence']['records']} billed "
            "points without counters - the token sums and the money are understated"
        )
    lines += ["", f"Passport alarms ({len(alarms)}):", ""]
    if not alarms:
        return [*lines, "No alarms."]
    return lines + [
        f"- {alarm.get('severity')} `{alarm.get('code')}`: {alarm.get('message')}"
        + (" **[affects primary metrics]**" if alarm.get("affects_primary_metrics") else "")
        for alarm in alarms
    ]


# --- Capacities --------------------------------------------------------------------------


def value_cell(quantity: Quantity) -> str:
    """Write ``value ± 1.96·SE`` with the method of calculation."""
    if quantity.value is None:
        text = WORDS["undefined"]
    elif quantity.se is None:
        text = format_number(float(quantity.value))
    else:
        text = (
            f"{format_number(float(quantity.value))} ± "
            f"{format_number(HALFWIDTH_QUANTILE * float(quantity.se))}"
        )
    if quantity.method is None:
        return text
    return f"{text} ({METHODS[quantity.method]})"


def public_table(quantities: Sequence[Quantity]) -> list[str]:
    """Table of the public quantities: a row per key and level, two calculations in one row."""
    groups: dict[tuple[str, float | None], list[Quantity]] = {}
    for quantity in quantities:
        groups.setdefault((quantity.key, quantity.level), []).append(quantity)
    lines = [
        f"| {WORDS['quantity']} | {WORDS['value']} | {WORDS['unit']} | {WORDS['status']} | "
        f"{WORDS['note']} |",
        "|---|---|---|---|---|",
    ]
    for (key, level), group in groups.items():
        label = quantity_label(key)
        title = label.title if level is None else f"{label.title}, q = {level:g}"
        args: dict[str, Any] = {}
        for item in group:
            args.update(item.args)
        values = " · ".join(value_cell(item) for item in group)
        statuses = " · ".join(f"`{item.status}`" for item in group)
        lines.append(
            f"| {title} | {values} | {label.unit} | {statuses} | "
            f"{format_note(label.note, level, args)} |"
        )
    return lines


def capacities_lines(passport: Mapping[str, Any], model: str) -> list[str]:
    """The public quantities of a model, or the reason why they are not computed."""
    block = _approximation(passport, model)
    rows = (block or {}).get("public")
    if isinstance(rows, list) and rows:
        return public_table([Quantity.from_dict(row) for row in rows])
    return [WORDS["no_public"].format(status=(block or {}).get("status"))]


# --- Figures -----------------------------------------------------------------------------


def figures_lines(figures: Mapping[str, Path], model: str) -> list[str]:
    """Links to the written figures of a model in the order of the passport."""
    lines: list[str] = []
    for name in FIGURE_NAMES:
        key = f"{model}/{name}"
        if key in figures:
            lines += ([""] if lines else []) + [
                f"![{FIGURE_CAPTIONS[name]}]({FIGURES_DIRNAME}/{key})"
            ]
    return lines


# --- Model parameters --------------------------------------------------------------------


def model_params_lines(passport: Mapping[str, Any], model: str) -> list[str]:
    """The formulas, the level ``α_phys``, the block shifts and the law parameters with χ²."""
    block = _approximation(passport, model)
    if block is None:
        return []
    fit = (passport.get("fit") or {}).get(model) or {}
    lines = [
        *model_formulas(model, bool(fit.get("shift_in_fit"))),
        f"Status: `{block.get('status')}`. Zero point: {block.get('zero_point', '—')}.",
    ]
    if block.get("reason"):
        return [*lines, "", f"Reason: {block['reason']}."]
    lines += [
        "",
        "A law of its axis with a free shift is drawn through the bins of every block "
        "(generalised least squares by the covariance submatrix of the bins); the shift - the "
        "law at the zero of the axis - is subtracted from the bins and moved into the level. "
        "The prediction of the trials does not change. Values are ± 1.96·SE.",
        "",
    ]
    alpha = block.get("alpha_phys")
    shifts = block.get("shifts")
    if isinstance(alpha, dict) and isinstance(shifts, dict):
        lines += ["| Quantity | Value |", "|---|---|"]
        lines.append(f"| `α_phys` | {approx_value(alpha.get('value'), alpha.get('se'))} |")
        for key, _, title in APPROXIMATION_BLOCKS:
            shift = shifts.get(key) or {}
            lines.append(
                f"| shift: {title} | {approx_value(shift.get('value'), shift.get('se'))} |"
            )
        lines.append("")
    else:
        lines += ["`α_phys` is not published: the laws are not measured for every block.", ""]
    laws = block.get("laws") or {}
    lines += ["| Block | Parameter | Value | Meaning |", "|---|---|---|---|"]
    for key, _, title in APPROXIMATION_BLOCKS:
        law = laws.get(key) or {}
        if law.get("status") != "measured":
            lines.append(
                f"| {title} | — | law not measured: `{law.get('status')}`, points "
                f"{law.get('n_points', '—')} | — |"
            )
            continue
        se = law.get("se") or {}
        held = set(law.get("held") or [])
        # The shift of the grid goes first: it is moved into the level.
        for name in sorted(se, key=lambda item: item != "delta_VS"):
            meaning = LAW_MEANINGS.get(name, "—")
            if name in held:
                meaning += "; held on the bound of its range - not measured"
            lines.append(
                f"| {title} | `{name}` | {approx_value(law.get(name), se.get(name))} | "
                f"{meaning} |"
            )
        lines.append(
            f"| {title} | χ² / points | {approx_number(law.get('chi2'))} / "
            f"{law.get('n_points')} | — |"
        )
    return lines


# --- Diagnostics -------------------------------------------------------------------------


def fit_lines(passport: Mapping[str, Any], model: str) -> list[str]:
    """Variance of the point shift with its origin, within-cluster correlation, weighting."""
    fit = (passport.get("fit") or {}).get(model) or {}
    method = fit.get("sigma2_method")
    origin = "" if method is None else f" ({method})"
    return [
        f"- between-point variance σ̂²{origin}: {optional_number(fit.get('sigma2'))} ± "
        f"{optional_number(fit.get('sigma2_se'))}",
        f"- within-cluster correlation ρ̂: {optional_number(fit.get('rho'))}",
        weighting_line(fit.get("cell_weighting")),
    ]


def plateau_law_text(law: Mapping[str, Any]) -> str:
    """Write the plateau hyperbola ``μ(v)`` drawn through the plateaus ``μ_v`` of the slices."""
    se = law.get("se") or {}
    held = law.get("held") or []
    kept = f"; held on the bound: {', '.join(f'`{name}`' for name in held)}" if held else ""
    return (
        "Plateau hyperbola through the `mu_v` points: "
        f"`kappa_V` = {approx_value(law.get('kappa_V'), se.get('kappa_V'))}, "
        f"`v_star` = {approx_value(law.get('v_star'), se.get('v_star'))} words, "
        f"`omega_V` = {approx_value(law.get('omega_V'), se.get('omega_V'))}{kept}; "
        f"χ² = {approx_number(law.get('plateau_chi2'))} on "
        f"{law.get('plateau_n_points')} points. "
    )


def plateau_lines(law: Any) -> list[str]:
    """The grid law by length slice: the plateau ``μ_v`` of every slice and its hyperbola."""
    if not isinstance(law, dict):
        return []
    lines = ["#### Plateau of the surface by length slice", ""]
    if law.get("status") != "measured":
        return [
            *lines,
            f"The fit is not measured: `{law.get('status')}`, points "
            f"{law.get('n_points', '—')}.",
        ]
    se = law.get("se") or {}
    lines += [
        "The grid law is fitted in two steps. Every length slice has its own plateau level "
        "`mu_v`, computed from its cells; the slope `kappa_S`, the level `lambda_S` of the "
        "slanted asymptote and the width `omega_S` are shared. The fit weights are the own "
        "variances of the cells, without their mutual dependence: the cells of a slice are "
        "strongly correlated, and the full covariance drove a plateau above all cells of its "
        "slice; the errors are carried by the full covariance. A slice enters the fit only if "
        "its plateau is computable from its points: at least two measured cells and at least "
        "one on the plateau (the plateau share is the largest share of the plateau in the "
        "derivative of a cell of the slice).",
        "",
        "The plateau hyperbola "
        + PLATEAU_LAW_FORMULA
        + " is drawn through the plateaus of all determined slices (at least five), weighted "
        r"by their own variances, and its horizontal asymptote $\mu_0$ - the plateau of short "
        f"texts - is the shift `delta_VS` = "
        f"{approx_value(law.get('delta_VS'), se.get('delta_VS'))}; `lambda_S`, `mu_v` and the "
        "hyperbola are published minus `delta_VS`.",
        "",
        plateau_law_text(law)
        + f"Shared: `kappa_S` = {approx_value(law.get('kappa_S'), se.get('kappa_S'))}, "
        f"`lambda_S` = {approx_value(law.get('lambda_S'), se.get('lambda_S'))}, "
        f"`omega_S` = {approx_value(law.get('omega_S'), se.get('omega_S'))}; "
        f"χ² by the own variances = {approx_number(law.get('chi2'))} on "
        f"{law.get('n_points')} cells with {law.get('n_parameters')} parameters.",
        "",
        "| Slice, words | ū | Cells | Plateau share | `mu_v` | Plateau of the grid law `μ(ū)` |",
        "|---|---|---|---|---|---|",
    ]
    for row in law.get("slices") or []:
        words = row.get("words") or ["?", "?"]
        if not row.get("determined"):
            lines.append(
                f"| {words[0]}–{words[1]} | {approx_number(row.get('u'))} | "
                f"{row.get('cells')} | — | not computable: {row.get('reason', '—')} | — |"
            )
            continue
        lines.append(
            f"| {words[0]}–{words[1]} | {approx_number(row.get('u'))} | {row.get('cells')} | "
            f"{approx_number(row.get('plateau_weight'))} | "
            f"{approx_value(row.get('mu_v'), row.get('se'))} | "
            f"{approx_number(row.get('law'))} |"
        )
    return lines


def _restored_label(key: str, row: dict[str, Any]) -> str:
    """Label a restored bin: a cell by words and positions, a bin by its bounds."""
    if key == "vs_grid":
        words = row.get("words") or ["?", "?"]
        positions = row.get("positions") or ["?", "?"]
        return (
            f"V{row.get('v_bin')}·S{row.get('s_bin')} (words {words[0]}–{words[1]}, "
            f"positions {positions[0]}–{positions[1]})"
        )
    if key == "density_profile":
        p = row.get("p") or ["?", "?"]
        return f"{row.get('bin')} (p {approx_number(p[0])}–{approx_number(p[1])})"
    bounds = row.get("bounds") or ["?", "?"]
    return f"{row.get('bin')} ({bounds[0]}–{bounds[1]})"


def _restored_abscissa(key: str, row: dict[str, Any]) -> str:
    """Write the abscissa of a bin - the centre of mass of its trials."""
    if key == "vs_grid":
        return f"s = {approx_number(row.get('x'))}, u = {approx_number(row.get('u'))}"
    return approx_number(row.get("x"))


def restored_lines(block: Mapping[str, Any]) -> list[str]:
    """The restored bins of every block with the law at their abscissa and the deviation."""
    lines = ["#### Restored bins", ""]
    restored = block.get("restored") or {}
    for _, key, title in APPROXIMATION_BLOCKS:
        rows = restored.get(key) or []
        lines += [
            f"#### {title}",
            "",
            "| Bin | Abscissa | Value | Law | Deviation, SE of the bin | Status |",
            "|---|---|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                f"| {_restored_label(key, row)} | {_restored_abscissa(key, row)} | "
                f"{approx_value(row.get('value'), row.get('se'))} | "
                f"{approx_number(row.get('law'))} | {approx_signed(row.get('z'))} | "
                f"{row.get('status')} |"
            )
        lines.append("")
    return lines


def correlation_lines(block: Mapping[str, Any]) -> list[str]:
    """Correlation of the components ``V, S, P, M, R`` by the increments of their bins."""
    correlations = block.get("component_correlations")
    if not isinstance(correlations, dict):
        return []
    components = list(correlations.get("components") or [])
    lines = [
        "#### Correlation of the components V, S, P, M, R",
        "",
        "Canonical correlation of the increments of neighbouring bins of two components by the "
        "covariance of the fit: the largest correlation between a linear combination of the "
        "increments of one component and a combination of the increments of the other. One "
        "means that the fit cannot tell which of the two components explains an observation; "
        "a dash - the component has no increments.",
        "",
    ]
    matrix = correlations.get("canonical") or []
    lines += [
        "#### Canonical correlation",
        "",
        "| | " + " | ".join(components) + " |",
        "|---|" + "---|" * len(components),
    ]
    for name, row in zip(components, matrix, strict=False):
        cells = ["—" if value is None else f"{float(value):.2f}" for value in row]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _joined(blocks: list[list[str]]) -> list[str]:
    """Join subsections with one blank line, trimming the blank edges of each."""
    lines: list[str] = []
    for block in blocks:
        start, end = 0, len(block)
        while start < end and not block[start]:
            start += 1
        while end > start and not block[end - 1]:
            end -= 1
        if start < end:
            lines += ([""] if lines else []) + block[start:end]
    return lines


def diagnostics_lines(passport: Mapping[str, Any], model: str) -> list[str]:
    """Engineering analytics of the fit of a model."""
    blocks = [fit_lines(passport, model)]
    block = _approximation(passport, model)
    if block is not None:
        if not block.get("reason"):
            laws = block.get("laws") or {}
            blocks.append(plateau_lines(laws.get("vs_grid")))
            blocks.append(restored_lines(block))
        blocks.append(correlation_lines(block))
    return _joined(blocks)


# --- Collapse of the answer --------------------------------------------------------------

#: The binned logit of the collapse, the laws of its axes and the rate law, in LaTeX.
_COLLAPSE_FORMULAS: tuple[str, ...] = (
    r"$$\ln\frac{\psi}{1 - \psi} = \alpha_\psi + V_\psi(\text{bin of } v)"
    r" + P_\psi(\text{bin of } p)$$",
    "",
    r"$$V_\psi(v) = \lambda_{V\psi} + \kappa_\psi \log_2\frac{v}{20}, \qquad"
    r" P_\psi(p) = \mu_{P\psi} + A_\psi\, p^{\gamma_\psi}$$",
    "",
    r"$$\ln\frac{\psi}{1 - \psi} = \lambda_\psi + \kappa_\psi \log_2\frac{v}{20}"
    r" + A_\psi\, p^{\gamma_\psi}, \qquad \lambda_\psi = \alpha_\psi + \lambda_{V\psi}"
    r" + \mu_{P\psi}$$",
)
#: The binned logit of the place of the collapse, the laws of its axes and the place law.
_PLACE_FORMULAS: tuple[str, ...] = (
    r"$$\ln\frac{F}{1 - F} = \alpha_F + S_F(\text{bin of } s) + P_F(\text{bin of } p)"
    r" + Q_F\big(\text{bin of } \tfrac{s - 1}{v}\big)$$",
    "",
    r"$$S_F(s) = \mu_{SF} + \tfrac{\kappa_{SF}}{2}\Big[\log_2\tfrac{s}{s_*}"
    r" + \sqrt{\log_2^2\tfrac{s}{s_*} + \omega_{SF}^2}\Big], \qquad"
    r" P_F(p) = \mu_{PF} + A_F\, p^{\gamma_F}, \qquad"
    r" Q_F = \lambda_{QF} + \kappa_{QF}\,\tfrac{s - 1}{v}$$",
    "",
    r"$$\ln\frac{F}{1 - F} = \lambda_F + \tfrac{\kappa_{SF}}{2}\Big[\log_2\tfrac{s}{s_*}"
    r" + \sqrt{\log_2^2\tfrac{s}{s_*} + \omega_{SF}^2}\Big] + A_F\, p^{\gamma_F}"
    r" + \kappa_{QF}\,\tfrac{s - 1}{v}, \qquad \lambda_F = \alpha_F + \mu_{SF} + \mu_{PF}"
    r" + \lambda_{QF}$$",
)
#: The rows of the parameter tables: block, key, letter, meaning.
_COLLAPSE_PARAMETERS: tuple[tuple[str, str, str, str], ...] = (
    ("rate_law", "lambda_psi", "λ_ψ", "logit of the collapse at v = 20 words and p → 0"),
    ("rate_law", "kappa_psi", "κ_ψ", "slope, logit per doubling of the text length"),
    ("rate_law", "A_psi", "A_ψ", "density term at the full density p = 1"),
    ("rate_law", "gamma_psi", "γ_ψ", "exponent of the density term"),
    ("collapse_length", "lambda_Vpsi", "λ_Vψ", "length shift: the length law at v = 20"),
    ("collapse_density", "mu_Ppsi", "μ_Pψ", "density shift: the density law at p → 0"),
)
_PLACE_PARAMETERS: tuple[tuple[str, str, str, str], ...] = (
    ("place_law", "lambda_F", "λ_F", "logit of the place on the plateau, at p → 0, at the start"),
    ("place_law", "kappa_SF", "κ_SF", "rise past the corner, logit per doubling of the word"),
    ("place_law", "s_star_F", "s_*", "corner: the word number where the asymptotes cross"),
    ("place_law", "omega_SF", "ω_SF", "width of the corner, doublings of the word number"),
    ("place_law", "A_F", "A_F", "density term at the full density p = 1"),
    ("place_law", "gamma_F", "γ_F", "exponent of the density term"),
    ("place_law", "kappa_QF", "κ_QF", "rise over the whole text, logit"),
    ("place_word", "mu_SF", "μ_SF", "word shift: the plateau of the word law, s → 0"),
    ("place_density", "mu_PF", "μ_PF", "density shift: the density law at p → 0"),
    ("place_share", "lambda_QF", "λ_QF", "share shift: the share line at the start of the text"),
)
#: The axes in the passport document: prefix, restored rows, title, density-like range.
_COLLAPSE_DOCUMENT_AXES: tuple[tuple[str, str, str, bool], ...] = (
    ("v_bin_", "length_profile", "text length v, words", False),
    ("p_bin_", "density_profile", "replacement density p", True),
)
_PLACE_DOCUMENT_AXES: tuple[tuple[str, str, str, bool], ...] = (
    ("s_bin_", "word_profile", "word number s", False),
    ("p_bin_", "density_profile", "replacement density p", True),
    ("q_bin_", "share_profile", "share of the text (s − 1)/v", False),
)


def _collapse_range(row: Mapping[str, Any], density: bool) -> str:
    """The range of a bin of the collapse block: words, rules, shares, or the density."""
    if density:
        low, high = (
            "1" if float(row[key]) >= 1.0 else f"1/{round(1.0 / float(row[key]))}"
            for key in ("lo", "hi")
        )
    else:
        low, high = (
            str(row[key]) if isinstance(row[key], int) else f"{float(row[key]):g}"
            for key in ("lo", "hi")
        )
    return low if low == high else f"{low}–{high}"


def _parameter_table(
    block: Mapping[str, Any], rows: Sequence[tuple[str, str, str, str]]
) -> list[str]:
    """A table of the final law and the shifts: value ± 1.96·SE, held on a bound, meaning."""
    lines = [f"| {WORDS['parameter']} | {WORDS['value']} | {WORDS['note']} |", "|---|---|---|"]
    for source, key, letter, meaning in rows:
        part = block[source] if source in ("rate_law", "place_law") else block["laws"][source]
        value, se = part["theta"][key] if "theta" in part else part[key], part["se"].get(key)
        held = key in (part.get("held") or [])
        text = f"{approx_number(value)} (held on its bound)" if held else approx_value(value, se)
        lines.append(f"| {letter} | {text} | {meaning} |")
    for source, law in block["laws"].items():
        lines.append(
            f"| χ² / points, `{source}` | {approx_number(law.get('chi2'))} / "
            f"{law.get('n_points')} | the law against its bins |"
        )
    return lines


def _bins_table(block: Mapping[str, Any], axes: Sequence[tuple[str, str, str, bool]]) -> list[str]:
    """The raw coefficients of the bins with their ranges, before the shift."""
    fit = block["fit"]
    dropped = "; ".join(
        f"`{item['bin']}` ({item['trials']} trials, all {item['outcome']})"
        for item in fit["dropped"]
    )
    lines = [
        f"The bins are fitted on {fit['rows']} trials ({fit['lost']} with the text lost); a bin "
        f"whose trials all have one outcome has no estimate and goes out with its trials: "
        f"{dropped or 'none'}. The coefficients are raw logits relative to the reference bin of "
        "the axis (`0`), before the shift.",
        "",
        f"| Axis | Bin | {WORDS['value']} |",
        "|---|---|---|",
    ]
    restored = block.get("restored") or {}
    for prefix, rows_key, title, density in axes:
        for row in restored.get(rows_key) or []:
            name = f"{prefix}{row['bin']}"
            if row["value"] is None:
                continue
            value = fit["theta"].get(name, 0.0)
            lines.append(
                f"| {title} | {_collapse_range(row, density)} | "
                f"{approx_value(value, fit['se'].get(name))} |"
            )
    return lines


def _check_table(check: Mapping[str, Any], title: str, what: str) -> list[str]:
    """The bins of a checked axis, its Wald statistic and the largest change of the others."""
    wald = check["wald"]
    change = check.get("largest_change_se")
    lines = [
        f"Check, {what}: the bins of the {title} fitted together with the bins above; Wald "
        f"statistic «all of them are zero» {approx_number(wald['statistic'])} with "
        f"{wald['df']} degrees of freedom, p = {approx_number(wald['p'])}; the largest change of "
        f"a bin above {approx_number(change)} of its SE.",
        "",
        f"| {title.capitalize()} | {WORDS['value']} |",
        "|---|---|",
    ]
    for row in check["bins"]:
        if row["value"] is not None:
            lines.append(
                f"| {_collapse_range(row, False)} | {approx_value(row['value'], row['se'])} |"
            )
    return lines


def collapse_lines(passport: Mapping[str, Any], figures: Mapping[str, Path]) -> list[str]:
    """The collapse of the answer: its rate and its place - bins, laws, shifts, checks, figures."""
    block = passport.get("collapse")
    if not isinstance(block, dict):
        return []
    lines = [
        f"{block['lost']} of {block['points']} answered points lost the text - their answer "
        "ended in a failure of the third kind: lines dropped, merged or added in the middle, or "
        "an answer cut before the end of the text. Both characteristics of the collapse are "
        "described by the method of the passport model: the logit is split into the bins of its "
        "axes (the weighted logit, the cluster sandwich by points), a law is drawn through the "
        "bins of every axis, and the shift of an axis - its law at the zero of the axis - goes "
        "into the level.",
        "",
        "### Rate",
        "",
        "The probability ψ that a point loses the text. Every answered point is one trial, the "
        "boundary cells of the area of analysis too, weighted by the run cube; the length law "
        "takes the inverse own variances of its bins as weights, the density law their full "
        "covariance:",
        "",
        *_COLLAPSE_FORMULAS,
        "",
        weighting_line(block.get("weighting")),
        "",
    ]
    if isinstance(block.get("rate_law"), dict):
        lines += _parameter_table(block, _COLLAPSE_PARAMETERS)
    else:
        lines.append("The laws of the rate are not measured: no shift, no rate law.")
    lines += ["", *_bins_table(block, _COLLAPSE_DOCUMENT_AXES), ""]
    lines += _check_table(block["rules_check"], "number of rules", "the number of rules")
    for name in COLLAPSE_FIGURE_NAMES:
        if name in figures:
            lines += ["", f"![{FIGURE_CAPTIONS[name]}]({FIGURES_DIRNAME}/{name})"]
    place = block.get("place")
    if isinstance(place, dict):
        lines += ["", *place_lines(place, figures)]
    return lines


def place_lines(place: Mapping[str, Any], figures: Mapping[str, Path]) -> list[str]:
    """The place of the collapse: bins, laws, shifts, the place law, checks, figures."""
    lines = [
        "### Place",
        "",
        f"The probability F that a point which lost the text has lost it by the word s. Every "
        f"replacement word of the {place['points']} points that lost the text is one trial - "
        f"{place['trials']} trials, {place['lost_trials']} of them on or after the line where "
        "the answer lost the text; the layout and the weights are those of these points. All "
        "laws are drawn over the full covariance of their bins:",
        "",
        *_PLACE_FORMULAS,
        "",
        weighting_line(place.get("weighting")),
        "",
    ]
    if isinstance(place.get("place_law"), dict):
        lines += _parameter_table(place, _PLACE_PARAMETERS)
    else:
        lines.append("The laws of the place are not measured: no shift, no place law.")
    lines += ["", *_bins_table(place, _PLACE_DOCUMENT_AXES), ""]
    lines += _check_table(place["checks"]["length"], "text length", "the text length")
    lines += [""]
    lines += _check_table(place["checks"]["rules"], "number of rules", "the number of rules")
    for name in PLACE_FIGURE_NAMES:
        if name in figures:
            lines += ["", f"![{FIGURE_CAPTIONS[name]}]({FIGURES_DIRNAME}/{name})"]
    return lines


# --- Bin cube ----------------------------------------------------------------------------


def bin_cube_lines(passport: Mapping[str, Any]) -> list[str]:
    """The filling of the purchase cube as text."""
    text = passport.get("bin_cube")
    if not text:
        return ["The cube is not built."]
    return ["```text", str(text), "```"]
