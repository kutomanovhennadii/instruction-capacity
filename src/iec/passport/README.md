# passport — the passport of a measurement from its published points

The package computes the passport of a published measurement - the laws and capacities of the three quantities `P_sync`, `H` and
`F` and everything they are computed from - writes its document and figures, and verifies a published measurement.

## What is here

| Module or folder | Role |
|---|---|
| `trials/` | the layout of the bins, the run cube and the weights of the points, the trials of the three quantities by the places of the alignment, the folded designs |
| `fitting/` | the fits of the bins: the plain logit, the GLMM with the shift of a request, the spread of the shift by the load and the chain that tries them in order |
| `laws/` | the laws through the fitted bins, their forms (`laws/forms/`, assigned by `config/passport_laws.yaml`) and the rows of the passport |
| `quantities/` | the sections of `P_sync`, `H` and `F`: the model of every quantity, its laws, level and capacities by the laws and by the bins |
| `document/` | `passport.md` and the figures of every quantity by the plan `config/passport_figures.yaml` |
| `passport_data` | the whole calculation: from the published records to the passport dictionary |
| `measurement_files` | the files of a measurement directory and of its passport: the points in one file or in numbered parts read in order; `passport_data.json` with its format version |
| `verify` | the four checks of a published measurement |

## Pipeline

1. **Trials.** For every `ok` record the alignment gives the places of the answer, and they give the trials of the three quantities
   by word-number bins. All answered points enter.
2. **Bins and weights.** Text length `v = D·n` from 20 words by doubling, density by the levels `⌊log₂ p⌋`, the number of rules by
   `{1}, {2}, {3-4}, {5-8}, ...`, the word number by the doubling bin of the answer line. The weight of a point is inverse to the
   filling of its run-cube cell, normalised to the mean weight 1.
3. **Fits.** A bin where every trial has one outcome is removed with its trials until a fixed point. The chain of fits gives the
   GLMM with heterogeneous random-intercept variance (`glmm_heterogeneous`), or the GLMM with one variance (`glmm`), or the plain
   logit (`logit`), or no laws (`none`).
4. **Laws, level, capacities.** Every quantity draws its laws through its bins with the forms of `config/passport_laws.yaml`, sums
   the shifts of the blocks into its level and reads the capacities at the level 0.5.
5. **Documents.** `passport_data.json`, `passport.md` and the figures in `figures/P_sync/`, `figures/H/`, `figures/F/` by the plan
   `config/passport_figures.yaml`: 24, and three more for every quantity with the spread by the load. Every file is written through
   a temporary file and a rename.

## Speed

The points are aligned independently of each other and the three sections are fitted independently of each other, so the
commands run both in a pool of processes (`calculation_pool`, `--workers`, by default one per processor). The points come
back in the order of the records and every section is the same computation in another process: the passport is the same to
the last digit with the pool and without it. On 8 processors the passport of v55 takes about 140 seconds instead of 213.
The tests in `tests/speed` guard the reference times of the alignment and of the sections (`-m speed`).

## Models

    P_sync:  η = VS(v, s) + P(p) + M(m) + b
    H:       η = VS(v, s) + P(p) + M(m) + R(r) + b
    F:       logit(1 − F) = V(v) + q(s/v) + P(p) + M(m) + b

`b ~ N(0, σ_j²)`, `ln σ_j = z_j·γ`, is the shift of a request in the GLMM with heterogeneous random-intercept variance, shared by
all its trials and integrated out of the likelihood. The bins describe the median request (`b = 0`). The quantities combine as
`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`, `π = m/D`; the formula has no law and no capacity of its own.

## Tolerance of `verify`

Reordering the summation of the same points moved the passport of the published measurement by at most `7.6e-9` relative in the
fits and the weights, and by at most `9.5e-6` relative in everything else. `verify` compares:

1. the composition exactly: keys, list lengths, strings, booleans, `null`, integers;
2. floats under the keys `fit` and `weighting` within `1e-6` relative;
3. all other floats within `1e-3` relative;
4. every float also has the absolute floor `1e-9`: a difference of zeros.

## Key signatures

    calculation_pool(workers) -> context manager of a ProcessPoolExecutor or None
    measurement_trials(description, records, progress=..., *, executor=None) -> MeasurementTrials
    build_passport_data(description, records, trials, name, assignment, progress=..., *, executor=None) -> dict[str, Any]
    write_passport_data(passport, directory) -> Path; read_passport_data(directory) -> dict[str, Any]
    verify_measurement(directory, vocab_dir, assignment, *, executor=None) -> VerificationReport
    compare_passports(published, recomputed) -> list[str]
    write_documents(passport, directory, assignment, plan) -> tuple[Path, tuple[Path, ...]]

## Errors

| Code | When |
|---|---|
| `PASSPORT_INPUT_MISSING` | the measurement has no answered points (`PassportError`) |
| `POINT_NOT_OK` | an answered point has no answer (`ContractViolationError`) |
| `MEASUREMENT_FILES_MISSING` | the measurement directory has no description or points file (`PassportError`) |
| `PASSPORT_DATA_INVALID` | `passport_data.json` is not a passport of a known format (`ContractViolationError`) |

The codes of the folders are listed in their own README files.

## Boundaries

The package does not choose points and does not call the model: it reads published records. The forms of the laws and the plan of
the figures are arguments of the calculation and of the document; the commands read them from `config/`.

Method: [The passport model](../../../docs/05-passport-model.md).
