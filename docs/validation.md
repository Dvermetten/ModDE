# Validation

How do we know that modde does what this documentation says? Four layers of
checks, from single operators to whole algorithms against published results.

## 1. Tests against the specification

Every module was first written down precisely (formulas, every random draw,
edge cases) and then implemented in C++. About 1 400 tests check operators
against that specification (for example: the parents of a donor are distinct,
p-best draws from the right top fraction, the archive never exceeds its
capacity, the budget is never exceeded), run invariants over randomly sampled
configurations, and compare every preset and module option with a regression
snapshot. They run on Windows, Linux and macOS for every change.

## 2. Logging

For every evaluation (initial population, trials, oppositional jumps), the
logged `curr_F`, `curr_CR`, `corrected`, `CS` and `ED` belong to the individual
being evaluated, both with per-point and with vectorised evaluation. A full ioh
log has exactly one row per evaluation.

## 3. Module interactions

Every option was run alone and combined with every option of every other module
(541 pairs, 46 160 runs on BBOB). All combinations run without errors; the
results are on the [Module interactions](interactions.md) page and on each
module's page.

## 4. Algorithms against independent implementations

The presets `DE`, `jDE`, `SHADE` and `L-SHADE` were compared with separate NumPy
implementations written directly from the papers (sharing no code with modde),
on BBOB f1–f24 in $d = 5$ and $d = 20$ (25 runs per function, budget $2000d$).
Final precision and anytime performance were compared per function with a
Mann-Whitney test.

| Variant | d | modde better / worse / no difference (final precision) | Mean AUC modde / reference |
|---|---|---|---|
| DE | 5 | 0 / 1 / 23 | 0.139 / 0.138 |
| DE | 20 | 0 / 0 / 24 | 0.031 / 0.031 |
| jDE | 5 | 1 / 0 / 23 | 0.102 / 0.100 |
| jDE | 20 | 2 / 0 / 22 | 0.047 / 0.048 |
| SHADE | 5 | 0 / 1 / 23 | 0.103 / 0.102 |
| SHADE | 20 | 1 / 2 / 21 | 0.064 / 0.064 |
| L-SHADE | 5 | 0 / 0 / 24 | 0.126 / 0.126 |
| L-SHADE | 20 | 2 / 0 / 22 | 0.049 / 0.049 |

At $\alpha = 0.05$ without correction, about one false positive per row is
expected. After a Holm correction, two differences remain; the only one where
modde is worse (SHADE, $d = 20$, f24) did not replicate on new seeds.

## 5. Against published competition results

`L-SHADE` was run under the CEC 2014 competition protocol (the organisers'
official C code for the functions, 51 runs, $10\,000\,d$ evaluations,
$d = 10, 30, 50$) and compared with the published per-run results of L-SHADE.

The final results agree on most functions, but in the middle of a run modde's
`L-SHADE` converged later than the published runs. The cause is a difference
between the L-SHADE paper and its authors' code: the code archives the
successful trial vectors instead of the replaced parents. With the paper's
archive, the independent reference implementation behaves exactly like modde;
with the code's archive, it reproduces the published runs. modde therefore
follows the paper by default and offers the code's behaviour as
[`archive_content="TRIALS"`](modules/selection_archive.md#what-enters-the-archive-archive_content).
With that option, modde's L-SHADE reproduces the published runs:

Checkpoints (14 per function, 30 functions) at which modde differs significantly from the published runs:

| d | `archive_content="PARENTS"` (paper, default) | `archive_content="TRIALS"` (authors' code) |
|---|---|---|
| 10 | 47 of 420 | 4 of 420 |
| 30 | 81 of 420 | 16 of 420 |
| 50 | 78 of 420 | 17 of 420 |

(Mann-Whitney test per function and checkpoint, Holm-corrected; 51 runs each.)

The `jSO-like` preset is an approximation of jSO and does worse than the
published jSO results on CEC 2017; its [notes](presets.md#jso-like) list the
missing mechanisms.
