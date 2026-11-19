# Population size

Parameters: `lambda0`, `population_reduction`, `lambda_min`.

- `lambda0` is the initial size. Its default (10·d) is the same for every configuration; no module changes it (D-D6, KI-31).
- `lambda_min` (default `max(4, n_parents_needed + 1)`, DERIVED) is the final size of a reduction schedule. Resolution guarantees `n_parents_needed + 1 ≤ lambda_min ≤ lambda0` (D-D5).

## NONE
`λ = lambda0` throughout.

## LINEAR (L-SHADE's LPSR)
At the end of `adapt()`, after the archive and memory updates, with progress
`t` after this generation's evaluations:

`λ_next = round(lambda0 + (lambda_min − lambda0) · t)`

If `λ_next < λ`: remove the `λ − λ_next` individuals with the **highest** `f`
(ties: the higher index is removed first), together with their stored
`F_i, CR_i`. Then trim the archive to its new capacity (selection_archive.md §3).
`λ` never increases.

Since `E` includes the initial population and any evaluations the problem had
before the run (D-B3), `t` measures progress on the problem's total budget.

## Random draws
None (archive trimming draws are specified in selection_archive.md).
