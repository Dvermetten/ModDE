# Selection, success record and archive

Parameters: `use_archive`, `archive_rate`.

## 1. One-to-one selection
For each `i` (minimisation):

`x_i ← u'_i` and `f_i ← f(u'_i)` if `f(u'_i) ≤ f_i` (ties go to the trial), otherwise unchanged.

Unevaluated trials have `f = +∞` (settings.md §2), so they never win.

## 2. Success record `R`
Fixed at selection time, so later reordering cannot misalign it (KI-7):

| Entry | Contents | Used by |
|---|---|---|
| `S` | `{i : f(u'_i) < f_i}` (strict improvement) | below |
| `S_F, S_CR` | `F̃_i, CR̃_i` for `i ∈ S` | SHADE memories |
| `Δf` | `f_i − f(u'_i) > 0` for `i ∈ S` (old minus new) | SHADE weights |
| replaced parents | old `x_i` for `i ∈ S` | archive |
| `W` | `{i : f(u'_i) ≤ f_i}` (trial selected, ties included) | jDE commit |

## 3. Archive (`use_archive`)
- Starts empty (KI-26).
- In `adapt()`: insert the replaced parents of `S` (`|S|` vectors).
- **Capacity** `c = round(archive_rate · λ)`, with `λ` the current population size (D-D4). Whenever `|A| > c` (after insertion, or after `λ` shrinks under population reduction), remove members chosen uniformly at random until `|A| = c`.
- Only vectors are stored (no fitness is needed).
- `archive_rate` replaces 0.0.4's `archive_size`, which was always overwritten with `2·λ` (KI-30).

## Random draws
Selection: none. Archive: the members removed when over capacity.
