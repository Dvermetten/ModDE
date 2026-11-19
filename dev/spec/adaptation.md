# F / CR control

Parameters: `F0`, `CR0`, `adaptation_method_F`, `adaptation_method_CR`,
`memory_size`, `shade_scale_F`, `shade_scale_CR`, `jde_tau_F`, `jde_F_lower`,
`jde_F_width`, `jde_tau_CR`, `use_jso_caps`.

F and CR are controlled **independently**: any F method can be combined with
any CR method (e.g. SHADE F + jDE CR). Each individual stores `F_i, CR_i`
(initially `F0, CR0`). Every generation:
- **step 0** samples the values used in that generation, `F̃_i, CR̃_i`;
- **`adapt()`** (step 7) updates memories and stored values from the success record `R` (selection_archive.md).

## 1. NONE
`F̃_i = F_i = F0`, `CR̃_i = CR_i = CR0`. No update.

## 2. SHADE family (memories)
Memories `M_F[0…H−1]` and `M_CR[0…H−1]`, with `H = memory_size`, initialised to
`F0` / `CR0` (spec README open point 3). There is one write index `k` (initially 0),
**shared** by both memories.

**Step 0 (sampling):** per individual, draw one slot `r_i ~ U{0…H−1}`, used for both F and CR.
- **CR** (`SHADE` or `LSHADE`): if `M_CR[r_i] = ⊥` (terminal, LSHADE only), `CR̃_i = 0`. Otherwise `CR̃_i = clip(N(M_CR[r_i], shade_scale_CR), 0, 1)`.
- **F**:
  - `SHADE`: `F̃_i ~ Cauchy(M_F[r_i], shade_scale_F)`;
  - `SHADE_MODIFIED`: `F̃_i ~ Cauchy(mean(M_F), shade_scale_F)`.

  In both cases, if `F̃_i ≤ 0`, redraw it (SHADE: with a new `r_i` for F only; SHADE_MODIFIED: from the same distribution). Then `F̃_i ← min(F̃_i, 1)`.

**`adapt()` (memory update):** if `S ≠ ∅`, with weights `w = Δf / ΣΔf`:
- `M_F[k] ← Σ w·S_F² / Σ w·S_F` (**weighted Lehmer mean**, KI-20; for both SHADE and SHADE_MODIFIED);
- `M_CR[k]`:
  - `SHADE`: `Σ w·S_CR` (weighted arithmetic mean);
  - `LSHADE`: if `M_CR[k] = ⊥` or `max(S_CR) = 0`, then `M_CR[k] ← ⊥` (terminal, stays terminal); otherwise `Σ w·S_CR² / Σ w·S_CR` (Lehmer mean). This is L-SHADE's rule (D-D10).
- `k ← (k + 1) mod H`.

If `S = ∅`, the memories and `k` are unchanged. A memory exists only if its parameter uses a SHADE-family method, but `k` advances whenever either memory is updated.

## 3. jDE (KI-2, KI-22)
**Step 0:** independently per individual:
- F: with probability `jde_tau_F`, `F̃_i = jde_F_lower + U(0,1)·jde_F_width`; otherwise `F̃_i = F_i`.
- CR: with probability `jde_tau_CR`, `CR̃_i ~ U(0, 1)`; otherwise `CR̃_i = CR_i`.

**`adapt()` (commit):** for `i ∈ W` (trial selected), `F_i ← F̃_i` and `CR_i ← CR̃_i`. The others keep their stored values.

## 4. jSO caps (`use_jso_caps`, KI-21)
Applied **after** step-0 sampling, using the progress `t` at the start of the generation:
- `CR̃_i ← max(CR̃_i, 0.7)` if `t < 0.25`; else `CR̃_i ← max(CR̃_i, 0.6)` if `t < 0.5`;
- `F̃_i ← min(F̃_i, 0.7)` if `t < 0.6`.

The caps change only the values *used*. What enters `S_F`/`S_CR` (and jDE's
commit) is the used, capped value. The thresholds are fixed constants (README open point 2).

## 5. Interaction with population size changes
When individuals are removed (population_size.md) or reordered (oppositional
generation), their stored `F_i, CR_i` move with them. The memories are not affected.

## Random draws
SHADE: `r_i`, one normal and one or more Cauchy draws per individual. jDE: two uniforms per adapted parameter per individual.
