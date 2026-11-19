# Mutation

Parameters: `mutation_base`, `mutation_reference`, `mutation_n_comps`,
`pbest_sampling`, `pbest_rate`, `pbest_rate_max`, `mutation_use_weighted_F`,
`use_archive` (archive maintenance: selection_archive.md).

## 1. Donor vector

For each target `i = 1…λ`, with `n = mutation_n_comps`:

```
v_i = b_i  +  F_ref,i · (x_ref,i − x_i)          (only if mutation_reference ≠ NONE)
           +  F̃_i · (x_r1 − x̃_r2)                (first difference term; x̃ from P ∪ A if use_archive)
           +  Σ_{k=2..n} F̃_i · (x_r(2k−1) − x_r(2k))
```

- **Base vector `b_i`** (`mutation_base`):
  - `RAND`: `x_r0`;
  - `BEST`: `x_best` (lowest `f`; ties broken by lowest index);
  - `TARGET`: `x_i`.
- **Reference term** (`mutation_reference`): `x_ref,i` is
  - `RAND`: `x_r`;
  - `BEST`: `x_best`;
  - `PBEST`: a p-best vector (§3).
- **Difference terms:** `n ≥ 0` terms. With `n = 0` there are none (then `use_archive` has no effect; resolution warns). **Each term uses its own pair of parents** (KI-1 fixed).
- **Archive term** (`use_archive`, KI-27): only the *first* difference term changes. Its second vector `x̃_r2` is drawn uniformly from `P ∪ A` (§4). The number of difference terms is still `n` (as in JADE's current-to-pbest/1 with archive).

## 2. Parents (D-D8, KI-23)

All population members drawn as parents (`r0` for a RAND base, `r` for a RAND
reference, `r1`, and `r2 … r2n` when taken from `P`) are **mutually distinct
and distinct from `i`**. They are drawn uniformly without replacement from
`{1…λ} \ {i}`. Resolution guarantees `λ ≥ n_parents_needed + 1` (settings.md),
also after population reduction (`lambda_min`).

`x_best` and the p-best vector are *not* required to be distinct from `i` or the parents.

## 3. p-best (`mutation_reference = PBEST`)

1. Sort the population by `f` (ascending; ties broken by index).
2. Choose `p_i`:
   - `FIXED`: `p_i = pbest_rate` (KI-25);
   - `RANDOM`: `p_i ~ U(2/λ, max(2/λ, pbest_rate_max))`, drawn per individual (SHADE).
3. `k_i = max(1, round(p_i · λ))`. `x_ref,i` is drawn uniformly from the `k_i` best individuals.

## 4. Archive draw (`use_archive`)

Let `A` be the archive (selection_archive.md). Draw an index uniformly from
`P ∪ A`, i.e. `U{0 … λ + |A| − 1}`. If it refers to a population member that
is `i` or one of the other parents of this donor, redraw. Members of `A` are
never rejected. With an empty archive this reduces to drawing from `P`.

## 5. Scale factors

- `F̃_i` is the scale factor of this generation (adaptation.md).
- **Weighted F** (`mutation_use_weighted_F`, KI-24): **only the reference term** uses
  `F_ref,i = w(t)·F̃_i`, where `w(t) = 0.7` if `t < 0.2`, `0.8` if `t < 0.4`, else `1.2`
  (jSO; fixed constants, README open point 2). Without weighted F, `F_ref,i = F̃_i`.
  With `mutation_reference = NONE` the module has no effect (resolution warns).
- `t` is the progress at the start of the generation.

## Random draws
Per individual: the distinct parents; for `PBEST`, `p_i` (if `RANDOM`) and the
index among the top `k_i`; for the archive term, one index plus redraws.
