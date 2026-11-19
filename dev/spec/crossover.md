# Crossover

Parameters: `crossover`, `eigenvalue_crossover`. Uses `CR̃_i` (adaptation.md).

## 1. Binomial (`BIN`)
For each `i`: draw `j_rand ~ U{1…d}`. For each coordinate `j`, draw `ρ_ij ~ U(0,1)`, and set

`u_ij = v_ij` if `ρ_ij ≤ CR̃_i` or `j = j_rand`, otherwise `x_ij`.

At least one coordinate always comes from the donor.

## 2. Exponential (`EXP`)
For each `i`:
1. Draw a start `k ~ U{1…d}` and set `L = 1`.
2. While `L < d`: draw `ρ ~ U(0,1)`; if `ρ < CR̃_i` then `L ← L + 1`, else stop.
3. `u_ij = v_ij` for the `L` cyclically consecutive coordinates `k, k+1, …, k+L−1 (mod d)`; otherwise `u_ij = x_ij`.

`L` follows a truncated geometric distribution: `P(L ≥ l) = CR̃^(l−1)`.

## 3. Eigenvector crossover (`eigenvalue_crossover`)
Before crossover:
1. Compute the covariance of the current population, `C = cov(X)` (`d × d`, normalised by `λ − 1`), and its eigendecomposition `C = B Λ Bᵀ` (`B` orthonormal, columns are eigenvectors).
2. Express target and donor in the eigenbasis: `x'_i = Bᵀ x_i`, `v'_i = Bᵀ v_i`. **KI-16 fixed:** 0.0.4 used `B x`, a rotation into a different basis.
3. Apply BIN or EXP to `x'_i, v'_i`, giving `u'_i`.
4. Rotate back: `u_i = B u'_i`.

Notes:
- With `λ ≤ d`, `C` is singular. The eigendecomposition still exists, and the zero-eigenvalue directions are an arbitrary orthonormal completion. This is acceptable, but noted for the interaction study.
- ⚠️ README open point 4: the paper may apply this only with some probability, and/or on a subset of the population (to verify).

## Random draws
- BIN: `d` uniforms and one `j_rand` per individual.
- EXP: one start index plus up to `d − 1` uniforms per individual.
- Eigen: none extra.
