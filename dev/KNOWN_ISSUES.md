# Known Issues in modde 0.0.4

An audit of the current Python implementation, written as input for the
module spec (Phase 1) and for [03_module_validation.md](03_module_validation.md).
Each issue must be **triaged** before porting: *fix in port*, *keep as a named
option*, or *document only*.

Legend: **Confirmed (run)** = reproduced by running the code;
**Confirmed (code)** = unambiguous from reading the code;
**To verify** = check against the original paper or the author's intent.

Line numbers refer to commit `eddc83b`.

## A. Correctness bugs

> **Triage (2026-11-19): all items in this section are fixed in the port.**
> Each fix gets a regression test (see [03_module_validation.md §5.1](03_module_validation.md#51-operator-level-tests-deterministic)).

| ID | Issue | Where | Status | Impact |
|----|-------|-------|--------|--------|
| KI-1 | `mutation_n_comps=2` adds the **same** difference vector twice. `curr_parent_idx += 2` sits outside the `for` loop, so both iterations use the same parent pair. "rand/2" is really "rand/1 with 2F". | [modularde.py:179-187](../modde/modularde.py#L179-L187) | Confirmed (code) | High: every `n_comps=2` result so far |
| KI-2 | jDE adaptation is inverted. New F/CR values are drawn when `rand > tau` (90% of the time) instead of `rand < tau` (10%). | [parameters.py:378-390](../modde/parameters.py#L378-L390) | Confirmed (code) | High: jDE behaves close to random F/CR |
| KI-3 | Problem bounds are ignored. `ModularDE` never passes `fitness_func.bounds` to `Parameters`, which defaults to `[-5, 5]`. | [modularde.py:43-47](../modde/modularde.py#L43-L47), [parameters.py:248-249](../modde/parameters.py#L248-L249) | Confirmed (run): problem on [-1,1] was searched on [-5,5] | High for any non-BBOB problem |
| KI-4 | `oversampling_factor` is a no-op. `n*(1+o)` points are sampled, but only the first `lambda_` are evaluated and kept. Combined with oppositional initialization, the mirrored half is dropped. | [modularde.py:52-71](../modde/modularde.py#L52-L71) | Confirmed (run): 50 evals with factor 1.0 | Medium |
| KI-5 | `used_budget` does not count the initial population (it calls `_fitness_func` directly). Every budget-based schedule (LPSR, jSO caps, weighted F) runs `lambda_` evaluations late. The ask-tell interface *does* count them, so the two interfaces behave differently. | [modularde.py:67-68](../modde/modularde.py#L67-L68) | Confirmed (run): 50 evals, `used_budget=0` | Medium |
| KI-6 | `oppositional_generation` takes min/max over the wrong axis. `x` is `(d, n)`, but `axis=0` gives per-individual extremes, not per-dimension. | [modularde.py:287-289](../modde/modularde.py#L287-L289) | Confirmed (code) | High for this module |
| KI-7 | `oppositional_generation` re-sorts the population but not `F`, `CR`, `old_population` or `improved_individuals_idx`. The SHADE memory update and the archive update in `adapt()` then use misaligned indices. | [modularde.py:286-298](../modde/modularde.py#L286-L298) | Confirmed (code) | High in combination with SHADE/archive |
| KI-8 | The budget can be exceeded. Termination is only checked between generations, and oppositional generation evaluates up to `lambda_` extra points without checking. | [modularde.py:275-284](../modde/modularde.py#L275-L284) | Confirmed (code) | Low–medium (affects fixed-budget benchmarking) |
| KI-9 | The Gaussian sampler draws `N(0.5, 1/6)` in the unit cube without clipping. About 0.27% of coordinates fall out of bounds (≈8% of individuals at d=30), and they are evaluated uncorrected. | [sampling.py:17](../modde/sampling.py#L17) | Confirmed (code) | Low |
| KI-10 | `AskTellDE` oppositional init uses `np.hstack` where `ModularDE` uses `np.vstack`, which gives the wrong shape. | [asktellde.py:113](../modde/asktellde.py#L113) | Confirmed (code) | Ask-tell + opposition broken |
| KI-11 | `AskTellDE` with oversampling: the population `f` has `n*(1+o)` slots, but only `lambda_` points are registered. Initialization never completes cleanly. | [asktellde.py:104-121](../modde/asktellde.py#L104-L121) | Confirmed (code) | Ask-tell + oversampling broken |
| KI-12 | `UserError` is undefined, so the guard raises `NameError` instead. | [modularde.py:96](../modde/modularde.py#L96) | Confirmed (code) | Low |
| KI-13 | `__all__` lists names that don't exist (`gaussian_sampling`, `Halton`, …), so `from modde import *` raises. | [__init__.py](../modde/__init__.py) | Confirmed (run) | Low |
| KI-14 | Per-evaluation stats (`init_stats`) assume evaluations arrive in population order. Oppositional generation breaks this. The `CS` statistic can be `nan` when the corrected point equals the target (cosine of a zero vector). | [modularde.py:344-377](../modde/modularde.py#L344-L377) | Confirmed (code) | Tracking data unreliable in some configs |
| KI-15 | `init_stats` does not work with `AskTellDE`, because `fitness_func` is overridden. | [asktellde.py:62-64](../modde/asktellde.py#L62-L64) | Confirmed (code) | Low |

## B. Deviations from the literature (to verify)

These may be intentional. Each needs a decision (see D-D3): follow the
literature by default, or keep the current behaviour as a named variant.

> **Triage (2026-11-19): all items move to the paper version.** Each change gets a
> short before/after impact report: BBOB d = 5 and 20, (a) the module alone with
> everything else at its default, (b) the L-SHADE configuration with and without the
> change. Study code and reports: [studies/paper_deviations/](studies/paper_deviations/).
>
> KI-21 is confirmed: jSO floors CR (`max`). Checked against the Boost.Math jSO implementation.
> KI-28 is confirmed: L-SHADE uses `N_min = 4`.
>
> Related, not yet an item: L-SHADE and jSO also use the weighted *Lehmer* mean for the
> **CR** memory, whereas SHADE (2013) uses the arithmetic mean. The `shade` module follows
> SHADE. Should the L-SHADE preset use Lehmer for CR (a separate option)? Open question.

| ID | Current behaviour | Literature | Where |
|----|-------------------|------------|-------|
| KI-20 | SHADE F-memory update uses a **weighted arithmetic mean** | SHADE (Tanabe & Fukunaga 2013) uses a **weighted Lehmer mean** for F (arithmetic for CR) | [parameters.py:321-324](../modde/parameters.py#L321-L324) |
| KI-21 | jSO caps: `CR = min(CR, 0.7)` early, `min(CR, 0.6)` later | jSO (Brest et al. 2017) uses `CR = max(CR, 0.7)` / `max(CR, 0.6)`, i.e. CR is floored, not capped. **To verify.** The F cap (`F ≤ 0.7` while < 60% budget) matches. | [parameters.py:421-427](../modde/parameters.py#L421-L427) |
| KI-22 | jDE: new F/CR are adopted for the next generation regardless of trial success | jDE keeps the new F/CR only if the trial vector survives selection | [parameters.py:378-390](../modde/parameters.py#L378-L390) |
| KI-23 | Parents are distinct from the target but **not from each other** | Classic DE requires `r1 ≠ r2 ≠ r3 ≠ i` | [modularde.py:529-532](../modde/modularde.py#L529-L532) |
| KI-24 | `mutation_use_weighted_F` scales **every** difference term | jSO applies `Fw` only to the `pbest − x` term | [modularde.py:119-126](../modde/modularde.py#L119-L126) |
| KI-25 | `pbest_value=p` draws `p_i ~ U(0, p)` per individual | JADE uses a fixed `p`. SHADE draws `p_i ~ U(2/N, 0.2)`, which matches the code's *default* branch. | [modularde.py:131-140](../modde/modularde.py#L131-L140) |
| KI-26 | The archive starts as a copy of the initial population | JADE/SHADE start with an empty archive | [modularde.py:73-74](../modde/modularde.py#L73-L74) |
| KI-27 | The archive difference is `x_r − a_r`, with `a_r` drawn from the archive only. (That the archive term replaces the single difference term when `n_comps=1` matches JADE's current-to-pbest/1 with archive, so that part is *not* a deviation; corrected 2026-11-19.) | JADE: `x_r1 − x̃_r2` with `x̃_r2 ∈ P ∪ A` (union, not archive only), distinct from `i` and `r1` | [modularde.py:168-177](../modde/modularde.py#L168-L177) |
| KI-28 | LPSR minimum population = number of parents needed (3 for the README L-SHADE config) | L-SHADE uses `N_min = 4` | [parameters.py:251-256](../modde/parameters.py#L251-L256) |

## C. Silent parameter overrides and hidden defaults

These are the main targets of WS3's "no silent override" goal.

| ID | Behaviour | Where |
|----|-----------|-------|
| KI-30 | A user-provided `archive_size` is **always overwritten** with `2 * lambda_`, and reset again during LPSR. Confirmed (run): `archive_size=7` → `100`. | [parameters.py:263-265](../modde/parameters.py#L263-L265), [:415-416](../modde/parameters.py#L415-L416) |
| KI-31 | The default `lambda_` changes from `10d` to `18d` when `lpsr=True` | [parameters.py:234-237](../modde/parameters.py#L234-L237) |
| KI-32 | `memory_size` defaults to 100 (SHADE) whenever SHADE is on. L-SHADE uses 6 (the README passes 6 explicitly). | [parameters.py:258-262](../modde/parameters.py#L258-L262) |
| KI-33 | The LPSR end size is derived from the mutation configuration, not settable | [parameters.py:251-256](../modde/parameters.py#L251-L256) |
| KI-34 | Hard-coded constants: jDE `tau_F`, `tau_CR`, `F_l=0.1`, `F_u=0.9`; SHADE sampling scales (0.1); pbest range `[2/n, 0.2]`; weighted-F schedule (0.7/0.8/1.2 at 20%/40%); jSO thresholds; HVB `alpha=0.5`; archive rate 2.0 | various |
| KI-35 | `budget` defaults to `10_000 * d` instead of coming from the problem or caller | [parameters.py:228](../modde/parameters.py#L228) |
| KI-36 | `F`/`CR` must be passed as numpy arrays (a length-1 array is broadcast). A scalar float raises a type error. Adaptive methods overwrite them after the first generation, without saying so. | [parameters.py:239-246](../modde/parameters.py#L239-L246) |
| KI-37 | `oppositional_initialization_factor` is declared and documented but **never used** | [parameters.py:152](../modde/parameters.py#L152) |
| KI-38 | `oversampling_factor`, `oppositional_generation_probability`, `pbest_value`, `memory_size` and `archive_size` are not in `__modules__`, so they are untested | [parameters.py:159-174](../modde/parameters.py#L159-L174) |

## D. Code health and performance (resolved by the port, listed for completeness)

- KI-40: Fitness arrays use `dtype=object`, and every individual is evaluated in its own Python call.
- KI-41: Exponential crossover is a Python double loop.
- KI-42: The LPSR removal step is O(n²) (a list comprehension calling `argsort` on every element).
- KI-43: `Population.__getitem__` copies on every index.
- KI-44: There are two RNG streams: a legacy `np.random.RandomState` plus a scipy QMC `Generator`. Halton output depends on the scipy version.
- KI-45: `bound_correction()` both corrects **and evaluates**, which couples two steps.
- KI-46: Dead code: `vector_target` and `vector_best` correction branches that are not valid options.
- KI-47: Tests only check reproduction of `expected.json` with one module switched on at a time (2D, 200 evaluations). No combinations, no behavioural assertions. CI runs Python 3.8–3.10 with `actions/*@v2`.
