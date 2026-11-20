# Reproducibility

**The same seed gives the same result on the same platform with the same modde
version.**

- Results may differ between operating systems and compilers (MSVC, GCC,
  Clang), because the C++ standard library's random distributions differ.
- Even on one platform, a different compiler version or CPU can change the last
  bits of a result (for example through the math functions of the C runtime). On
  functions with plateaus this can occasionally change the course of a run.
  Bit-identical results are guaranteed only for the same build on the same machine.
- Results may change between modde versions; the changelog says when.
- Results never match modde 0.0.4 exactly: the random number generator changed,
  and several bugs and deviations from the original papers were fixed (see the
  [migration guide](migration.md)).

Set the seed with `seed=...`. Without it, a random seed is drawn and stored in
`optimizer.settings.seed`, so any run can be repeated afterwards.

The ask-tell interface follows exactly the same trajectory as `run()` for the
same seed and settings.
