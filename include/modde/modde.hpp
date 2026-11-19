#pragma once

#include <functional>
#include <memory>

#include "modde/operators.hpp"
#include "modde/parameters.hpp"

namespace modde
{
    //! The objective. At least one of the two must be set (D-A11):
    //! `batch` takes a d x n matrix (one point per column) and returns n values.
    struct Objective
    {
        std::function<Float(const Vector &)> single;
        std::function<Vector(const Matrix &)> batch;
    };

    //! Modular DE with a split step API (D-A10). One generation:
    //! sample_parameters, mutate, crossover, correct_bounds, evaluate, select,
    //! oppositional_generation, adapt (dev/spec/README.md).
    class ModularDE
    {
    public:
        explicit ModularDE(const Settings &settings);

        Parameters p;
        std::shared_ptr<Mutation> mutation;
        std::shared_ptr<BoundCorrection> bound_correction;

        //! Sample, evaluate and select the initial population (spec/initialization.md).
        //! = initial_candidates() + evaluation + finish_initialization().
        void initialize(Objective &objective);

        void sample_parameters();
        void mutate();
        void crossover();
        void correct_bounds();
        //! Evaluate the corrected trials (= evaluation + tell_trials()).
        void evaluate(Objective &objective);
        //! One-to-one selection, success record, jDE commit; counts a generation.
        void select();
        //! Returns true if a jump happened (= draw_opposition() + evaluation + finish_opposition()).
        bool oppositional_generation(Objective &objective);
        void adapt();

        //! One generation (initialises first if needed). Returns false once a
        //! termination condition holds.
        bool step(Objective &objective);
        //! Step until termination.
        void run(Objective &objective);
        [[nodiscard]] bool break_conditions() const;

        // ---- building blocks for ask-tell (D-A12): each evaluation split in two ----
        //! Initial candidates (d x m) to be evaluated; consumes the RNG like initialize().
        Matrix initial_candidates();
        //! Keep the best lambda0 of the candidates; f(i) = +inf for unevaluated ones.
        void finish_initialization(const Matrix &candidates, const Vector &f);
        //! Number of the next n points that the remaining budget allows to evaluate.
        [[nodiscard]] Index evaluable(Index n) const;
        //! Count evaluations of the columns of X (one per entry of f) and update the best.
        void record_evaluations(const Matrix &X, const Vector &f);
        //! Fitness of the first f.size() corrected trials (others stay unevaluated).
        void tell_trials(const Vector &f);
        //! Draw the jump decision; if it jumps, p.opposites holds the points to evaluate.
        bool draw_opposition();
        //! Merge the evaluated opposite points (first fO.size() of p.opposites).
        void finish_opposition(const Vector &fO);
        void update_stats();

    private:
        //! Evaluate the columns of X in order, at most the remaining budget;
        //! unevaluated entries are +inf. `before(i)` runs before evaluating column i
        //! in per-evaluation mode (tracking statistics).
        Vector evaluate_points(const Matrix &X, Objective &objective, const std::function<void(Index)> &before);
    };
} // namespace modde
