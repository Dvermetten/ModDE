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
        void initialize(Objective &objective);

        void sample_parameters();
        void mutate();
        void crossover();
        void correct_bounds();
        void evaluate(Objective &objective);
        void select();
        //! Returns true if a jump happened.
        bool oppositional_generation(Objective &objective);
        void adapt();

        //! One generation (initialises first if needed). Returns false once a
        //! termination condition holds.
        bool step(Objective &objective);
        //! Step until termination.
        void run(Objective &objective);
        [[nodiscard]] bool break_conditions() const;

    private:
        //! Evaluate the columns of X in order, at most the remaining budget;
        //! unevaluated entries are +inf. `before(i)` runs before evaluating column i
        //! in per-evaluation mode (tracking statistics).
        Vector evaluate_points(const Matrix &X, Objective &objective, const std::function<void(Index)> &before);
        void update_generation_stats();
    };
} // namespace modde
