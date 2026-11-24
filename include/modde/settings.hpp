#pragma once

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "modde/common.hpp"
#include "modde/modules.hpp"

namespace modde
{
    //! Where a resolved value came from (D-D1, spec/settings.md §1).
    enum class Provenance
    {
        USER,
        PROBLEM,
        DEFAULT,
        DERIVED
    };

    //! User-facing settings. Unset (nullopt) values are filled by resolve().
    //! Names match src/modde/_registry.py; a test checks the defaults agree.
    struct Settings
    {
        std::size_t dim;
        Modules modules;

        // problem / run
        std::optional<std::size_t> budget;
        std::optional<Vector> lb;
        std::optional<Vector> ub;
        std::optional<Float> target;
        std::optional<std::size_t> max_generations;
        std::optional<std::uint64_t> seed;
        bool track_evaluations = false;

        // initialisation
        std::optional<Matrix> initial_population; //!< n x d (rows = individuals), as given
        std::optional<Vector> initial_fitness;
        std::optional<Float> oppositional_initialization_factor;
        std::optional<Float> oversampling_factor;

        // population size
        std::optional<std::size_t> lambda0;
        std::optional<std::size_t> lambda_min;

        // mutation / archive
        std::optional<Float> pbest_rate;
        std::optional<Float> pbest_rate_max;
        std::optional<Float> archive_rate;

        // bound correction
        std::optional<Float> hvb_alpha;

        // F / CR
        std::optional<Float> F0;
        std::optional<Float> CR0;
        std::optional<std::size_t> memory_size;
        std::optional<Float> shade_scale_F;
        std::optional<Float> shade_scale_CR;
        std::optional<Float> jde_tau_F;
        std::optional<Float> jde_F_lower;
        std::optional<Float> jde_F_width;
        std::optional<Float> jde_tau_CR;

        // oppositional generation
        std::optional<Float> oppositional_jump_rate;

        //! Names whose value came from the problem (set by the front end).
        std::set<std::string> problem_fields;

        // Filled by resolve()
        bool resolved = false;
        std::map<std::string, Provenance> provenance;
        std::vector<std::string> warnings;

        explicit Settings(std::size_t dim) : dim(dim) {}

        //! Validate, fill defaults and derived values, record provenance and
        //! warnings. Throws std::invalid_argument on conflicts (D-D1).
        [[nodiscard]] Settings resolve() const;

        //! Distinct population members (besides the target) the mutation needs.
        [[nodiscard]] std::size_t n_parents_needed() const;
    };
} // namespace modde
