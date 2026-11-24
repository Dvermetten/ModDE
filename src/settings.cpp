// Settings resolution: dev/spec/settings.md §1.

#include <algorithm>
#include <random>
#include <sstream>
#include <stdexcept>

#include "modde/sampling.hpp"
#include "modde/settings.hpp"

namespace modde
{
    namespace
    {
        [[noreturn]] void fail(const std::string &msg) { throw std::invalid_argument(msg); }

        template <typename T>
        void check_range(const std::optional<T> &v, const char *name, T lo, std::optional<T> hi = std::nullopt)
        {
            if (!v)
                return;
            if (*v < lo || (hi && *v > *hi))
            {
                std::ostringstream oss;
                oss << name << " = " << *v << " is outside its valid range [" << lo << ", "
                    << (hi ? std::to_string(*hi) : std::string("inf")) << "]";
                fail(oss.str());
            }
        }
    } // namespace

    std::size_t Settings::n_parents_needed() const
    {
        std::size_t n = 2 * modules.mutation_n_comps;
        n += modules.mutation_base == MutationBase::RAND;
        n += modules.mutation_reference == MutationReference::RAND;
        return n;
    }

    Settings Settings::resolve() const
    {
        Settings s = *this;
        s.warnings.clear();
        s.provenance.clear();
        const auto d = dim;
        if (d < 1)
            fail("dim must be >= 1");

        // Record provenance of a field: USER (or PROBLEM) if set, else fill.
        auto mark = [&](const char *name, bool was_set, Provenance fallback)
        {
            if (was_set)
                s.provenance[name] = problem_fields.count(name) ? Provenance::PROBLEM : Provenance::USER;
            else
                s.provenance[name] = fallback;
        };
        auto def = [&](auto &field, const char *name, auto value)
        {
            const bool was_set = field.has_value();
            if (!was_set)
                field = value;
            mark(name, was_set, Provenance::DEFAULT);
        };

        // ---- bounds (D-B2: required, no silent [-5, 5]) ----------------------
        if (!s.lb || !s.ub)
            fail("lb and ub are required (they are taken from the problem for ioh problems)");
        if (static_cast<std::size_t>(s.lb->size()) != d || static_cast<std::size_t>(s.ub->size()) != d)
            fail("lb and ub must have length dim");
        if (!s.lb->allFinite() || !s.ub->allFinite())
            fail("lb and ub must be finite");
        if (!((s.ub->array() - s.lb->array()) > 0).all())
            fail("lb must be strictly smaller than ub in every dimension");
        mark("lb", true, Provenance::USER);
        mark("ub", true, Provenance::USER);

        // ---- defaults ----------------------------------------------------------
        def(s.budget, "budget", static_cast<std::size_t>(10000 * d));
        def(s.lambda0, "lambda0", static_cast<std::size_t>(10 * d));
        def(s.oppositional_initialization_factor, "oppositional_initialization_factor", Float{1.0});
        def(s.oversampling_factor, "oversampling_factor", Float{0.0});
        def(s.pbest_rate, "pbest_rate", Float{0.11});
        def(s.pbest_rate_max, "pbest_rate_max", Float{0.2});
        def(s.archive_rate, "archive_rate", Float{1.0});
        def(s.hvb_alpha, "hvb_alpha", Float{0.5});
        def(s.F0, "F0", Float{0.5});
        def(s.CR0, "CR0", Float{0.9});
        def(s.memory_size, "memory_size", std::size_t{6});
        def(s.shade_scale_F, "shade_scale_F", Float{0.1});
        def(s.shade_scale_CR, "shade_scale_CR", Float{0.1});
        def(s.jde_tau_F, "jde_tau_F", Float{0.1});
        def(s.jde_F_lower, "jde_F_lower", Float{0.1});
        def(s.jde_F_width, "jde_F_width", Float{0.9});
        def(s.jde_tau_CR, "jde_tau_CR", Float{0.1});
        def(s.oppositional_jump_rate, "oppositional_jump_rate", Float{0.3});
        if (s.target)
            mark("target", true, Provenance::USER);
        if (s.max_generations)
            mark("max_generations", true, Provenance::USER);
        {
            const bool was_set = s.seed.has_value();
            if (!was_set)
                s.seed = (static_cast<std::uint64_t>(std::random_device{}()) << 32) ^ std::random_device{}();
            mark("seed", was_set, Provenance::DEFAULT);
        }

        // ---- derived -------------------------------------------------------------
        const std::size_t needed = s.n_parents_needed();
        {
            const bool was_set = s.lambda_min.has_value();
            if (!was_set)
                // never above lambda0: a derived value must not cause an error (spec/population_size.md)
                s.lambda_min = std::min<std::size_t>(*s.lambda0, std::max<std::size_t>(4, needed + 1));
            mark("lambda_min", was_set, Provenance::DERIVED);
        }

        // ---- ranges --------------------------------------------------------------
        const auto &m = s.modules;
        check_range<std::size_t>(s.budget, "budget", 1);
        check_range<std::size_t>(s.lambda0, "lambda0", 1);
        check_range<std::size_t>(s.lambda_min, "lambda_min", 1);
        check_range<std::size_t>(s.memory_size, "memory_size", 1);
        check_range<std::size_t>(s.max_generations, "max_generations", 1);
        check_range<Float>(s.oppositional_initialization_factor, "oppositional_initialization_factor", 0, 1.0);
        check_range<Float>(s.oversampling_factor, "oversampling_factor", 0);
        check_range<Float>(s.pbest_rate, "pbest_rate", 0, 1.0);
        check_range<Float>(s.pbest_rate_max, "pbest_rate_max", 0, 1.0);
        check_range<Float>(s.archive_rate, "archive_rate", 0);
        check_range<Float>(s.hvb_alpha, "hvb_alpha", 0, 1.0);
        check_range<Float>(s.F0, "F0", 0);
        check_range<Float>(s.CR0, "CR0", 0, 1.0);
        check_range<Float>(s.shade_scale_F, "shade_scale_F", 0);
        check_range<Float>(s.shade_scale_CR, "shade_scale_CR", 0);
        check_range<Float>(s.jde_tau_F, "jde_tau_F", 0, 1.0);
        check_range<Float>(s.jde_F_lower, "jde_F_lower", 0);
        check_range<Float>(s.jde_F_width, "jde_F_width", 0);
        check_range<Float>(s.jde_tau_CR, "jde_tau_CR", 0, 1.0);
        check_range<Float>(s.oppositional_jump_rate, "oppositional_jump_rate", 0, 1.0);

        // ---- consistency (errors) -----------------------------------------------
        if (*s.lambda0 < needed + 1)
            fail("lambda0 = " + std::to_string(*s.lambda0) + " is too small: the mutation needs " +
                 std::to_string(needed) + " distinct parents besides the target (lambda0 >= " +
                 std::to_string(needed + 1) + ")");
        if (m.population_reduction == PopulationReduction::LINEAR)
        {
            if (*s.lambda_min < needed + 1)
                fail("lambda_min = " + std::to_string(*s.lambda_min) + " is too small for the mutation (needs >= " +
                     std::to_string(needed + 1) + ")");
            if (*s.lambda_min > *s.lambda0)
                fail("lambda_min = " + std::to_string(*s.lambda_min) + " is larger than lambda0 = " +
                     std::to_string(*s.lambda0));
        }
        if (m.shade_F() && (*s.F0 <= 0 || *s.shade_scale_F <= 0))
            fail("SHADE F adaptation requires F0 > 0 and shade_scale_F > 0");
        if (m.base_sampler == BaseSampler::SOBOL && d > static_cast<std::size_t>(sobol::max_dim))
            fail("SOBOL supports at most " + std::to_string(sobol::max_dim) + " dimensions");

        std::size_t init_evals = *s.lambda0;
        if (s.initial_population)
        {
            const auto &P = *s.initial_population;
            if (static_cast<std::size_t>(P.cols()) != d)
                fail("initial_population must have dim columns (n x d)");
            if (static_cast<std::size_t>(P.rows()) < *s.lambda0)
                fail("initial_population has fewer rows than lambda0");
            for (Index i = 0; i < P.rows(); ++i)
                if (((P.row(i).transpose().array() < s.lb->array()) || (P.row(i).transpose().array() > s.ub->array())).any())
                    fail("initial_population contains points outside the bounds");
            if (s.initial_fitness && s.initial_fitness->size() != P.rows())
                fail("initial_fitness must have one value per row of initial_population");
            init_evals = s.initial_fitness ? 0 : static_cast<std::size_t>(P.rows());
            mark("initial_population", true, Provenance::USER);
        }
        else if (s.initial_fitness)
            fail("initial_fitness requires initial_population");
        if (*s.budget < init_evals)
            fail("budget = " + std::to_string(*s.budget) + " is smaller than the initial population evaluations (" +
                 std::to_string(init_evals) + ")");

        // ---- warnings: user values that have no effect (step 5) ------------------
        auto inactive = [&](const char *name, bool user_set, bool active)
        {
            if (user_set && !active)
                s.warnings.push_back(std::string(name) + " has no effect with the current modules");
        };
        const bool pbest = m.mutation_reference == MutationReference::PBEST;
        inactive("pbest_rate", pbest_rate.has_value(), pbest && m.pbest_sampling == PbestSampling::FIXED);
        inactive("pbest_rate_max", pbest_rate_max.has_value(), pbest && m.pbest_sampling == PbestSampling::RANDOM);
        inactive("archive_rate", archive_rate.has_value(), m.use_archive);
        inactive("hvb_alpha", hvb_alpha.has_value(), m.bound_correction == BoundCorrectionType::HVB);
        inactive("oppositional_initialization_factor", oppositional_initialization_factor.has_value(),
                 m.oppositional_initialization != OppositionalInitialization::NONE);
        inactive("memory_size", memory_size.has_value(), m.shade_F() || m.shade_CR());
        inactive("shade_scale_F", shade_scale_F.has_value(), m.shade_F());
        inactive("shade_scale_CR", shade_scale_CR.has_value(), m.shade_CR());
        inactive("jde_tau_F", jde_tau_F.has_value(), m.adaptation_method_F == AdaptationF::JDE);
        inactive("jde_F_lower", jde_F_lower.has_value(), m.adaptation_method_F == AdaptationF::JDE);
        inactive("jde_F_width", jde_F_width.has_value(), m.adaptation_method_F == AdaptationF::JDE);
        inactive("jde_tau_CR", jde_tau_CR.has_value(), m.adaptation_method_CR == AdaptationCR::JDE);
        inactive("oppositional_jump_rate", oppositional_jump_rate.has_value(), m.oppositional_generation);
        inactive("lambda_min", lambda_min.has_value(), m.population_reduction == PopulationReduction::LINEAR);
        if (m.use_archive && m.mutation_n_comps == 0)
            s.warnings.push_back("use_archive has no effect when mutation_n_comps = 0");
        if (m.mutation_n_comps == 0 && m.mutation_reference == MutationReference::NONE)
            s.warnings.push_back(
                m.mutation_base == MutationBase::BEST
                    ? "mutation_n_comps = 0 with mutation_base = BEST and no mutation_reference: every donor is "
                      "the best individual, so the population collapses (interaction study: conflict)"
                : m.mutation_base == MutationBase::TARGET
                    ? "mutation_n_comps = 0 with mutation_base = TARGET and no mutation_reference: every donor "
                      "equals its target, so the population never changes"
                    : "mutation_n_comps = 0 without a mutation_reference: donors are random population members "
                      "without a difference vector, and the search only recombines existing points "
                      "(interaction study: worse than classic DE)");
        if (m.use_archive && *s.archive_rate == 0)
            s.warnings.push_back("archive_rate = 0: the archive has capacity 0 and stays empty");
        if (m.mutation_use_weighted_F && m.mutation_reference == MutationReference::NONE)
            s.warnings.push_back("mutation_use_weighted_F has no effect when mutation_reference = NONE");
        if (s.initial_population &&
            (m.base_sampler != BaseSampler::UNIFORM ||
             m.oppositional_initialization != OppositionalInitialization::NONE || oversampling_factor.has_value()))
            s.warnings.push_back("sampling modules have no effect when initial_population is given");

        s.resolved = true;
        return s;
    }
} // namespace modde
