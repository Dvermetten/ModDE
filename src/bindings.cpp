// Python bindings for the modde C++ core (module: modde._core).
//
// Phase 2: the full core with its split step API. The user-facing API
// (fmin, presets, ioh wrapping, ConfigSpace) is added in Phase 3 in Python.
// Populations cross the boundary as n x d arrays (D-A9).

#include <string>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "modde/modde.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace modde;

namespace
{
    std::string compiler_id()
    {
#if defined(_MSC_VER)
        return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
        return "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__);
#elif defined(__GNUC__)
        return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#else
        return "unknown";
#endif
    }

    Vector individual_norms(const Eigen::Ref<const RowMatrix> &X)
    {
        const Matrix population = X.transpose();
        return population.colwise().norm().transpose();
    }

    RowMatrix to_rows(const Matrix &X) { return X.transpose(); }

    Objective make_objective(nb::callable f, bool vectorized)
    {
        Objective o;
        if (vectorized)
        {
            nb::object asarray = nb::module_::import_("numpy").attr("asarray");
            o.batch = [f, asarray](const Matrix &X) -> Vector
            {
                RowMatrix Xt = X.transpose();
                nb::object res = f(nb::cast(Xt, nb::rv_policy::copy));
                nb::object arr = asarray(res, "dtype"_a = "float64").attr("ravel")();
                return nb::cast<Vector>(arr);
            };
        }
        else
        {
            o.single = [f](const Vector &x) -> Float { return nb::cast<Float>(f(nb::cast(x, nb::rv_policy::copy))); };
        }
        return o;
    }

    std::shared_ptr<Sampler> sampler_by_name(const std::string &name, bool scramble)
    {
        if (name == "UNIFORM")
            return std::make_shared<UniformSampler>();
        if (name == "GAUSSIAN")
            return std::make_shared<GaussianSampler>();
        if (name == "SOBOL")
            return std::make_shared<SobolSampler>(scramble);
        if (name == "HALTON")
            return std::make_shared<HaltonSampler>(scramble);
        if (name == "LHS")
            return std::make_shared<LhsSampler>();
        throw std::invalid_argument("unknown sampler " + name);
    }

    template <typename E>
    void bind_enum(nb::module_ &m, const char *name, std::initializer_list<std::pair<const char *, E>> values)
    {
        auto e = nb::enum_<E>(m, name);
        for (const auto &[n, v] : values)
            e.value(n, v);
    }
} // namespace

NB_MODULE(_core, m)
{
    m.doc() = "modde C++ core.";

    // ---- build / toolchain ----------------------------------------------------
    m.def("build_info", []()
          {
              nb::dict info;
              info["compiler"] = compiler_id();
              info["cplusplus"] = static_cast<long>(__cplusplus);
#if defined(_MSVC_LANG)
              info["msvc_lang"] = static_cast<long>(_MSVC_LANG);
#endif
              info["eigen"] = std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) + "." +
                              std::to_string(EIGEN_MINOR_VERSION);
#if defined(NB_VERSION_MAJOR)
              info["nanobind"] = std::to_string(NB_VERSION_MAJOR) + "." + std::to_string(NB_VERSION_MINOR) + "." +
                                 std::to_string(NB_VERSION_PATCH);
#endif
              return info; },
          "Information about how the extension was compiled.");
    m.def("individual_norms", &individual_norms, "X"_a, "Toolchain check: Euclidean norm of each row of an (n x d) array.");

    // ---- enums --------------------------------------------------------------------
    bind_enum<BaseSampler>(m, "BaseSampler", {{"UNIFORM", BaseSampler::UNIFORM}, {"GAUSSIAN", BaseSampler::GAUSSIAN}, {"SOBOL", BaseSampler::SOBOL}, {"HALTON", BaseSampler::HALTON}, {"LHS", BaseSampler::LHS}});
    bind_enum<OppositionalInitialization>(m, "OppositionalInitialization", {{"NONE", OppositionalInitialization::NONE}, {"OPPOSITE", OppositionalInitialization::OPPOSITE}, {"QUASI_OPPOSITE", OppositionalInitialization::QUASI_OPPOSITE}});
    bind_enum<MutationBase>(m, "MutationBase", {{"RAND", MutationBase::RAND}, {"BEST", MutationBase::BEST}, {"TARGET", MutationBase::TARGET}});
    bind_enum<MutationReference>(m, "MutationReference", {{"NONE", MutationReference::NONE}, {"RAND", MutationReference::RAND}, {"BEST", MutationReference::BEST}, {"PBEST", MutationReference::PBEST}});
    bind_enum<PbestSampling>(m, "PbestSampling", {{"RANDOM", PbestSampling::RANDOM}, {"FIXED", PbestSampling::FIXED}});
    bind_enum<CrossoverType>(m, "CrossoverType", {{"BIN", CrossoverType::BIN}, {"EXP", CrossoverType::EXP}});
    bind_enum<BoundCorrectionType>(m, "BoundCorrectionType", {{"SATURATE", BoundCorrectionType::SATURATE}, {"NONE", BoundCorrectionType::NONE}, {"UNIF_RESAMPLE", BoundCorrectionType::UNIF_RESAMPLE}, {"COTN", BoundCorrectionType::COTN}, {"TOROIDAL", BoundCorrectionType::TOROIDAL}, {"MIRROR", BoundCorrectionType::MIRROR}, {"HVB", BoundCorrectionType::HVB}, {"EXPC_TARGET", BoundCorrectionType::EXPC_TARGET}, {"EXPC_CENTER", BoundCorrectionType::EXPC_CENTER}, {"EXPS", BoundCorrectionType::EXPS}});
    bind_enum<AdaptationF>(m, "AdaptationF", {{"NONE", AdaptationF::NONE}, {"SHADE", AdaptationF::SHADE}, {"SHADE_MODIFIED", AdaptationF::SHADE_MODIFIED}, {"JDE", AdaptationF::JDE}});
    bind_enum<AdaptationCR>(m, "AdaptationCR", {{"NONE", AdaptationCR::NONE}, {"SHADE", AdaptationCR::SHADE}, {"LSHADE", AdaptationCR::LSHADE}, {"JDE", AdaptationCR::JDE}});
    bind_enum<PopulationReduction>(m, "PopulationReduction", {{"NONE", PopulationReduction::NONE}, {"LINEAR", PopulationReduction::LINEAR}});
    bind_enum<Provenance>(m, "Provenance", {{"USER", Provenance::USER}, {"PROBLEM", Provenance::PROBLEM}, {"DEFAULT", Provenance::DEFAULT}, {"DERIVED", Provenance::DERIVED}});

    // ---- Modules / Settings -------------------------------------------------------
    nb::class_<Modules>(m, "Modules")
        .def(nb::init<>())
        .def_rw("base_sampler", &Modules::base_sampler)
        .def_rw("oppositional_initialization", &Modules::oppositional_initialization)
        .def_rw("mutation_base", &Modules::mutation_base)
        .def_rw("mutation_reference", &Modules::mutation_reference)
        .def_rw("mutation_n_comps", &Modules::mutation_n_comps)
        .def_rw("pbest_sampling", &Modules::pbest_sampling)
        .def_rw("mutation_use_weighted_F", &Modules::mutation_use_weighted_F)
        .def_rw("use_archive", &Modules::use_archive)
        .def_rw("crossover", &Modules::crossover)
        .def_rw("eigenvalue_crossover", &Modules::eigenvalue_crossover)
        .def_rw("bound_correction", &Modules::bound_correction)
        .def_rw("adaptation_method_F", &Modules::adaptation_method_F)
        .def_rw("adaptation_method_CR", &Modules::adaptation_method_CR)
        .def_rw("use_jso_caps", &Modules::use_jso_caps)
        .def_rw("population_reduction", &Modules::population_reduction)
        .def_rw("oppositional_generation", &Modules::oppositional_generation);

    nb::class_<Settings>(m, "Settings")
        .def(nb::init<std::size_t>(), "dim"_a)
        .def_rw("dim", &Settings::dim)
        .def_rw("modules", &Settings::modules)
        .def_rw("budget", &Settings::budget)
        .def_rw("lb", &Settings::lb)
        .def_rw("ub", &Settings::ub)
        .def_rw("target", &Settings::target)
        .def_rw("max_generations", &Settings::max_generations)
        .def_rw("seed", &Settings::seed)
        .def_rw("track_evaluations", &Settings::track_evaluations)
        .def_rw("initial_population", &Settings::initial_population)
        .def_rw("initial_fitness", &Settings::initial_fitness)
        .def_rw("oppositional_initialization_factor", &Settings::oppositional_initialization_factor)
        .def_rw("oversampling_factor", &Settings::oversampling_factor)
        .def_rw("lambda0", &Settings::lambda0)
        .def_rw("lambda_min", &Settings::lambda_min)
        .def_rw("pbest_rate", &Settings::pbest_rate)
        .def_rw("pbest_rate_max", &Settings::pbest_rate_max)
        .def_rw("archive_rate", &Settings::archive_rate)
        .def_rw("hvb_alpha", &Settings::hvb_alpha)
        .def_rw("F0", &Settings::F0)
        .def_rw("CR0", &Settings::CR0)
        .def_rw("memory_size", &Settings::memory_size)
        .def_rw("shade_scale_F", &Settings::shade_scale_F)
        .def_rw("shade_scale_CR", &Settings::shade_scale_CR)
        .def_rw("jde_tau_F", &Settings::jde_tau_F)
        .def_rw("jde_F_lower", &Settings::jde_F_lower)
        .def_rw("jde_F_width", &Settings::jde_F_width)
        .def_rw("jde_tau_CR", &Settings::jde_tau_CR)
        .def_rw("oppositional_jump_rate", &Settings::oppositional_jump_rate)
        .def_rw("problem_fields", &Settings::problem_fields)
        .def_ro("resolved", &Settings::resolved)
        .def_ro("provenance", &Settings::provenance)
        .def_ro("warnings", &Settings::warnings)
        .def("resolve", &Settings::resolve, "Validate, fill defaults/derived values and record provenance.")
        .def("n_parents_needed", &Settings::n_parents_needed);

    nb::class_<Stats>(m, "Stats")
        .def_ro("generation", &Stats::generation)
        .def_rw("evaluations", &Stats::evaluations, "Counted evaluations; the front end may set an offset (D-B3).")
        .def_ro("best_x", &Stats::best_x)
        .def_ro("best_f", &Stats::best_f)
        .def_ro("popmean", &Stats::popmean)
        .def_ro("popstd", &Stats::popstd)
        .def_ro("F_memory_mean", &Stats::F_memory_mean)
        .def_ro("curr_F", &Stats::curr_F)
        .def_ro("curr_CR", &Stats::curr_CR)
        .def_ro("corrected", &Stats::corrected)
        .def_ro("CS", &Stats::CS)
        .def_ro("ED", &Stats::ED)
        .def_ro("corr_so_far", &Stats::corr_so_far);

    // ---- optimizer --------------------------------------------------------------------
    using DE = ModularDE;
    auto with_objective = [](void (DE::*method)(Objective &))
    {
        return [method](DE &de, nb::callable f, bool vectorized)
        {
            Objective o = make_objective(f, vectorized);
            (de.*method)(o);
        };
    };

    nb::class_<DE>(m, "ModularDE")
        .def(nb::init<const Settings &>(), "settings"_a)
        .def("initialize", with_objective(&DE::initialize), "f"_a, "vectorized"_a = false)
        .def("sample_parameters", &DE::sample_parameters)
        .def("mutate", &DE::mutate)
        .def("crossover", &DE::crossover)
        .def("correct_bounds", &DE::correct_bounds)
        .def("evaluate", with_objective(&DE::evaluate), "f"_a, "vectorized"_a = false)
        .def("select", &DE::select)
        .def("oppositional_generation", [](DE &de, nb::callable f, bool vectorized)
             { Objective o = make_objective(f, vectorized); return de.oppositional_generation(o); }, "f"_a, "vectorized"_a = false)
        .def("adapt", &DE::adapt)
        .def("step", [](DE &de, nb::callable f, bool vectorized)
             { Objective o = make_objective(f, vectorized); return de.step(o); }, "f"_a, "vectorized"_a = false)
        .def("run", with_objective(&DE::run), "f"_a, "vectorized"_a = false)
        .def("break_conditions", &DE::break_conditions)
        // ask-tell building blocks (D-A12); n x d at the boundary
        .def("initial_candidates", [](DE &de) { return to_rows(de.initial_candidates()); },
             "Initial candidates (m x d) to evaluate; consumes the RNG like initialize().")
        .def("finish_initialization", [](DE &de, const RowMatrix &C, const Vector &f)
             { de.finish_initialization(C.transpose(), f); }, "candidates"_a, "f"_a,
             "Keep the best lambda0 candidates; f = +inf for unevaluated ones.")
        .def("evaluable", &DE::evaluable, "n"_a, "How many of the next n points the remaining budget allows.")
        .def("record_evaluations", [](DE &de, const RowMatrix &X, const Vector &f)
             { de.record_evaluations(X.transpose(), f); }, "X"_a, "f"_a,
             "Count evaluations of the rows of X and update the best solution.")
        .def("tell_trials", &DE::tell_trials, "f"_a, "Fitness of the first len(f) corrected trials.")
        .def("draw_opposition", &DE::draw_opposition, "Draw the jump decision; True if `opposites` must be evaluated.")
        .def_prop_ro("opposites", [](const DE &de) { return to_rows(de.p.opposites); })
        .def("finish_opposition", &DE::finish_opposition, "f"_a, "Merge the first len(f) evaluated opposite points.")
        .def("update_stats", &DE::update_stats)
        // state (n x d views are copies)
        .def_prop_ro("settings", [](const DE &de) { return de.p.settings; })
        .def_prop_ro("stats", [](DE &de) -> Stats & { return de.p.stats; }, nb::rv_policy::reference_internal)
        .def_prop_ro("lambda_", [](const DE &de) { return de.p.lambda; })
        .def_prop_ro("initialized", [](const DE &de) { return de.p.initialized; })
        .def_prop_ro("progress", [](const DE &de) { return de.p.progress(); })
        .def_prop_rw("population", [](const DE &de) { return to_rows(de.p.pop.X); },
                     [](DE &de, const RowMatrix &X)
                     {
                         if (X.rows() != de.p.lambda || X.cols() != de.p.d)
                             throw std::invalid_argument("population must have shape (lambda, d)");
                         de.p.pop.X = X.transpose();
                     })
        .def_prop_rw("fitness", [](const DE &de) { return de.p.pop.f; }, [](DE &de, const Vector &f) { de.p.pop.f = f; })
        .def_prop_rw("F", [](const DE &de) { return de.p.pop.F; }, [](DE &de, const Vector &v) { de.p.pop.F = v; })
        .def_prop_rw("CR", [](const DE &de) { return de.p.pop.CR; }, [](DE &de, const Vector &v) { de.p.pop.CR = v; })
        .def_prop_rw("F_used", [](const DE &de) { return de.p.F_used; }, [](DE &de, const Vector &v) { de.p.F_used = v; })
        .def_prop_rw("CR_used", [](const DE &de) { return de.p.CR_used; }, [](DE &de, const Vector &v) { de.p.CR_used = v; })
        .def_prop_rw("donors", [](const DE &de) { return to_rows(de.p.V); }, [](DE &de, const RowMatrix &X) { de.p.V = X.transpose(); })
        .def_prop_rw("trials", [](const DE &de) { return to_rows(de.p.U); }, [](DE &de, const RowMatrix &X) { de.p.U = X.transpose(); })
        .def_prop_ro("trials_corrected", [](const DE &de) { return to_rows(de.p.U_corrected); })
        .def_prop_rw("trial_fitness", [](const DE &de) { return de.p.f_trial; }, [](DE &de, const Vector &f) { de.p.f_trial = f; })
        .def_prop_ro("corrected", [](const DE &de)
                     { nb::list l; for (bool b : de.p.corrected) l.append(b); return l; })
        .def_prop_ro("last_parents", [](const DE &de) { return de.p.last_parents; })
        .def_prop_ro("archive", [](const DE &de)
                     {
                         RowMatrix A(de.p.archive.size(), de.p.d);
                         for (Index i = 0; i < de.p.archive.size(); ++i)
                             A.row(i) = de.p.archive.members[static_cast<size_t>(i)].transpose();
                         return A; })
        .def_prop_ro("archive_capacity", [](const DE &de) { return de.p.archive_capacity(); })
        .def_prop_rw("memory_F", [](const DE &de) { return de.p.memory.M_F; }, [](DE &de, const Vector &v) { de.p.memory.M_F = v; })
        .def_prop_rw("memory_CR", [](const DE &de) { return de.p.memory.M_CR; }, [](DE &de, const Vector &v) { de.p.memory.M_CR = v; })
        .def_prop_ro("memory_index", [](const DE &de) { return de.p.memory.k; })
        .def_prop_ro("success", [](const DE &de)
                     {
                         const auto &r = de.p.success;
                         nb::dict out;
                         out["S"] = r.S;
                         out["S_F"] = r.S_F;
                         out["S_CR"] = r.S_CR;
                         out["delta_f"] = r.delta_f;
                         out["replaced"] = to_rows(r.replaced);
                         nb::list W;
                         for (bool b : r.W)
                             W.append(b);
                         out["W"] = W;
                         return out; });

    // ---- testing hooks -----------------------------------------------------------------
    m.def("_sample", [](const std::string &name, Index d, Index n, std::uint64_t seed, bool scramble)
          {
              Rng rng(seed);
              return to_rows(sampler_by_name(name, scramble)->sample(d, n, rng)); },
          "sampler"_a, "d"_a, "n"_a, "seed"_a = 0, "scramble"_a = true,
          "Draw n points in [0,1]^d with a sampler (testing hook). Returns n x d.");
}
