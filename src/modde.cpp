// Generation loop: dev/spec/README.md, initialization.md, selection_archive.md,
// population_size.md, oppositional_generation.md, settings.md §2-4.
//
// Every evaluation is split into "produce points" and "accept fitness" halves
// so that run()/step() and the ask-tell interface share one code path (D-A12).

#include <cmath>
#include <stdexcept>

#include "modde/modde.hpp"

namespace modde
{
    Parameters::Parameters(const Settings &user_settings)
        : settings(user_settings.resolved ? user_settings : user_settings.resolve()),
          modules(settings.modules),
          d(static_cast<Index>(settings.dim)),
          lambda(static_cast<Index>(*settings.lambda0)),
          rng(*settings.seed),
          sampler(make_sampler(settings.modules.base_sampler))
    {
        const auto H = static_cast<Index>(*settings.memory_size);
        memory.M_F = Vector::Constant(H, *settings.F0);
        memory.M_CR = Vector::Constant(H, *settings.CR0);
        stats.best_x = Vector::Constant(d, NaN);
    }

    void Parameters::trim_archive()
    {
        const Index cap = archive_capacity();
        while (archive.size() > cap)
        {
            const auto r = static_cast<size_t>(rng.randint(archive.size()));
            archive.members[r] = std::move(archive.members.back());
            archive.members.pop_back();
        }
    }

    ModularDE::ModularDE(const Settings &settings)
        : p(settings),
          mutation(std::make_shared<Mutation>()),
          bound_correction(make_bound_correction(p.modules.bound_correction, *p.settings.hvb_alpha))
    {
    }

    // ------------------------------------------------------------------ evaluation

    Index ModularDE::evaluable(Index n) const
    {
        return std::min<Index>(n, static_cast<Index>(p.remaining_budget()));
    }

    void ModularDE::record_evaluations(const Matrix &X, const Vector &f)
    {
        p.stats.evaluations += static_cast<std::size_t>(f.size());
        for (Index i = 0; i < f.size(); ++i)
            if (f(i) < p.stats.best_f)
            {
                p.stats.best_f = f(i);
                p.stats.best_x = X.col(i);
            }
    }

    Vector ModularDE::evaluate_points(const Matrix &X, Objective &objective,
                                      const std::function<void(Index)> &before)
    {
        if (!objective.single && !objective.batch)
            throw std::invalid_argument("objective has no callable");
        const Index n = X.cols();
        Vector f = Vector::Constant(n, INF);
        const Index count = evaluable(n);
        if (count == 0)
            return f;

        if (p.settings.track_evaluations || !objective.batch)
        {
            for (Index i = 0; i < count; ++i)
            {
                before(i);
                f(i) = objective.single ? objective.single(X.col(i)) : objective.batch(X.col(i))(0);
                ++p.stats.evaluations; // counted one by one: per-evaluation statistics stay in sync
                if (f(i) < p.stats.best_f)
                {
                    p.stats.best_f = f(i);
                    p.stats.best_x = X.col(i);
                }
            }
        }
        else
        {
            const Vector fb = objective.batch(X.leftCols(count));
            if (fb.size() != count)
                throw std::runtime_error("vectorised objective returned " + std::to_string(fb.size()) +
                                         " values for " + std::to_string(count) + " points");
            f.head(count) = fb;
            record_evaluations(X.leftCols(count), fb);
        }
        return f;
    }

    // ------------------------------------------------------------------ initialisation

    Matrix ModularDE::initial_candidates()
    {
        const auto &s = p.settings;
        const auto &m = p.modules;
        if (s.initial_population)
            return s.initial_population->transpose();

        const Vector &lb = *s.lb;
        const Vector range = *s.ub - lb;
        const auto lambda0 = static_cast<Float>(*s.lambda0);
        const auto m_total = static_cast<Index>(std::ceil(lambda0 * (1.0 + *s.oversampling_factor)));
        const bool opposition = m.oppositional_initialization != OppositionalInitialization::NONE;
        const Index m_base = opposition ? (m_total + 1) / 2 : m_total;

        const Matrix Z = p.sampler->sample(p.d, m_base, p.rng);
        const Matrix base = (Z.array().colwise() * range.array()).colwise() + lb.array();
        Matrix candidates(p.d, m_total);
        candidates.leftCols(m_base) = base;
        if (opposition)
        {
            const Vector c = (lb + *s.ub) / 2.0;
            const Float phi = *s.oppositional_initialization_factor;
            for (Index i = 0; m_base + i < m_total; ++i)
            {
                const Vector opposite = lb + *s.ub - base.col(i);
                Vector x = c + phi * (opposite - c);
                if (m.oppositional_initialization == OppositionalInitialization::QUASI_OPPOSITE)
                    for (Index j = 0; j < p.d; ++j)
                        x(j) = c(j) + p.rng.uniform() * (x(j) - c(j));
                candidates.col(m_base + i) = x;
            }
        }
        return candidates;
    }

    void ModularDE::finish_initialization(const Matrix &candidates, const Vector &f)
    {
        const auto &s = p.settings;
        const auto lambda0 = static_cast<Index>(*s.lambda0);
        if (f.size() != candidates.cols())
            throw std::invalid_argument("one fitness value per candidate is required (+inf if not evaluated)");
        for (Index i = 0; i < f.size(); ++i) // also covers user-supplied initial_fitness (not counted)
            if (f(i) < p.stats.best_f)
            {
                p.stats.best_f = f(i);
                p.stats.best_x = candidates.col(i);
            }

        Population all;
        all.X = candidates;
        all.f = f;
        all.F = Vector::Constant(f.size(), *s.F0);
        all.CR = Vector::Constant(f.size(), *s.CR0);
        Indices order = all.argsort(); // ties: lower index
        order.resize(static_cast<size_t>(lambda0));
        all.keep(order);
        p.pop = std::move(all);
        p.lambda = lambda0;
        p.archive.members.clear();
        p.initialized = true;
        update_stats();
    }

    void ModularDE::initialize(Objective &objective)
    {
        const auto &s = p.settings;
        const Matrix candidates = initial_candidates();
        Vector f;
        if (s.initial_population && s.initial_fitness)
            f = *s.initial_fitness; // known values: not evaluated, not counted
        else
        {
            f = evaluate_points(candidates, objective,
                                [&](Index)
                                {
                                    p.stats.curr_F = *s.F0;
                                    p.stats.curr_CR = *s.CR0;
                                    p.stats.corrected = false;
                                    p.stats.CS = p.stats.ED = 0;
                                });
        }
        finish_initialization(candidates, f);
    }

    // ------------------------------------------------------------------ generation

    void ModularDE::sample_parameters() { modde::sample_parameters(p); }

    void ModularDE::mutate() { mutation->apply(p); }

    void ModularDE::crossover() { apply_crossover(p); }

    void ModularDE::correct_bounds() { apply_bound_correction(p, *bound_correction); }

    void ModularDE::tell_trials(const Vector &f)
    {
        if (f.size() > p.lambda)
            throw std::invalid_argument("more fitness values than trials");
        p.f_trial = Vector::Constant(p.lambda, INF);
        p.f_trial.head(f.size()) = f;
        record_evaluations(p.U_corrected.leftCols(f.size()), f);
    }

    void ModularDE::evaluate(Objective &objective)
    {
        p.f_trial = evaluate_points(p.U_corrected, objective,
                                    [&](Index i)
                                    {
                                        auto &st = p.stats;
                                        st.curr_F = p.F_used(i);
                                        st.curr_CR = p.CR_used(i);
                                        st.corrected = p.corrected[static_cast<size_t>(i)];
                                        st.corr_so_far += st.corrected;
                                        st.CS = p.CS(i);
                                        st.ED = p.ED(i);
                                    });
    }

    void ModularDE::select()
    {
        const Index n = p.lambda;
        auto &rec = p.success;
        rec.S.clear();
        rec.W.assign(static_cast<size_t>(n), false);
        std::vector<Float> sF, sCR, df;
        std::vector<Vector> replaced;

        for (Index i = 0; i < n; ++i)
        {
            const Float ft = p.f_trial(i);
            if (!(ft <= p.pop.f(i)))
                continue;
            rec.W[static_cast<size_t>(i)] = true;
            if (ft < p.pop.f(i))
            {
                rec.S.push_back(i);
                sF.push_back(p.F_used(i));
                sCR.push_back(p.CR_used(i));
                df.push_back(p.pop.f(i) - ft);
                replaced.push_back(p.pop.X.col(i));
            }
            p.pop.X.col(i) = p.U_corrected.col(i);
            p.pop.f(i) = ft;
        }
        const auto k = static_cast<Index>(rec.S.size());
        rec.S_F = Eigen::Map<Vector>(sF.data(), k);
        rec.S_CR = Eigen::Map<Vector>(sCR.data(), k);
        rec.delta_f = Eigen::Map<Vector>(df.data(), k);
        rec.replaced.resize(p.d, k);
        for (Index j = 0; j < k; ++j)
            rec.replaced.col(j) = replaced[static_cast<size_t>(j)];

        commit_jde(p); // before any reordering
        ++p.stats.generation;
    }

    bool ModularDE::draw_opposition()
    {
        if (!p.modules.oppositional_generation)
            return false;
        if (!(p.rng.uniform() < *p.settings.oppositional_jump_rate))
            return false;
        const Vector a = p.pop.X.rowwise().minCoeff();
        const Vector b = p.pop.X.rowwise().maxCoeff();
        p.opposites = (-p.pop.X).colwise() + (a + b);
        return true;
    }

    void ModularDE::finish_opposition(const Vector &fO_evaluated)
    {
        const Index n = p.lambda;
        if (fO_evaluated.size() > n)
            throw std::invalid_argument("more fitness values than opposite points");
        Vector fO = Vector::Constant(n, INF);
        fO.head(fO_evaluated.size()) = fO_evaluated;

        // Best n of X u O; the stable sort keeps existing members first on ties.
        Population all;
        all.X.resize(p.d, 2 * n);
        all.X << p.pop.X, p.opposites;
        all.f.resize(2 * n);
        all.f << p.pop.f, fO;
        all.F.resize(2 * n);
        all.F << p.pop.F, p.pop.F; // opposite points inherit F_i, CR_i
        all.CR.resize(2 * n);
        all.CR << p.pop.CR, p.pop.CR;
        Indices order = all.argsort();
        order.resize(static_cast<size_t>(n));
        all.keep(order);
        p.pop = std::move(all);
    }

    bool ModularDE::oppositional_generation(Objective &objective)
    {
        if (!draw_opposition())
            return false;
        const Vector fO = evaluate_points(p.opposites, objective,
                                          [&](Index i)
                                          {
                                              p.stats.curr_F = p.pop.F(i);
                                              p.stats.curr_CR = p.pop.CR(i);
                                              p.stats.corrected = false;
                                              p.stats.CS = p.stats.ED = 0;
                                          });
        finish_opposition(fO);
        return true;
    }

    void ModularDE::adapt()
    {
        const auto &m = p.modules;
        const auto &s = p.settings;

        if (m.use_archive)
        {
            // PARENTS: the replaced parents (papers); TRIALS: the successful trial vectors
            // (Tanabe's L-SHADE code, KI-48, D-D15). U_corrected is unchanged since select().
            for (Index j = 0; j < p.success.replaced.cols(); ++j)
                p.archive.members.emplace_back(m.archive_content == ArchiveContent::TRIALS
                                                   ? Vector(p.U_corrected.col(p.success.S[static_cast<size_t>(j)]))
                                                   : Vector(p.success.replaced.col(j)));
            p.trim_archive();
        }

        update_memory(p);

        if (m.population_reduction == PopulationReduction::LINEAR)
        {
            const auto l0 = static_cast<Float>(*s.lambda0), lmin = static_cast<Float>(*s.lambda_min);
            const auto next = static_cast<Index>(std::llround(l0 + (lmin - l0) * p.progress()));
            if (next < p.lambda)
            {
                Indices keep = p.pop.argsort(); // ascending f, ties: lower index first
                keep.resize(static_cast<size_t>(next));
                std::sort(keep.begin(), keep.end()); // preserve the original order
                p.pop.keep(keep);
                p.lambda = next;
                if (m.use_archive)
                    p.trim_archive();
            }
        }
    }

    bool ModularDE::break_conditions() const
    {
        const auto &s = p.settings;
        if (p.stats.evaluations >= *s.budget)
            return true;
        if (s.target && p.stats.best_f <= *s.target)
            return true;
        if (s.max_generations && p.stats.generation >= *s.max_generations)
            return true;
        return false;
    }

    bool ModularDE::step(Objective &objective)
    {
        if (!p.initialized)
            initialize(objective);
        if (break_conditions())
            return false;

        sample_parameters();
        mutate();
        crossover();
        correct_bounds();
        evaluate(objective);
        select(); // evaluated trials are always selected (D-D7)
        if (!break_conditions())
        {
            const bool jumped = oppositional_generation(objective);
            if (!(jumped && break_conditions()))
                adapt();
        }
        update_stats();
        return !break_conditions();
    }

    void ModularDE::run(Objective &objective)
    {
        while (step(objective))
        {
        }
    }

    void ModularDE::update_stats()
    {
        auto &st = p.stats;
        const auto &X = p.pop.X;
        st.popmean = X.mean();
        st.popstd = std::sqrt((X.array() - st.popmean).square().mean());
        st.F_memory_mean = p.modules.shade_F() ? p.memory.M_F.mean() : NaN;
    }
} // namespace modde
