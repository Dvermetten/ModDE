// Generation loop: dev/spec/README.md, initialization.md, selection_archive.md,
// population_size.md, oppositional_generation.md, settings.md §2-4.

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

    Vector ModularDE::evaluate_points(const Matrix &X, Objective &objective,
                                      const std::function<void(Index)> &before)
    {
        if (!objective.single && !objective.batch)
            throw std::invalid_argument("objective has no callable");
        const Index n = X.cols();
        Vector f = Vector::Constant(n, INF);
        const Index count = std::min<Index>(n, static_cast<Index>(p.remaining_budget()));
        if (count == 0)
            return f;

        if (p.settings.track_evaluations || !objective.batch)
        {
            for (Index i = 0; i < count; ++i)
            {
                before(i);
                f(i) = objective.single ? objective.single(X.col(i)) : objective.batch(X.col(i))(0);
                ++p.stats.evaluations;
            }
        }
        else
        {
            const Vector fb = objective.batch(X.leftCols(count));
            if (fb.size() != count)
                throw std::runtime_error("vectorised objective returned " + std::to_string(fb.size()) +
                                         " values for " + std::to_string(count) + " points");
            f.head(count) = fb;
            p.stats.evaluations += static_cast<std::size_t>(count);
        }

        for (Index i = 0; i < count; ++i)
            if (f(i) < p.stats.best_f)
            {
                p.stats.best_f = f(i);
                p.stats.best_x = X.col(i);
            }
        return f;
    }

    void ModularDE::initialize(Objective &objective)
    {
        const auto &s = p.settings;
        const auto &m = p.modules;
        const Vector &lb = *s.lb;
        const Vector range = *s.ub - lb;
        const Index lambda0 = static_cast<Index>(*s.lambda0);

        Matrix candidates;
        Vector f;
        auto init_tracking = [&](Index)
        {
            p.stats.curr_F = *s.F0;
            p.stats.curr_CR = *s.CR0;
            p.stats.corrected = false;
            p.stats.CS = p.stats.ED = 0;
        };

        if (s.initial_population)
        {
            candidates = s.initial_population->transpose();
            f = s.initial_fitness ? *s.initial_fitness : evaluate_points(candidates, objective, init_tracking);
            if (s.initial_fitness)
                for (Index i = 0; i < f.size(); ++i)
                    if (f(i) < p.stats.best_f)
                    {
                        p.stats.best_f = f(i);
                        p.stats.best_x = candidates.col(i);
                    }
        }
        else
        {
            const auto m_total = static_cast<Index>(std::ceil(static_cast<Float>(lambda0) * (1.0 + *s.oversampling_factor)));
            const bool opposition = m.oppositional_initialization != OppositionalInitialization::NONE;
            const Index m_base = opposition ? (m_total + 1) / 2 : m_total;

            const Matrix Z = p.sampler->sample(p.d, m_base, p.rng);
            const Matrix base = (Z.array().colwise() * range.array()).colwise() + lb.array();
            candidates.resize(p.d, m_total);
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
            f = evaluate_points(candidates, objective, init_tracking);
        }

        // keep the best lambda0 (ties: lower index)
        Population all;
        all.X = candidates;
        all.f = f;
        all.F = Vector::Constant(f.size(), *s.F0);
        all.CR = Vector::Constant(f.size(), *s.CR0);
        Indices order = all.argsort();
        order.resize(static_cast<size_t>(lambda0));
        all.keep(order);
        p.pop = std::move(all);
        p.lambda = lambda0;
        p.archive.members.clear();
        p.initialized = true;
        update_generation_stats();
    }

    void ModularDE::sample_parameters() { modde::sample_parameters(p); }

    void ModularDE::mutate() { mutation->apply(p); }

    void ModularDE::crossover() { apply_crossover(p); }

    void ModularDE::correct_bounds() { apply_bound_correction(p, *bound_correction); }

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
    }

    bool ModularDE::oppositional_generation(Objective &objective)
    {
        if (!p.modules.oppositional_generation)
            return false;
        if (!(p.rng.uniform() < *p.settings.oppositional_jump_rate))
            return false;

        const Index n = p.lambda;
        const Vector a = p.pop.X.rowwise().minCoeff();
        const Vector b = p.pop.X.rowwise().maxCoeff();
        const Matrix O = (-p.pop.X).colwise() + (a + b);
        const Vector fO = evaluate_points(O, objective,
                                          [&](Index i)
                                          {
                                              p.stats.curr_F = p.pop.F(i);
                                              p.stats.curr_CR = p.pop.CR(i);
                                              p.stats.corrected = false;
                                              p.stats.CS = p.stats.ED = 0;
                                          });

        // Best n of X u O; stable sort keeps existing members first on ties.
        Population all;
        all.X.resize(p.d, 2 * n);
        all.X << p.pop.X, O;
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
        return true;
    }

    void ModularDE::adapt()
    {
        const auto &m = p.modules;
        const auto &s = p.settings;

        if (m.use_archive)
        {
            for (Index j = 0; j < p.success.replaced.cols(); ++j)
                p.archive.members.emplace_back(p.success.replaced.col(j));
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
        {
            initialize(objective);
            if (break_conditions())
                return false;
        }
        if (break_conditions())
            return false;

        sample_parameters();
        mutate();
        crossover();
        correct_bounds();
        evaluate(objective);
        select(); // evaluated trials are always selected (D-D7)
        ++p.stats.generation;
        if (!break_conditions())
        {
            const bool jumped = oppositional_generation(objective);
            if (!(jumped && break_conditions()))
                adapt();
        }
        update_generation_stats();
        return !break_conditions();
    }

    void ModularDE::run(Objective &objective)
    {
        while (step(objective))
        {
        }
    }

    void ModularDE::update_generation_stats()
    {
        auto &st = p.stats;
        const auto &X = p.pop.X;
        st.popmean = X.mean();
        st.popstd = std::sqrt((X.array() - st.popmean).square().mean());
        st.F_memory_mean = p.modules.shade_F() ? p.memory.M_F.mean() : NaN;
    }
} // namespace modde
