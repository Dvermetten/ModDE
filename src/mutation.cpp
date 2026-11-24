// Mutation: dev/spec/mutation.md.

#include <algorithm>
#include <cmath>

#include "modde/operators.hpp"

namespace modde
{
    void Mutation::apply(Parameters &p)
    {
        const auto &m = p.modules;
        const auto &s = p.settings;
        const Matrix &X = p.pop.X;
        const Index n = p.lambda;
        const auto n_comps = static_cast<Index>(m.mutation_n_comps);
        const bool archive_term = m.use_archive && n_comps >= 1;
        const Index n_draw = static_cast<Index>(s.n_parents_needed()) - (archive_term ? 1 : 0);

        const Index best = p.pop.best();
        Indices order;
        if (m.mutation_reference == MutationReference::PBEST)
            order = p.pop.argsort();

        // Weighted F (jSO) on the reference term only (KI-24)
        Float w = 1.0;
        if (m.mutation_use_weighted_F)
        {
            const Float t = p.progress();
            w = t < 0.2 ? 0.7 : (t < 0.4 ? 0.8 : 1.2);
        }

        p.V.resize(p.d, n);
        p.last_parents.assign(static_cast<size_t>(n), {});
        for (Index i = 0; i < n; ++i)
        {
            // Mutually distinct parents, all != i (KI-23)
            const Indices parents = p.rng.sample_distinct(n, n_draw, {i});
            size_t pos = 0;
            const Float Fi = p.F_used(i);

            auto v = p.V.col(i); // written in place, no temporaries
            switch (m.mutation_base)
            {
            case MutationBase::RAND:
                v = X.col(parents[pos++]);
                break;
            case MutationBase::BEST:
                v = X.col(best);
                break;
            case MutationBase::TARGET:
                v = X.col(i);
                break;
            }

            if (m.mutation_reference != MutationReference::NONE)
            {
                Index ref = best;
                switch (m.mutation_reference)
                {
                case MutationReference::RAND:
                    ref = parents[pos++];
                    break;
                case MutationReference::BEST:
                    ref = best;
                    break;
                case MutationReference::PBEST:
                {
                    const Float lo = 2.0 / static_cast<Float>(n);
                    const Float p_i = m.pbest_sampling == PbestSampling::FIXED
                                          ? *s.pbest_rate
                                          : p.rng.uniform(lo, std::max(lo, *s.pbest_rate_max));
                    const Index k = std::clamp<Index>(std::llround(p_i * static_cast<Float>(n)), 1, n);
                    ref = order[static_cast<size_t>(p.rng.randint(k))];
                    break;
                }
                case MutationReference::NONE:
                    break;
                }
                v += (w * Fi) * (X.col(ref) - X.col(i));
            }

            for (Index c = 0; c < n_comps; ++c) // KI-1: every term has its own pair
            {
                const Index r1 = parents[pos++];
                if (c == 0 && archive_term)
                {
                    // x~_r2 uniformly from P u A; population draws must differ from i and the parents (KI-27)
                    const Index total = n + p.archive.size();
                    Index r;
                    while (true)
                    {
                        r = p.rng.randint(total);
                        if (r >= n)
                            break;
                        if (r != i && std::find(parents.begin(), parents.end(), r) == parents.end())
                            break;
                    }
                    if (r < n)
                        v += Fi * (X.col(r1) - X.col(r));
                    else
                        v += Fi * (X.col(r1) - p.archive.members[static_cast<size_t>(r - n)]);
                }
                else
                {
                    const Index r2 = parents[pos++];
                    v += Fi * (X.col(r1) - X.col(r2));
                }
            }
            p.last_parents[static_cast<size_t>(i)] = parents;
        }
    }
} // namespace modde
