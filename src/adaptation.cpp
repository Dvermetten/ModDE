// F / CR control: dev/spec/adaptation.md.

#include <algorithm>
#include <cmath>

#include "modde/operators.hpp"

namespace modde
{
    void sample_parameters(Parameters &p)
    {
        const auto &m = p.modules;
        const auto &s = p.settings;
        const Index n = p.lambda;
        const Index H = p.memory.M_F.size();
        const bool shade = m.shade_F() || m.shade_CR();

        p.F_used.resize(n);
        p.CR_used.resize(n);
        for (Index i = 0; i < n; ++i)
        {
            const Index r = shade ? p.rng.randint(H) : 0; // one slot for F and CR

            // ---- CR ----
            switch (m.adaptation_method_CR)
            {
            case AdaptationCR::NONE:
                p.CR_used(i) = p.pop.CR(i);
                break;
            case AdaptationCR::SHADE:
            case AdaptationCR::LSHADE:
            {
                const Float mcr = p.memory.M_CR(r);
                p.CR_used(i) = std::isnan(mcr) ? 0.0 : std::clamp(p.rng.normal(mcr, *s.shade_scale_CR), 0.0, 1.0);
                break;
            }
            case AdaptationCR::JDE:
                p.CR_used(i) = p.rng.uniform() < *s.jde_tau_CR ? p.rng.uniform() : p.pop.CR(i);
                break;
            }

            // ---- F ----
            switch (m.adaptation_method_F)
            {
            case AdaptationF::NONE:
                p.F_used(i) = p.pop.F(i);
                break;
            case AdaptationF::SHADE:
            {
                Index rf = r;
                Float x = p.rng.cauchy(p.memory.M_F(rf), *s.shade_scale_F);
                while (x <= 0) // regenerate with a new slot for F only
                {
                    rf = p.rng.randint(H);
                    x = p.rng.cauchy(p.memory.M_F(rf), *s.shade_scale_F);
                }
                p.F_used(i) = std::min<Float>(x, 1.0);
                break;
            }
            case AdaptationF::SHADE_MODIFIED:
            {
                const Float loc = p.memory.M_F.mean();
                Float x = p.rng.cauchy(loc, *s.shade_scale_F);
                while (x <= 0)
                    x = p.rng.cauchy(loc, *s.shade_scale_F);
                p.F_used(i) = std::min<Float>(x, 1.0);
                break;
            }
            case AdaptationF::JDE:
                p.F_used(i) = p.rng.uniform() < *s.jde_tau_F ? *s.jde_F_lower + p.rng.uniform() * *s.jde_F_width
                                                            : p.pop.F(i);
                break;
            }
        }

        // jSO caps on the used values (KI-21)
        if (m.use_jso_caps)
        {
            const Float t = p.progress();
            if (t < 0.25)
                p.CR_used = p.CR_used.cwiseMax(0.7);
            else if (t < 0.5)
                p.CR_used = p.CR_used.cwiseMax(0.6);
            if (t < 0.6)
                p.F_used = p.F_used.cwiseMin(0.7);
        }
    }

    void commit_jde(Parameters &p)
    {
        const auto &m = p.modules;
        for (Index i = 0; i < p.lambda; ++i)
        {
            if (!p.success.W[static_cast<size_t>(i)])
                continue;
            if (m.adaptation_method_F == AdaptationF::JDE)
                p.pop.F(i) = p.F_used(i);
            if (m.adaptation_method_CR == AdaptationCR::JDE)
                p.pop.CR(i) = p.CR_used(i);
        }
    }

    void update_memory(Parameters &p)
    {
        const auto &m = p.modules;
        if (!(m.shade_F() || m.shade_CR()) || p.success.S.empty())
            return;
        const auto &rec = p.success;
        const Vector w = rec.delta_f / rec.delta_f.sum();
        auto &mem = p.memory;

        if (m.shade_F()) // weighted Lehmer mean (KI-20)
            mem.M_F(mem.k) = (w.array() * rec.S_F.array().square()).sum() / (w.array() * rec.S_F.array()).sum();

        if (m.adaptation_method_CR == AdaptationCR::SHADE)
            mem.M_CR(mem.k) = (w.array() * rec.S_CR.array()).sum();
        else if (m.adaptation_method_CR == AdaptationCR::LSHADE)
        {
            if (std::isnan(mem.M_CR(mem.k)) || rec.S_CR.maxCoeff() == 0.0)
                mem.M_CR(mem.k) = NaN; // terminal value
            else
                mem.M_CR(mem.k) =
                    (w.array() * rec.S_CR.array().square()).sum() / (w.array() * rec.S_CR.array()).sum();
        }
        mem.k = (mem.k + 1) % mem.M_F.size();
    }
} // namespace modde
