// Bound correction: dev/spec/bound_correction.md.

#include <cmath>
#include <stdexcept>

#include "modde/operators.hpp"

namespace modde
{
    namespace
    {
        //! Distance from the bound, Exp(1) truncated to [0, a] (Exp-C / Exp-S).
        Float truncated_exponential(Float a, Rng &rng)
        {
            if (a <= 0)
                return 0;
            const Float rho = rng.uniform();
            return -std::log(1.0 - rho * (1.0 - std::exp(-a)));
        }

        struct NoCorrection final : BoundCorrection
        {
            [[nodiscard]] bool enabled() const override { return false; }
            Float repair(Float y, int, Float, Rng &) const override { return y; }
        };
        struct Saturate final : BoundCorrection
        {
            Float repair(Float, int b, Float, Rng &) const override { return b; }
        };
        struct UniformResample final : BoundCorrection
        {
            Float repair(Float, int, Float, Rng &rng) const override { return rng.uniform(); }
        };
        struct COTN final : BoundCorrection
        {
            Float repair(Float, int b, Float, Rng &rng) const override
            {
                Float r;
                do // KI-17: redraw until inside the domain
                    r = std::abs(b - std::abs(rng.normal(0.0, 1.0 / 3.0)));
                while (r > 1.0);
                return r;
            }
        };
        struct Toroidal final : BoundCorrection
        {
            Float repair(Float y, int, Float, Rng &) const override { return y - std::floor(y); }
        };
        struct Mirror final : BoundCorrection
        {
            Float repair(Float y, int, Float, Rng &) const override
            {
                const Float fl = std::floor(y);
                const Float mod2 = fl - 2.0 * std::floor(fl / 2.0); // Python-style mod (non-negative)
                return std::abs(y - fl - mod2);
            }
        };
        struct HVB final : BoundCorrection
        {
            Float alpha;
            explicit HVB(Float alpha) : alpha(alpha) {}
            Float repair(Float, int b, Float tau, Rng &) const override { return alpha * b + (1.0 - alpha) * tau; }
        };
        struct ExponentialTarget final : BoundCorrection
        {
            Float repair(Float, int b, Float tau, Rng &rng) const override
            {
                const Float delta = truncated_exponential(std::abs(b - tau), rng);
                return b + (1 - 2 * b) * delta;
            }
        };
        struct ExponentialFixed final : BoundCorrection
        {
            Float a; //!< 0.5 = towards the centre (EXPC_CENTER), 1 = whole domain (EXPS)
            explicit ExponentialFixed(Float a) : a(a) {}
            Float repair(Float, int b, Float, Rng &rng) const override
            {
                const Float delta = truncated_exponential(a, rng);
                return b + (1 - 2 * b) * delta;
            }
        };
    } // namespace

    std::shared_ptr<BoundCorrection> make_bound_correction(BoundCorrectionType type, Float hvb_alpha)
    {
        switch (type)
        {
        case BoundCorrectionType::NONE:
            return std::make_shared<NoCorrection>();
        case BoundCorrectionType::SATURATE:
            return std::make_shared<Saturate>();
        case BoundCorrectionType::UNIF_RESAMPLE:
            return std::make_shared<UniformResample>();
        case BoundCorrectionType::COTN:
            return std::make_shared<COTN>();
        case BoundCorrectionType::TOROIDAL:
            return std::make_shared<Toroidal>();
        case BoundCorrectionType::MIRROR:
            return std::make_shared<Mirror>();
        case BoundCorrectionType::HVB:
            return std::make_shared<HVB>(hvb_alpha);
        case BoundCorrectionType::EXPC_TARGET:
            return std::make_shared<ExponentialTarget>();
        case BoundCorrectionType::EXPC_CENTER:
            return std::make_shared<ExponentialFixed>(0.5);
        case BoundCorrectionType::EXPS:
            return std::make_shared<ExponentialFixed>(1.0);
        }
        throw std::invalid_argument("unknown bound correction");
    }

    void apply_bound_correction(Parameters &p, const BoundCorrection &bc)
    {
        const Index n = p.U.cols(), d = p.d;
        const Vector &lb = *p.settings.lb;
        const Vector range = *p.settings.ub - lb;

        p.U_corrected = p.U;
        p.corrected.assign(static_cast<size_t>(n), false);
        p.CS = Vector::Zero(n);
        p.ED = Vector::Zero(n);
        if (!bc.enabled())
            return;

        for (Index i = 0; i < n; ++i)
        {
            bool any = false;
            for (Index j = 0; j < d; ++j)
            {
                const Float y = (p.U(j, i) - lb(j)) / range(j);
                if (y >= 0.0 && y <= 1.0)
                    continue;
                const int b = y > 1.0 ? 1 : 0;
                const Float tau = (p.pop.X(j, i) - lb(j)) / range(j);
                p.U_corrected(j, i) = lb(j) + bc.repair(y, b, tau, p.rng) * range(j);
                any = true;
            }
            p.corrected[static_cast<size_t>(i)] = any;
            if (any)
            {
                // Statistics in normalised space (spec/settings.md §4)
                const Vector y_new = (p.U_corrected.col(i) - lb).cwiseQuotient(range);
                const Vector y_old = (p.U.col(i) - lb).cwiseQuotient(range);
                const Vector y_par = (p.pop.X.col(i) - lb).cwiseQuotient(range);
                p.ED(i) = (y_new - y_old).norm();
                const Vector a = y_new - y_par, c = y_old - y_par;
                const Float denom = a.norm() * c.norm();
                p.CS(i) = denom > 0 ? a.dot(c) / denom : 0.0; // KI-14: 0 instead of NaN
            }
        }
    }
} // namespace modde
