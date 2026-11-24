#pragma once

#include <algorithm>
#include <numeric>
#include <random>

#include "modde/common.hpp"

namespace modde
{
    //! The single random number source of an optimizer instance (D-A13).
    //!
    //! Every random draw in the library goes through this class, so the
    //! distributions can later be replaced by platform-independent ones (FR-4).
    //! Guarantee: same seed -> same draws on the same platform/compiler.
    class Rng
    {
    public:
        explicit Rng(std::uint64_t seed) : engine_(seed) {}

        //! U[0, 1)
        Float uniform() { return unit_(engine_); }

        //! U[a, b)
        Float uniform(Float a, Float b) { return a + (b - a) * uniform(); }

        //! N(mu, sd); returns mu when sd == 0.
        Float normal(Float mu, Float sd)
        {
            if (sd <= 0)
                return mu;
            return std::normal_distribution<Float>(mu, sd)(engine_);
        }

        //! Cauchy(loc, scale); returns loc when scale == 0.
        Float cauchy(Float loc, Float scale)
        {
            if (scale <= 0)
                return loc;
            return std::cauchy_distribution<Float>(loc, scale)(engine_);
        }

        //! Uniform integer in {0, ..., n - 1}; requires n >= 1.
        Index randint(Index n) { return std::uniform_int_distribution<Index>(0, n - 1)(engine_); }

        //! 64 random bits.
        std::uint64_t bits() { return engine_(); }

        //! Random permutation of {0, ..., n - 1} (Fisher-Yates).
        //! Written out instead of std::shuffle: the algorithm of std::shuffle is
        //! implementation-defined and changed between MSVC releases, which made
        //! seeded runs differ between compiler versions.
        Indices permutation(Index n)
        {
            Indices p(static_cast<size_t>(n));
            std::iota(p.begin(), p.end(), Index{0});
            for (Index i = n - 1; i > 0; --i)
                std::swap(p[static_cast<size_t>(i)], p[static_cast<size_t>(randint(i + 1))]);
            return p;
        }

        //! k distinct values from {0, ..., n - 1} that are not in `exclude`
        //! (uniform without replacement, by rejection). Requires enough values.
        Indices sample_distinct(Index n, Index k, const Indices &exclude)
        {
            Indices out;
            out.reserve(static_cast<size_t>(k));
            while (static_cast<Index>(out.size()) < k)
            {
                const Index r = randint(n);
                if (std::find(exclude.begin(), exclude.end(), r) != exclude.end())
                    continue;
                if (std::find(out.begin(), out.end(), r) != out.end())
                    continue;
                out.push_back(r);
            }
            return out;
        }

    private:
        std::mt19937_64 engine_;
        std::uniform_real_distribution<Float> unit_{0.0, 1.0};
    };
} // namespace modde
