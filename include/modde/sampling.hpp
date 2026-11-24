#pragma once

#include <memory>

#include "modde/common.hpp"
#include "modde/modules.hpp"
#include "modde/rng.hpp"

namespace modde
{
    namespace sobol
    {
        extern const int max_dim;
        extern const unsigned short degree[];
        extern const unsigned int polynomial[];
        extern const unsigned int offset[];
        extern const unsigned int initial[];
    } // namespace sobol

    //! Draws points in [0, 1]^d (spec/initialization.md §2). Python-overridable
    //! hook later (D-A8), hence the virtual interface.
    class Sampler
    {
    public:
        virtual ~Sampler() = default;
        //! Returns a d x m matrix with one point per column.
        virtual Matrix sample(Index d, Index m, Rng &rng) = 0;
    };

    class UniformSampler final : public Sampler
    {
    public:
        Matrix sample(Index d, Index m, Rng &rng) override;
    };

    //! N(0.5, 1/6) per coordinate, redrawn until inside [0, 1] (KI-9).
    class GaussianSampler final : public Sampler
    {
    public:
        Matrix sample(Index d, Index m, Rng &rng) override;
    };

    //! Sobol (Joe & Kuo direction numbers, Gray-code order) with a random
    //! digital shift per dimension. `scramble = false` gives the plain sequence.
    class SobolSampler final : public Sampler
    {
    public:
        explicit SobolSampler(bool scramble = true) : scramble_(scramble) {}
        Matrix sample(Index d, Index m, Rng &rng) override;

    private:
        bool scramble_;
    };

    //! Halton (first d primes, starting at index 1) with a random digit
    //! permutation per dimension (pi(0) = 0). `scramble = false` gives the plain sequence.
    class HaltonSampler final : public Sampler
    {
    public:
        explicit HaltonSampler(bool scramble = true) : scramble_(scramble) {}
        Matrix sample(Index d, Index m, Rng &rng) override;

    private:
        bool scramble_;
    };

    //! Latin hypercube: one random permutation of the m strata per dimension.
    class LhsSampler final : public Sampler
    {
    public:
        Matrix sample(Index d, Index m, Rng &rng) override;
    };

    std::shared_ptr<Sampler> make_sampler(BaseSampler type);
} // namespace modde
