// Samplers: dev/spec/initialization.md §2.

#include <cmath>
#include <numeric>
#include <stdexcept>

#include "modde/sampling.hpp"

namespace modde
{
    Matrix UniformSampler::sample(Index d, Index m, Rng &rng)
    {
        Matrix Z(d, m);
        for (Index i = 0; i < m; ++i)
            for (Index j = 0; j < d; ++j)
                Z(j, i) = rng.uniform();
        return Z;
    }

    Matrix GaussianSampler::sample(Index d, Index m, Rng &rng)
    {
        Matrix Z(d, m);
        for (Index i = 0; i < m; ++i)
            for (Index j = 0; j < d; ++j)
            {
                Float z;
                do
                    z = rng.normal(0.5, 1.0 / 6.0);
                while (z < 0.0 || z > 1.0);
                Z(j, i) = z;
            }
        return Z;
    }

    Matrix SobolSampler::sample(Index d, Index m, Rng &rng)
    {
        if (d > sobol::max_dim)
            throw std::invalid_argument("SOBOL supports at most " + std::to_string(sobol::max_dim) + " dimensions");
        constexpr int bits = 32;
        if (m > (Index{1} << bits) - 1)
            throw std::invalid_argument("too many Sobol points requested");

        // Direction numbers V[j][k] (k = 1..bits), scaled to 32 bits.
        std::vector<std::vector<std::uint32_t>> V(static_cast<size_t>(d), std::vector<std::uint32_t>(bits + 1));
        for (int k = 1; k <= bits; ++k)
            V[0][k] = std::uint32_t{1} << (bits - k);
        for (Index j = 1; j < d; ++j)
        {
            const auto e = static_cast<size_t>(j - 1);
            const int s = sobol::degree[e];
            const unsigned a = sobol::polynomial[e];
            const unsigned *mi = &sobol::initial[sobol::offset[e]];
            auto &v = V[static_cast<size_t>(j)];
            for (int k = 1; k <= std::min(s, bits); ++k)
                v[k] = static_cast<std::uint32_t>(mi[k - 1]) << (bits - k);
            for (int k = s + 1; k <= bits; ++k)
            {
                v[k] = v[k - s] ^ (v[k - s] >> s);
                for (int l = 1; l <= s - 1; ++l)
                    v[k] ^= ((a >> (s - 1 - l)) & 1u) * v[k - l];
            }
        }

        std::vector<std::uint32_t> shift(static_cast<size_t>(d), 0), x(static_cast<size_t>(d), 0);
        if (scramble_)
            for (auto &s : shift)
                s = static_cast<std::uint32_t>(rng.bits() >> 32);

        Matrix Z(d, m);
        const Float scale = std::ldexp(1.0, -bits);
        for (Index i = 0; i < m; ++i)
        {
            if (i > 0)
            {
                // c = position (1-based) of the rightmost zero bit of i - 1
                std::uint64_t value = static_cast<std::uint64_t>(i - 1);
                int c = 1;
                while (value & 1u)
                {
                    value >>= 1;
                    ++c;
                }
                for (Index j = 0; j < d; ++j)
                    x[static_cast<size_t>(j)] ^= V[static_cast<size_t>(j)][static_cast<size_t>(c)];
            }
            for (Index j = 0; j < d; ++j)
                Z(j, i) = static_cast<Float>(x[static_cast<size_t>(j)] ^ shift[static_cast<size_t>(j)]) * scale;
        }
        return Z;
    }

    namespace
    {
        std::vector<int> first_primes(Index n)
        {
            std::vector<int> primes;
            for (int c = 2; static_cast<Index>(primes.size()) < n; ++c)
            {
                bool prime = true;
                for (int p : primes)
                {
                    if (p * p > c)
                        break;
                    if (c % p == 0)
                    {
                        prime = false;
                        break;
                    }
                }
                if (prime)
                    primes.push_back(c);
            }
            return primes;
        }
    } // namespace

    Matrix HaltonSampler::sample(Index d, Index m, Rng &rng)
    {
        const auto primes = first_primes(d);
        Matrix Z(d, m);
        for (Index j = 0; j < d; ++j)
        {
            const int b = primes[static_cast<size_t>(j)];
            Indices perm(static_cast<size_t>(b));
            std::iota(perm.begin(), perm.end(), Index{0});
            if (scramble_ && b > 2)
            {
                // random permutation of the non-zero digits; pi(0) = 0 keeps the expansion finite
                Indices tail = rng.permutation(b - 1);
                for (int k = 1; k < b; ++k)
                    perm[static_cast<size_t>(k)] = tail[static_cast<size_t>(k - 1)] + 1;
            }
            for (Index i = 0; i < m; ++i)
            {
                long long index = static_cast<long long>(i) + 1; // skip the all-zero point
                Float f = 1.0 / b, value = 0.0;
                while (index > 0)
                {
                    value += static_cast<Float>(perm[static_cast<size_t>(index % b)]) * f;
                    index /= b;
                    f /= b;
                }
                Z(j, i) = value;
            }
        }
        return Z;
    }

    Matrix LhsSampler::sample(Index d, Index m, Rng &rng)
    {
        Matrix Z(d, m);
        for (Index j = 0; j < d; ++j)
        {
            const Indices perm = rng.permutation(m);
            for (Index i = 0; i < m; ++i)
                Z(j, i) = (static_cast<Float>(perm[static_cast<size_t>(i)]) + rng.uniform()) / static_cast<Float>(m);
        }
        return Z;
    }

    std::shared_ptr<Sampler> make_sampler(BaseSampler type)
    {
        switch (type)
        {
        case BaseSampler::UNIFORM:
            return std::make_shared<UniformSampler>();
        case BaseSampler::GAUSSIAN:
            return std::make_shared<GaussianSampler>();
        case BaseSampler::SOBOL:
            return std::make_shared<SobolSampler>();
        case BaseSampler::HALTON:
            return std::make_shared<HaltonSampler>();
        case BaseSampler::LHS:
            return std::make_shared<LhsSampler>();
        }
        throw std::invalid_argument("unknown sampler");
    }
} // namespace modde
