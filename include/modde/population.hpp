#pragma once

#include <algorithm>
#include <numeric>

#include "modde/common.hpp"

namespace modde
{
    //! The population: individuals as columns of X (d x n), with their fitness
    //! and their stored control parameters F_i, CR_i (spec/adaptation.md).
    struct Population
    {
        Matrix X;  //!< d x n
        Vector f;  //!< n
        Vector F;  //!< n, stored F_i
        Vector CR; //!< n, stored CR_i

        [[nodiscard]] Index n() const { return f.size(); }
        [[nodiscard]] Index d() const { return X.rows(); }

        //! Indices sorted by ascending f; ties by lower index first.
        [[nodiscard]] Indices argsort() const
        {
            Indices idx(static_cast<size_t>(n()));
            std::iota(idx.begin(), idx.end(), Index{0});
            std::stable_sort(idx.begin(), idx.end(), [&](Index a, Index b) { return f(a) < f(b); });
            return idx;
        }

        //! Index of the best individual (lowest f, lowest index on ties).
        [[nodiscard]] Index best() const
        {
            Index b = 0;
            for (Index i = 1; i < n(); ++i)
                if (f(i) < f(b))
                    b = i;
            return b;
        }

        //! Keep (and reorder to) the given individuals.
        void keep(const Indices &idx)
        {
            const Index m = static_cast<Index>(idx.size());
            Matrix X2(d(), m);
            Vector f2(m), F2(m), CR2(m);
            for (Index j = 0; j < m; ++j)
            {
                const Index i = idx[static_cast<size_t>(j)];
                X2.col(j) = X.col(i);
                f2(j) = f(i);
                F2(j) = F(i);
                CR2(j) = CR(i);
            }
            X = std::move(X2);
            f = std::move(f2);
            F = std::move(F2);
            CR = std::move(CR2);
        }
    };
} // namespace modde
