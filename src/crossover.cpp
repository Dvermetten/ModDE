// Crossover: dev/spec/crossover.md.

#include "modde/operators.hpp"

namespace modde
{
    void apply_crossover(Parameters &p)
    {
        const Index n = p.lambda, d = p.d;
        const bool eigen = p.modules.eigenvalue_crossover && n >= 2;

        Matrix B;
        Matrix Xb, Vb;
        if (eigen)
        {
            const Vector mean = p.pop.X.rowwise().mean();
            const Matrix centred = p.pop.X.colwise() - mean;
            const Matrix C = centred * centred.transpose() / static_cast<Float>(n - 1);
            Eigen::SelfAdjointEigenSolver<Matrix> solver(C);
            B = solver.eigenvectors();
            Xb = B.transpose() * p.pop.X; // KI-16: B^T x, not B x
            Vb = B.transpose() * p.V;
        }
        const Matrix &Xs = eigen ? Xb : p.pop.X;
        const Matrix &Vs = eigen ? Vb : p.V;

        Matrix Us = Xs;
        for (Index i = 0; i < n; ++i)
        {
            const Float CR = p.CR_used(i);
            if (p.modules.crossover == CrossoverType::BIN)
            {
                const Index j_rand = p.rng.randint(d);
                for (Index j = 0; j < d; ++j)
                {
                    const Float rho = p.rng.uniform();
                    if (rho <= CR || j == j_rand)
                        Us(j, i) = Vs(j, i);
                }
            }
            else
            {
                const Index k = p.rng.randint(d);
                Index L = 1;
                while (L < d && p.rng.uniform() < CR)
                    ++L;
                for (Index l = 0; l < L; ++l)
                {
                    const Index j = (k + l) % d;
                    Us(j, i) = Vs(j, i);
                }
            }
        }
        p.U = eigen ? Matrix(B * Us) : Us;
    }
} // namespace modde
