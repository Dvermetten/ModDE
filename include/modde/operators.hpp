#pragma once

#include <memory>

#include "modde/parameters.hpp"

namespace modde
{
    //! Mutation (spec/mutation.md). Writes p.V. Virtual so it can become a
    //! Python-overridable hook (D-A8).
    class Mutation
    {
    public:
        virtual ~Mutation() = default;
        virtual void apply(Parameters &p);
    };

    //! Crossover (spec/crossover.md). Writes p.U.
    void apply_crossover(Parameters &p);

    //! Bound correction of a single out-of-bounds coordinate in normalised
    //! space (spec/bound_correction.md). `b` = violated bound (0 or 1),
    //! `tau` = normalised coordinate of the parent.
    class BoundCorrection
    {
    public:
        virtual ~BoundCorrection() = default;
        [[nodiscard]] virtual bool enabled() const { return true; }
        virtual Float repair(Float y, int b, Float tau, Rng &rng) const = 0;
    };

    std::shared_ptr<BoundCorrection> make_bound_correction(BoundCorrectionType type, Float hvb_alpha);

    //! Applies `bc` to p.U, writing p.U_corrected, p.corrected, p.CS, p.ED.
    void apply_bound_correction(Parameters &p, const BoundCorrection &bc);

    //! Step 0: F~, CR~ for this generation (spec/adaptation.md).
    void sample_parameters(Parameters &p);

    //! jDE keep-on-success commit; called at selection, before any reordering.
    void commit_jde(Parameters &p);

    //! SHADE memory update from the success record (spec/adaptation.md §2).
    void update_memory(Parameters &p);
} // namespace modde
