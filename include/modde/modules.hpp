#pragma once

#include <cstddef>

namespace modde
{
    // Module options. Names and order match src/modde/_registry.py
    // (the first option is the default).

    enum class BaseSampler
    {
        UNIFORM,
        GAUSSIAN,
        SOBOL,
        HALTON,
        LHS
    };

    enum class OppositionalInitialization
    {
        NONE,
        OPPOSITE,
        QUASI_OPPOSITE
    };

    enum class MutationBase
    {
        RAND,
        BEST,
        TARGET
    };

    enum class MutationReference
    {
        NONE,
        RAND,
        BEST,
        PBEST
    };

    enum class PbestSampling
    {
        RANDOM,
        FIXED
    };

    enum class ArchiveContent
    {
        PARENTS,
        TRIALS
    };

    enum class CrossoverType
    {
        BIN,
        EXP
    };

    enum class BoundCorrectionType
    {
        SATURATE,
        NONE,
        UNIF_RESAMPLE,
        COTN,
        TOROIDAL,
        MIRROR,
        HVB,
        EXPC_TARGET,
        EXPC_CENTER,
        EXPS
    };

    enum class AdaptationF
    {
        NONE,
        SHADE,
        SHADE_MODIFIED,
        JDE
    };

    enum class AdaptationCR
    {
        NONE,
        SHADE,
        LSHADE,
        JDE
    };

    enum class PopulationReduction
    {
        NONE,
        LINEAR
    };

    //! All module switches (D-A6). Defaults = classic DE/rand/1/bin (D-D2).
    struct Modules
    {
        BaseSampler base_sampler = BaseSampler::UNIFORM;
        OppositionalInitialization oppositional_initialization = OppositionalInitialization::NONE;
        MutationBase mutation_base = MutationBase::RAND;
        MutationReference mutation_reference = MutationReference::NONE;
        std::size_t mutation_n_comps = 1;
        PbestSampling pbest_sampling = PbestSampling::RANDOM;
        bool mutation_use_weighted_F = false;
        bool use_archive = false;
        ArchiveContent archive_content = ArchiveContent::PARENTS;
        CrossoverType crossover = CrossoverType::BIN;
        bool eigenvalue_crossover = false;
        BoundCorrectionType bound_correction = BoundCorrectionType::SATURATE;
        AdaptationF adaptation_method_F = AdaptationF::NONE;
        AdaptationCR adaptation_method_CR = AdaptationCR::NONE;
        bool use_jso_caps = false;
        PopulationReduction population_reduction = PopulationReduction::NONE;
        bool oppositional_generation = false;

        bool shade_F() const
        {
            return adaptation_method_F == AdaptationF::SHADE || adaptation_method_F == AdaptationF::SHADE_MODIFIED;
        }
        bool shade_CR() const
        {
            return adaptation_method_CR == AdaptationCR::SHADE || adaptation_method_CR == AdaptationCR::LSHADE;
        }
    };
} // namespace modde
