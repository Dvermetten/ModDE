#pragma once

#include <algorithm>
#include <cmath>
#include <memory>

#include "modde/common.hpp"
#include "modde/population.hpp"
#include "modde/rng.hpp"
#include "modde/sampling.hpp"
#include "modde/settings.hpp"

namespace modde
{
    //! External archive of replaced parents or successful trials (spec/selection_archive.md §3).
    struct Archive
    {
        std::vector<Vector> members;
        [[nodiscard]] Index size() const { return static_cast<Index>(members.size()); }
    };

    //! SHADE memories (spec/adaptation.md §2). NaN in M_CR marks a terminal slot.
    struct Memory
    {
        Vector M_F;
        Vector M_CR;
        Index k = 0;
    };

    //! Success record of the last selection (spec/selection_archive.md §2).
    struct SuccessRecord
    {
        Indices S;          //!< strict improvements
        Vector S_F, S_CR;   //!< F~, CR~ of S
        Vector delta_f;     //!< f_old - f_new > 0 of S
        Matrix replaced;    //!< d x |S| old parents of S
        std::vector<bool> W; //!< trial selected (ties included), per individual
    };

    //! Statistics (spec/settings.md §4).
    struct Stats
    {
        std::size_t generation = 0;
        std::size_t evaluations = 0; //!< counted evaluations (incl. any offset from the problem)
        Vector best_x;
        Float best_f = INF;
        Float popmean = 0, popstd = 0, F_memory_mean = NaN;
        // per evaluation (tracking mode)
        Float curr_F = NaN, curr_CR = NaN, CS = 0, ED = 0;
        bool corrected = false;
        std::size_t corr_so_far = 0;
        std::size_t nan_evaluations = 0; //!< objective values that were NaN (treated as +inf)
    };

    //! Complete runtime state of one optimizer (D-A6).
    struct Parameters
    {
        Settings settings; //!< resolved
        Modules modules;
        Index d;
        Index lambda; //!< current population size
        Rng rng;
        std::shared_ptr<Sampler> sampler;

        Population pop;
        Vector F_used, CR_used; //!< F~, CR~ of the current generation
        Matrix V;               //!< donors, d x lambda
        Matrix U;               //!< trials before bound correction
        Matrix U_corrected;     //!< trials after bound correction
        Vector f_trial;         //!< fitness of the trials (+inf if not evaluated)
        Index n_trials_evaluated = 0; //!< trials 0..n-1 were evaluated; the others never win selection
        std::vector<bool> corrected; //!< per trial: any coordinate corrected
        Vector CS, ED;               //!< per trial correction statistics
        std::vector<Indices> last_parents; //!< population parents used per donor (testing/analysis)
        Matrix opposites;                  //!< points of a pending oppositional jump, d x lambda

        Archive archive;
        Memory memory;
        SuccessRecord success;
        Stats stats;
        bool initialized = false;
        bool stop_requested = false; //!< set from outside (e.g. the problem reached its target); ends the run

        explicit Parameters(const Settings &user_settings);

        //! Progress t = min(1, E / budget).
        [[nodiscard]] Float progress() const
        {
            return std::min<Float>(1.0, static_cast<Float>(stats.evaluations) / static_cast<Float>(*settings.budget));
        }
        [[nodiscard]] std::size_t remaining_budget() const
        {
            return stats.evaluations >= *settings.budget ? 0 : *settings.budget - stats.evaluations;
        }
        [[nodiscard]] Index archive_capacity() const
        {
            return static_cast<Index>(std::llround(*settings.archive_rate * static_cast<Float>(lambda)));
        }
        //! Remove random archive members until it fits its capacity.
        void trim_archive();
    };
} // namespace modde
