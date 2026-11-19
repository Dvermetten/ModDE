// Python bindings for the modde C++ core (module: modde._core).
//
// Phase 0: toolchain check only. Exposes build information and one small
// Eigen-based function so the nanobind <-> Eigen <-> NumPy path is exercised in CI.

#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include "modde/common.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace
{
    std::string compiler_id()
    {
#if defined(_MSC_VER)
        return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
        return "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__);
#elif defined(__GNUC__)
        return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#else
        return "unknown";
#endif
    }

    // Accepts an (n x d) NumPy array (rows = individuals, as in the Python API),
    // converts it to the internal d x n layout and returns the Euclidean norm of
    // each individual.
    modde::Vector individual_norms(const Eigen::Ref<const Eigen::Matrix<modde::Float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &X)
    {
        const modde::Matrix population = X.transpose(); // d x n, column-major
        return population.colwise().norm().transpose();
    }
} // namespace

NB_MODULE(_core, m)
{
    m.doc() = "modde C++ core (development build).";

    m.def(
        "build_info",
        []()
        {
            nb::dict info;
            info["compiler"] = compiler_id();
            info["cplusplus"] = static_cast<long>(__cplusplus);
#if defined(_MSVC_LANG)
            info["msvc_lang"] = static_cast<long>(_MSVC_LANG);
#endif
            info["eigen"] = std::to_string(EIGEN_WORLD_VERSION) + "." +
                            std::to_string(EIGEN_MAJOR_VERSION) + "." +
                            std::to_string(EIGEN_MINOR_VERSION);
#if defined(NB_VERSION_MAJOR)
            info["nanobind"] = std::to_string(NB_VERSION_MAJOR) + "." +
                               std::to_string(NB_VERSION_MINOR) + "." +
                               std::to_string(NB_VERSION_PATCH);
#endif
            return info;
        },
        "Information about how the extension was compiled.");

    m.def("individual_norms", &individual_norms, "X"_a,
          "Euclidean norm of each row of an (n x d) array. Toolchain check for "
          "the NumPy <-> Eigen conversion; not part of the public API.");
}
