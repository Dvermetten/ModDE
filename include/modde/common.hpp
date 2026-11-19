#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Dense>

namespace modde
{
    //! Floating point type used throughout the library (D-A15).
    using Float = double;

    //! Column vector.
    using Vector = Eigen::Matrix<Float, Eigen::Dynamic, 1>;

    //! Column-major matrix. Populations are stored as d x n, so each individual
    //! (a column) is contiguous in memory (D-A9). The Python API exposes n x d.
    using Matrix = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>;

    //! Row-major matrix, used only at the Python boundary (n x d arrays).
    using RowMatrix = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    using Index = Eigen::Index;
    using Indices = std::vector<Index>;

    constexpr Float INF = std::numeric_limits<Float>::infinity();
    constexpr Float NaN = std::numeric_limits<Float>::quiet_NaN();
} // namespace modde
