#pragma once

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

    //! Index vector.
    using Indices = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;
} // namespace modde
