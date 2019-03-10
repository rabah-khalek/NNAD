#include "Utilities.h"
#include "Globals.h"

template<typename T> T Sigmoid(T const& x)
{
    return T(1) / (T(1) + exp(-x));
}

template <typename T> T dSigmoid(T const &x)
{
    const T s = Sigmoid(x);
    return s * (T(1) - s);
}

template double Sigmoid<double>(double const&);
template double dSigmoid<double>(double const &);

template ceres::Jet<double, GLOBALS::kStride> Sigmoid<ceres::Jet<double, GLOBALS::kStride>>(ceres::Jet<double, GLOBALS::kStride> const &);
template ceres::Jet<double, GLOBALS::kStride> dSigmoid<ceres::Jet<double, GLOBALS::kStride>>(ceres::Jet<double, GLOBALS::kStride> const &);