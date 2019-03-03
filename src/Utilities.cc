#include "Utilities.h"

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

template ceres::Jet<double, 4> Sigmoid<ceres::Jet<double, 4>>(ceres::Jet<double, 4> const &);
template ceres::Jet<double, 4> dSigmoid<ceres::Jet<double, 4>>(ceres::Jet<double, 4> const &);