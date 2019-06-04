//
// Author: Rabah Abdul Khalek - rabah.khalek@gmail.com
//

#include "AutoDiffCostFunction.h"
#include "Globals.h"

AutoDiffCostFunction::AutoDiffCostFunction(int const &Np,
                                         vectdata const &Data,
                                         std::vector<int> const &NNarchitecture,
                                         int const &Seed) : _Np(Np),
                                                            _Data(Data),
                                                            _NNarchitecture(NNarchitecture),
                                                            _Seed(Seed){}

template <typename T>
bool AutoDiffCostFunction::operator()(T const *const *parameters, T *residuals) const
{
    FeedForwardNN<T> *nn = new FeedForwardNN<T>(_NNarchitecture, _Seed);
    std::vector<T> pars;
    for (int i = 0; i < _Np; i++)
    {
        pars.push_back(parameters[i][0]);
    }
    nn->SetParameters(pars);
    // Set parameters of the NN

    for (int id = 0; id < _Data.size(); id++)
    {
        std::vector<T> input;
        T x = T(std::get<0>(_Data[id]));
        input.push_back(x);
        const std::vector<T> v = nn->Evaluate(input);
        residuals[id] = (v[0] - std::get<1>(_Data[id])) / std::get<2>(_Data[id]);
    }

    delete nn;
    return true;
    }

// template fixed types
template bool AutoDiffCostFunction::operator()(double const *const *parameters, double *residuals) const;
template bool AutoDiffCostFunction::operator()(ceres::Jet<double, GLOBALS::kStride> const *const *parameters, ceres::Jet<double, GLOBALS::kStride> *residuals) const;
