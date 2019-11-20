//
// Author: Rabah Abdul Khalek - rabah.khalek@gmail.com
//

#include "AutoDiffCostFunction.h"

AutoDiffCostFunction::AutoDiffCostFunction(int const &Np,
                                         vectdata const &Data,
                                         std::vector<int> const &NNarchitecture,
                                         int const &Seed) : _Np(Np),
                                                            _Data(Data),
                                                            _NNarchitecture(NNarchitecture),
                                                            _Seed(Seed)
{
    std::get<nnad::FeedForwardNN<ceres::Jet<double, GLOBALS::kStride>>*>(nns) = new nnad::FeedForwardNN<ceres::Jet<double, GLOBALS::kStride>>(_NNarchitecture, _Seed);
    std::get<nnad::FeedForwardNN<double>*>(nns) = new nnad::FeedForwardNN<double>(_NNarchitecture, _Seed);
}

AutoDiffCostFunction::~AutoDiffCostFunction()
{
    delete std::get<nnad::FeedForwardNN<ceres::Jet<double, GLOBALS::kStride>> *>(nns);
    delete std::get<nnad::FeedForwardNN<double> *>(nns);
}


template <typename T>
bool AutoDiffCostFunction::operator()(T const *const *parameters, T *residuals) const
{
    static nnad::FeedForwardNN<T> nn(_NNarchitecture, _Seed); //another way to call NN
    
    std::vector<T> pars;
    for (int i = 0; i < _Np; i++)
    {
        pars.push_back(parameters[i][0]);
    }
    //std::get<nnad::FeedForwardNN<T>*>(nns)->SetParameters(pars);
    nn.SetParameters(pars); //another way to call NN

    // Set parameters of the NN

    for (int id = 0; id < _Data.size(); id++)
    {
        std::vector<T> input;
        T x = T(std::get<0>(_Data[id]));
        input.push_back(x);
        //const std::vector<T> v = std::get<nnad::FeedForwardNN<T>*>(nns)->Evaluate(input);
        const std::vector<T> v = nn.Evaluate(input); //another way to call NN

        residuals[id] = (v[0] - std::get<1>(_Data[id])) / std::get<2>(_Data[id]);
    }

    return true;
}

// template fixed types
template bool AutoDiffCostFunction::operator()(double const *const *parameters, double *residuals) const;
template bool AutoDiffCostFunction::operator()(ceres::Jet<double, GLOBALS::kStride> const *const *parameters, ceres::Jet<double, GLOBALS::kStride> *residuals) const;
