//
// Author: Rabah Abdul Khalek - rabah.khalek@gmail.com
//

#include "NumericCostFunction.h"

NumericCostFunction::NumericCostFunction(int const &Np,
                                       vectdata const &Data,
                                       std::vector<int> const &NNarchitecture,
                                       int const &Seed) : _Np(Np),
                                                          _Data(Data),
                                                          _NNarchitecture(NNarchitecture),
                                                          _Seed(Seed)
{}

bool NumericCostFunction::operator()(double const *const *parameters, double *residuals) const
{
    //TODO: pass the info from main
    FeedForwardNN<double> *nn = new FeedForwardNN<double>(_NNarchitecture, _Seed);
    std::vector<double> pars;
    for (int i = 0; i < _Np; i++)
    {
        pars.push_back(parameters[i][0]);
    }
    nn->SetParameters(pars);
    // Set parameters of the NN

    for (int id = 0; id < _Data.size(); id++)
    {
        std::vector<double> input;
        double x = std::get<0>(_Data[id]);
        input.push_back(x);
        const std::vector<double> v = nn->Evaluate(input);
        residuals[id] = (v[0] - std::get<1>(_Data[id])) / std::get<2>(_Data[id]);
    }

    delete nn;
    return true;
}