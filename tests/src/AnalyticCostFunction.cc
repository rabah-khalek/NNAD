//
// Author: Valerio Bertone: valerio.bertone@cern.ch
//

#include "AnalyticCostFunction.h"

AnalyticCostFunction::AnalyticCostFunction(int const &Np,
                                           vectdata const &Data,
                                           std::vector<int> &NNarchitecture,
                                           int const &Seed) : _Np(Np),
                                                              _Data(Data),
                                                              _NNarchitecture(NNarchitecture),
                                                              _Seed(Seed)
{
    _nn = new nnad::FeedForwardNN<double>(_NNarchitecture, _Seed);

    // Set number of residuals (i.e. number of data points)
    set_num_residuals(_Data.size());

    // Set sizes of the parameter blocks. There are as many parameter
    // blocks as free parameters and each block has size 1.
    for (int ip = 0; ip < _Np; ip++)
        mutable_parameter_block_sizes()->push_back(1);
}

bool AnalyticCostFunction::Evaluate(double const *const *parameters,
                                            double *residuals,
                                            double **jacobians) const
{
    const int nd = _Data.size();

    std::vector<double> pars;
    for (int i = 0; i < _Np; i++)
        pars.push_back(parameters[i][0]);

    _nn->SetParameters(pars);

    // Residuals and Jacobian
    if (jacobians != NULL)
    {
        for (int id = 0; id < nd; id++)
        {
            std::vector<double> input;
            double x = std::get<0>(_Data[id]);
            input.push_back(x);
            const std::vector<double> vd = _nn->Derive(input);

            residuals[id] = (vd[0] - std::get<1>(_Data[id])) / std::get<2>(_Data[id]);
            for (int ip = 0; ip < _Np; ip++)
                jacobians[ip][id] = (vd[ip + 1]) / std::get<2>(_Data[id]);
        }
    }
    // Only residuals
    else
    {
        for (int id = 0; id < nd; id++)
        {
            std::vector<double> input;
            double x = std::get<0>(_Data[id]);
            input.push_back(x);
            const std::vector<double> v = _nn->Evaluate(input);
            residuals[id] = (v[0] - std::get<1>(_Data[id])) / std::get<2>(_Data[id]);
        }
    }
    return true;
}

AnalyticCostFunction::~AnalyticCostFunction(){delete _nn;};
