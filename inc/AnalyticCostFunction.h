//
// Author: Valerio Bertone: valerio.bertone@cern.ch
//

#pragma once

// Ceres solver
#include <ceres/ceres.h>
#include "FeedForwardNN.h"

// Typedef for the data
typedef std::tuple<double, double, double> Datapoint;
typedef std::vector<Datapoint> vectdata;

// A CostFunction implementing analytically derivatives for the chi2.
class AnalyticCostFunction : public ceres::CostFunction
{
  public:
    AnalyticCostFunction(int const &Np,
                         vectdata const &Data,
                         std::vector<int> &NNarchitecture,
                         int const &Seed) : _Np(Np),
                                            _Data(Data),
                                            _NNarchitecture(NNarchitecture),
                                            _Seed(Seed)
    {
        
		// Set number of residuals (i.e. number of data points)
		set_num_residuals(_Data.size());

		// Set sizes of the parameter blocks. There are as many parameter
		// blocks as free parameters and each block has size 1.
		for (int ip = 0; ip < _Np; ip++)
		mutable_parameter_block_sizes()->push_back(1);
        
    }
    virtual ~AnalyticCostFunction() {}
    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const
    {
        const int nd = _Data.size();
        //TODO: pass the info from main
        FeedForwardNN<double> *nn = new FeedForwardNN<double>(_NNarchitecture, _Seed);
        std::vector<double> pars;
        for (int i = 0; i < _Np; i++)
            pars.push_back(parameters[i][0]);

        nn->SetParameters(pars);

        // Residuals and Jacobian
        if (jacobians != NULL)
        {
            for (int id = 0; id < nd; id++)
            {
                std::vector<double> input;
                double x = std::get<0>(_Data[id]);
                input.push_back(x);
                const std::vector<double> vd = nn->Derive(input);

                residuals[id] = (vd[0] - std::get<1>(_Data[id])) / std::get<2>(_Data[id]);
                for (int ip = 0; ip < _Np; ip++)
                    jacobians[ip][id] = (vd[ip + 1]) / std::get<2>(_Data[id]);
            }
        }
        // Only residuals
        else
        {
            // Set parameters of the NN

            for (int id = 0; id < nd; id++)
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
        return true;
    }

  private:
    int _Np;
    vectdata _Data;
    std::vector<int> _NNarchitecture;
    int _Seed;
};
