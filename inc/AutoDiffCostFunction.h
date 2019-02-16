//
// Author: Rabah Abdul Khalek - rabah.khalek@gmail.com
//

#pragma once

// Ceres solver
#include <ceres/ceres.h>
#include "FeedForwardNN.h"

// Typedef for the data
typedef std::tuple<double,double,double> Datapoint;
typedef std::vector<Datapoint> vectdata;

class my_AutoDiffCostFunctor
{
  public:
    my_AutoDiffCostFunctor(int const &Np,
                           vectdata const &Data) : _Np(Np),
                                                   _Data(Data)
    {
        /*
		// Set number of residuals (i.e. number of data points)
		set_num_residuals(_Data.size());

		// Set sizes of the parameter blocks. There are as many parameter
		// blocks as free parameters and each block has size 1.
		for (int ip = 0; ip < _Np; ip++)
		mutable_parameter_block_sizes()->push_back(1);
    */
    }

    template <typename T>
    bool operator()(T const *const *parameters, T *residuals) const
    {
        //TODO: pass the info from main
        FeedForwardNN<T> *nn = new FeedForwardNN<T>({1, 5, 1}, 5);
        std::vector<T> pars;
        for(int i=0; i<_Np;i++)
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

  private:
    int _Np;
    vectdata _Data;
};

