//
// Author: Rabah Abdul Khalek - rabah.khalek@gmail.com
//

#pragma once

// Ceres solver
#include <ceres/ceres.h>
#include "FeedForwardNN.imp.h"

// Typedef for the data
typedef std::tuple<double,double,double> Datapoint;
typedef std::vector<Datapoint> vectdata;

class AutoDiffCostFunction
{
  public:
    AutoDiffCostFunction(int const &,
                         vectdata const &,
                         std::vector<int> const &,
                         int const &);

    template <typename T>
    bool operator()(T const *const *, T *) const;
    
  private:
    int _Np;
    vectdata _Data;
    std::vector<int> _NNarchitecture;
    int _Seed;
};

