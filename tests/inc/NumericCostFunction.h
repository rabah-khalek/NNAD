//
// Author: Rabah Abdul Khalek - rabah.khalek@gmail.com
//

#pragma once

// Ceres solver
#include <ceres/ceres.h>

// NNAD
#include "NNAD/FeedForwardNN.h"

// Typedef for the data
typedef std::tuple<double, double, double> Datapoint;
typedef std::vector<Datapoint> vectdata;

struct NumericCostFunction
{
  ~NumericCostFunction();

  NumericCostFunction(int const &,
                      vectdata const &,
                      std::vector<int> const &,
                      int const &);

  bool operator()(double const *const *, double *) const;

  nnad::FeedForwardNN<double> *_nn;
  int _Np;
  vectdata _Data;
  std::vector<int> _NNarchitecture;
  int _Seed;
};
