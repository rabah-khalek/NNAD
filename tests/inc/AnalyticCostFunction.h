//
// Author: Valerio Bertone: valerio.bertone@cern.ch
//

#pragma once

// Ceres solver
#include <ceres/ceres.h>

// NNAD
#include "NNAD/FeedForwardNN.h"

// Typedef for the data
typedef std::tuple<double, double, double> Datapoint;
typedef std::vector<Datapoint> vectdata;

// A CostFunction implementing analytically derivatives for the chi2.
class AnalyticCostFunction : public ceres::CostFunction
{
  public:
    AnalyticCostFunction(int const &,
                         vectdata const &,
                         std::vector<int> &,
                         int const &);

    virtual ~AnalyticCostFunction();

    virtual bool Evaluate(double const *const *,
                          double *,
                          double **) const;

  private:
    nnad::FeedForwardNN<double> *_nn;
    int _Np;
    vectdata _Data;
    std::vector<int> _NNarchitecture;
    int _Seed;
};
