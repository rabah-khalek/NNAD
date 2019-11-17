//
// Author: Rabah Abdul Khalek - rabah.khalek@gmail.com
//

#pragma once

// Ceres solver
#include <ceres/ceres.h>
#include "Globals.h"
#include <iostream>
#include <tuple>


// NNAD
#include <NNAD/FeedForwardNN.h>

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

    ~AutoDiffCostFunction();

        template <typename T>
        bool operator()(T const *const *, T *) const;

  private:
    int _Np;
    vectdata _Data;
    std::vector<int> _NNarchitecture;
    int _Seed;
    std::tuple<nnad::FeedForwardNN<double>*, nnad::FeedForwardNN<ceres::Jet<double, GLOBALS::kStride>>*> nns;
};

