//
// Author: Valerio Bertone: valerio.bertone@cern.ch
//

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <math.h>

#include "ceres/jet.h"

template<typename T> T Sigmoid(T const& );

template <typename T> T dSigmoid(T const &);


inline void Error(std::string const& Message)
{
    std::cout << "\033[1;31mERROR: " << Message << "\033[0m" << std::endl;
    exit(-10);
}


inline void Warning(std::string const& Message)
{
    std::cout << "\033[1;33mWarning: " << Message << "\033[0m" << std::endl;
}
