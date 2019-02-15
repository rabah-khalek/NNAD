//
// Author: Valerio Bertone: valerio.bertone@cern.ch
//

#pragma once

#include <vector>
#include <string>
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>

template<typename T> T Sigmoid(T const& x)
{
    return T(1) / (T(1) + exp(-x));
}

template <typename T> T dSigmoid(T const &x)
{
    const T s = Sigmoid(x);
    return s * (T(1) - s);
}

void Error(std::string const& Message)
{
    std::cout << "\033[1;31mERROR: " << Message << "\033[0m" << std::endl;
    exit(-10);
}


void Warning(std::string const& Message)
{
    std::cout << "\033[1;33mWarning: " << Message << "\033[0m" << std::endl;
}
