//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//

#pragma once

#include <iostream>
#include <string>
#include <math.h>

namespace nnad
{
  template<typename T> T Sigmoid(T const &x)
  {
    return T(1) / ( T(1) + exp(-x) );
  }

  template <typename T> T dSigmoid(T const &x)
  {
    const T s = Sigmoid(x);
    return s * ( T(1) - s );
  }

  template<typename T> T Tanh(T const &x)
  {
    return tanh(x);
  }

  template <typename T> T dTanh(T const &x)
  {
    return T(1) / pow(cosh(x), 2);
  }

  template<typename T> T Linear(T const &x)
  {
    return x;
  }

  template <typename T> T dLinear(T const &x)
  {
    return T(1);
  }

  template<typename T> T Quadratic(T const &x)
  {
    return x * x;
  }

  template <typename T> T dQuadratic(T const &x)
  {
    return 2 * x;
  }

  inline void Error(std::string const &Message)
  {
    std::cout << "\033[1;31mERROR: " << Message << "\033[0m" << std::endl;
    exit(-10);
  }

  inline void Warning(std::string const &Message)
  {
    std::cout << "\033[1;33mWarning: " << Message << "\033[0m" << std::endl;
  }
}
