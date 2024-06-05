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
  template<typename T> T Linear(T const &x)
  {
    return x;
  }

  template <typename T> T dLinear(T const &x)
  {
    return T(1);
  }

  template<typename T> T Sigmoid(T const &x)
  {
    return T(1) / (T(1) + exp(-x));
  }

  template <typename T> T dSigmoid(T const &x)
  {
    const T s = Sigmoid(x);
    return s * (T(1) - s);
  }

  template <typename T> T Tanh(T const &x)
  {
    const T e = exp( T(2) * x );
    return ( e - T(1) ) / ( e + T(1) );
  }

  template <typename T> T dTanh(T const &x)
  {
    const T t = Tanh(x);
    return T(1) - t * t;
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
