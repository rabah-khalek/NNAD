//
// Author: Valerio Bertone: valerio.bertone@cern.ch
//

#pragma once

#include "Utilities.h"
#include "Matrix.h"

#include <vector>
#include <functional>
#include <map>
#include <iostream>
#include <math.h>

// Typedef for the coordinates of the NN parameters
typedef std::tuple<bool, int, int, int> coordinates;

/**
 * @brief the FeedForwardNN class evaluates a feed-forward NN and its
 * derivatives for each of the free parameters. The function "Derive"
 * returns a vector that contains as many vectors as free parameters
 * of the NN (links and biases) plus one times the number of output
 * nodes (say "Nout"). The first "Nout" elements of the vector
 * returned by "Derive" coincide with the output of "Evaluate",
 * i.e. to the NN itself evaluated at some "x". The following elements
 * correspond to the derivative w.r.t. the free parameters computed at
 * "x". The ordering of the derivatives is a little weird, this is a
 * consequence of the way the backward propagation algorithm works.
 * They start from the output layer and proceed backward to the first
 * hidden layer (the input layer is assumed to have no free
 * parameters). For each layer, the index "i" runs over the noded of
 * the layer itself while the index "j" runs over the nodes of the
 * preceding layer. For each value of "i", first comes the derivative
 * w.r.t. the bias "theta_i" then all the derivatives w.r.t. the links
 * "omega_{ij}", for all accessible values of "j".
 */
template<class T>
class FeedForwardNN
{
 public:

  // Constructors
   FeedForwardNN(std::vector<int> const &Arch,
                 int const &RandomSeed = 0,
                 std::function<T(T const &)> const &ActFun = Sigmoid<T>,
                 std::function<T(T const &)> const &dActFun = dSigmoid<T>,
                 bool const &LinearOutput = true);

   FeedForwardNN(std::vector<int> const &Arch,
                 std::vector<T> const &Pars,
                 std::function<T(T const &)> const &ActFun = Sigmoid,
                 std::function<T(T const &)> const &dActFun = dSigmoid,
                 bool const &LinearOutput = true);

   // Setters
   void SetParameters(std::vector<T> const &);

   void SetParameter(int const &, T const &);

   void SetParameter(std::string const &, T const &);

   void SetLink(int const &, int const &, int const &j, T const &);
   void SetBias(int const &, int const &, T const &);

   // Getters
   std::vector<int> GetArchitecture() const { return _Arch; }
   bool IsOutputLinear() const { return _LinearOutput; }
   std::function<T(T const &)> GetActivationFunction() const { return _ActFun; }
   std::function<T(T const &)> GetDerActivationFunction() const { return _dActFun; }
   int GetParameterNumber() const { return _Np; }
   T GetLink(int const &l, int const &i, int const &j) const { return _Links.at(l).GetElement(i, j); }
   T GetBias(int const &l, int const &i) const { return _Biases.at(l).GetElement(i, 0); }
   std::map<int, coordinates> GetIntMap() const { return _IntMap; }
   std::map<std::string, coordinates> GetStrMap() const { return _StrMap; }
   std::map<std::string, int> GetStrIntMap() const { return _StrIntMap; }
   
   std::vector<T> GetParameters() const;

   // Evaluate NN in Input
   std::vector<T> Evaluate(std::vector<T> const &) const;


   // Evaluate NN and its derivatives in Input
   std::vector<T> Derive(std::vector<T> const &) const;

 private:
  const std::vector<int>                     _Arch;
  const std::function<T(T const&)> _ActFun;
  const std::function<T(T const&)> _dActFun;
  const bool                                 _LinearOutput;
  int                                        _Np;
  std::map<int, Matrix<T> >                      _Links;
  std::map<int, Matrix<T> >                      _Biases;
  std::map<int, coordinates>                 _IntMap;
  std::map<std::string, coordinates>         _StrMap;
  std::map<std::string, int>                 _StrIntMap;
};
