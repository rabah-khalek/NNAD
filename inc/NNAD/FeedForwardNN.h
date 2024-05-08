//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//

#pragma once

#include "Matrix.h"
#include "Distributions.h"

#include <vector>
#include <functional>
#include <map>
#include <math.h>
#include <iostream>
#include <tuple>

namespace nnad
{
  // Typedef for the coordinates of the NN parameters
  typedef std::tuple<bool, int, int, int> coordinates;

  // Typedef for the learning tensor
  typedef std::vector<std::map<double, double>> tensor;

  // Available output functions
  enum OutputFunction: int {ACTIVATION, LINEAR, QUADRATIC};

  /**
   * @brief the FeedForwardNN class evaluates a feed-forward NN and
   * its derivatives for each of the free parameters. The function
   * "Derive" returns a vector that contains as many vectors as free
   * parameters of the NN (links and biases) plus one times the number
   * of output nodes (say "Nout"). The first "Nout" elements of the
   * vector returned by "Derive" coincide with the output of
   * "Evaluate", i.e. to the NN itself evaluated at some "x". The
   * following elements correspond to the derivative w.r.t. the free
   * parameters computed at "x". The ordering of the derivatives is a
   * little weird, this is a consequence of the way the backward
   * propagation algorithm works. They start from the output layer and
   * proceed backward to the first hidden layer (the input layer is
   * assumed to have no free parameters). For each layer, the index
   * "i" runs over the nodes of the layer itself while the index "j"
   * runs over the nodes of the preceding layer. For each value of
   * "i", first comes the derivative w.r.t. the bias "theta_i" then
   * all derivatives w.r.t. the links "omega_{ij}", for all accessible
   * values of "j".
   */
  template<class T>
  class FeedForwardNN
  {
  public:
    //_________________________________________________________________________________
    FeedForwardNN(std::vector<int>                const& Arch,
                  int                             const& RandomSeed,
                  bool                            const& Report = false,
                  std::function<T(T const&)>      const& ActFun = Sigmoid<T>,
                  std::function<T(T const&)>      const& dActFun = dSigmoid<T>,
                  OutputFunction                  const& OutputFunc = LINEAR,
                  InitDistribution                const& InitDist = UNIFORM,
                  std::vector<std::vector<T>>     const& DistParams = {},
                  bool                            const& NTKScaling = false):
      _Arch(Arch),
      _ActFun(ActFun),
      _dActFun(dActFun),
      _OutputFunc(OutputFunc),
      _InitDist(InitDist),
      _DistParams(DistParams),
      _NTKScaling(NTKScaling)
    {
      // Number of layers
      const int nl = (int) _Arch.size();

      // Store sub seeds
      srand(RandomSeed);
      std::vector<int> sub_seeds;
      for (int l = 1; l < nl; l++)
        {
          sub_seeds.push_back(rand());
          sub_seeds.push_back(rand());
        }

      // Number of parameters
      _Np = 0;

      // Initialise links and biases with random numbers. Links are
      // treated as matrices that connect consecutive layers with as
      // many rows as nodes of the rightmost layer and as many columns
      // as the leftmost layer. Biases are treated as one-column
      // matrices associated to each layer (but the input one) with as
      // many rows as nodes.
      int iseed = 0;
      for (int l = 1; l < nl; l++)
        {
          _Links.insert({l, Matrix<T>{_Arch[l], _Arch[l - 1], sub_seeds.at(iseed++), InitDist, {}}});
          _Biases.insert({l, Matrix<T>{_Arch[l], 1, sub_seeds.at(iseed++), InitDist, {}}});
          _Np += _Arch[l] * (_Arch[l - 1] + 1);
        }

      // Construct maps that index the free parameters. Each parameter is
      // associated to an integer and a string.
      int count = 0;
      for (int l = nl - 1; l > 0; l--)
        for (int i = 0; i < _Arch[l]; i++)
          {
            // Biases
            const coordinates coord = std::make_tuple(false, l, i, 0);
            const std::string name = "B_" + std::to_string(l) + "_" + std::to_string(i);
            _StrIntMap.insert({name, count});
            _IntMap.insert({count++, coord});
            _StrMap.insert({name, coord});

            // Links
            for (int j = 0; j < _Arch[l - 1]; j++)
              {
                const coordinates coord = std::make_tuple(true, l, i, j);
                const std::string name = "L_" + std::to_string(l) + "_" + std::to_string(i) + "_" + std::to_string(j);
                _StrIntMap.insert({name, count});
                _IntMap.insert({count++, coord});
                _StrMap.insert({name, coord});
              }
          }

      // Define activation function of the output nodes and its
      // derivative.
      switch (_OutputFunc)
        {
        case ACTIVATION:
          _OutputActFun  = _ActFun;
          _dOutputActFun = _dActFun;
          break;
        case LINEAR:
          _OutputActFun  = [] (T const& x) -> T { return x; };
          _dOutputActFun = [] (T const&)   -> T { return T{1}; };
          break;
        case QUADRATIC:
          _OutputActFun  = [] (T const& x) -> T { return x * x; };
          _dOutputActFun = [] (T const& x) -> T { return 2 * x; };
          break;
        }

      // Report NN parameters
      if (Report)
        {
          std::cout << "\nFeed-forward neural network initialised:" << std::endl;
          std::cout << "- architecture = [ ";
          for (auto const& n : _Arch)
            std::cout << n << " ";
          std::cout << "]" << std::endl;
          std::cout << "- number of parameters = " << _Np << std::endl;
          // Select distribution at initialisation
          switch(_InitDist)
            {
            case UNIFORM:
              std::cout << "- uniform distribution at initialisation" << std::endl;
              break;
            case GAUSSIAN:
              std::cout << "- Gaussian distribution at initialisation" << std::endl;
            }
          // Select integration method
          switch (_OutputFunc)
            {
            case ACTIVATION:
              std::cout << "- activation-like output function" << std::endl;
              break;
            case LINEAR:
              std::cout << "- linear output function" << std::endl;
              break;
            case QUADRATIC:
              std::cout << "- quadratic output function" << std::endl;
              break;
            default:
              Error("FeedForwardNN: Unknown output function.");
            }
          std::cout << "\n";
        }
    }

    //_________________________________________________________________________________
    FeedForwardNN(std::vector<int>           const& Arch,
                  int                        const& RandomSeed,
                  OutputFunction             const& OutputFunc = LINEAR,
                  bool                       const& Report = false,
                  std::function<T(T const&)> const& ActFun = Sigmoid<T>,
                  std::function<T(T const&)> const& dActFun = dSigmoid<T>):
      FeedForwardNN(Arch, RandomSeed, Report, ActFun, dActFun, OutputFunc)
    {
    }

    //_________________________________________________________________________________
    FeedForwardNN(std::vector<int>           const& Arch,
                  std::vector<T>             const& Pars,
                  std::function<T(T const&)> const& ActFun = Sigmoid<T>,
                  std::function<T(T const&)> const& dActFun = dSigmoid<T>,
                  OutputFunction             const& OutputFunc = LINEAR,
                  bool                       const& Report = false):
      FeedForwardNN(Arch, 0, Report, ActFun, dActFun, OutputFunc)
    {
      SetParameters(Pars);
    }

    //_________________________________________________________________________________
    // Copy constructor
    FeedForwardNN(FeedForwardNN<T> *NN):
      FeedForwardNN(NN->_Arch, 0, false,
                    NN->GetActivationFunction(),
                    NN->GetDerActivationFunction(),
                    NN->OutputFunctionType(),
                    NN->GetInitDistribution(),
                    {},
                    NN->GetInitNTKScaling())
    {
      SetParameters(NN->GetParameters());
    }

    //_________________________________________________________________________________
    void SetParameters(std::vector<T> const& Pars)
    {
      // Number of parameters
      const int np = (int) Pars.size();

      // Check that the size of the parameter vector is equal to the
      // number of parameters of the current NN.
      if (np != _Np)
        Error("SetParameters: the number of parameters does not match that of the NN.");

      for (int ip = 0; ip < np; ip++)
        {
          const coordinates coord = _IntMap.at(ip);
          if (std::get<0>(coord))
            _Links.at(std::get<1>(coord)).SetElement(std::get<2>(coord), std::get<3>(coord), Pars[ip]);
          else
            _Biases.at(std::get<1>(coord)).SetElement(std::get<2>(coord), std::get<3>(coord), Pars[ip]);
        }
    }

    //_________________________________________________________________________________
    void SetParameter(int const& ip, T const& Par)
    {
      // Check that the index "ip" is within the allowed bounds
      if (ip < 0 || ip >= _Np)
        Error("SetParameter: index 'ip' out of range.");

      // Get coordinates
      const coordinates coord = _IntMap.at(ip);

      if (std::get<0>(coord))
        _Links.at(std::get<1>(coord)).SetElement(std::get<2>(coord), std::get<3>(coord), Par);
      else
        _Biases.at(std::get<1>(coord)).SetElement(std::get<2>(coord), std::get<3>(coord), Par);
    }

    //_________________________________________________________________________________
    void SetParameter(std::string const& sp, T const& Par)
    {
      // Get coordinates
      const coordinates coord = _StrMap.at(sp);

      if (std::get<0>(coord))
        _Links.at(std::get<1>(coord)).SetElement(std::get<2>(coord), std::get<3>(coord), Par);
      else
        _Biases.at(std::get<1>(coord)).SetElement(std::get<2>(coord), std::get<3>(coord), Par);
    }

    //_________________________________________________________________________________
    void SetLink(int const& l, int const& i, int const& j, T const& Par)
    {
      _Links.at(l).SetElement(i, j, Par);
    }

    //_________________________________________________________________________________
    void SetBias(int const& l, int const& i, T const& Par)
    {
      _Biases.at(l).SetElement(i, 0, Par);
    }

    //_________________________________________________________________________________
    std::vector<int> GetArchitecture() const
    {
      return _Arch;
    }

    //_________________________________________________________________________________
    OutputFunction OutputFunctionType() const
    {
      return _OutputFunc;
    }

    //_________________________________________________________________________________
    InitDistribution GetInitDistribution() const
    {
      return _InitDist;
    }

    //_________________________________________________________________________________
    bool GetInitNTKScaling() const
    {
      return _NTKScaling;
    }

    //_________________________________________________________________________________
    std::function<T(T const&)> GetActivationFunction() const
    {
      return _ActFun;
    }

    //_________________________________________________________________________________
    std::function<T(T const&)> GetDerActivationFunction() const
    {
      return _dActFun;
    }

    //_________________________________________________________________________________
    int GetParameterNumber() const
    {
      return _Np;
    }

    //_________________________________________________________________________________
    T GetLink(int const& l, int const& i, int const& j) const
    {
      return _Links.at(l).GetElement(i, j);
    }

    //_________________________________________________________________________________
    T GetBias(int const& l, int const& i) const
    {
      return _Biases.at(l).GetElement(i, 0);
    }

    //_________________________________________________________________________________
    std::map<int, coordinates> GetIntMap() const
    {
      return _IntMap;
    }

    //_________________________________________________________________________________
    std::map<std::string, coordinates> GetStrMap() const
    {
      return _StrMap;
    }

    //_________________________________________________________________________________
    std::map<std::string, int> GetStrIntMap() const
    {
      return _StrIntMap;
    }

    //_________________________________________________________________________________
    std::vector<T> GetParameters() const
    {
      // Initialise output vector.
      std::vector<T> output(_Np);

      // Element counter
      int count = 0;

      // Number of layers
      const int nl = (int) _Arch.size();

      // Now run backwards on the layers to compute the derivatives. Same
      // ordering as in the 'Derive' function.
      for (int l = nl - 1; l > 0; l--)
        for (int i = 0; i < _Arch[l]; i++)
          {
            // Biases
            output[count++] = _Biases.at(l).GetElement(i, 0);

            // Links
            for (int j = 0; j < _Arch[l - 1]; j++)
              output[count++] = _Links.at(l).GetElement(i, j);
          }

      return output;
    }

    //_________________________________________________________________________________
    std::vector<T> Evaluate(std::vector<T> const& Input) const
    {
      // Check that the size of the input vector is equal to the number of
      // input nodes.
      if ((int) Input.size() != _Arch[0])
        Error("Evaluate: the number of inputs does not match the number of input nodes.");

      // Number of layers
      const int nl = (int) _Arch.size();

      // Construct output of the NN recursively for the hidden layers
      Matrix<T> y{(int) Input.size(), 1, Input};
      for (int l = 1; l < nl - 1; l++)
        {
          // NTK_FLAG
          if (_NTKScaling)
            y = y * T(T(1) / T(sqrt(_Arch[l-1])));

          y = Matrix<T> {_Links.at(l) * y + _Biases.at(l), _ActFun};
        }

      // Now take care of the output layer according to the selected
      // output function.
      // NTK_FLAG
      if (_NTKScaling)
        y = y * T(T(1) / sqrt(_Arch[nl-2]));
      return (Matrix<T> {_Links.at(nl - 1) * y + _Biases.at(nl - 1), _OutputActFun}).GetVector();
    }

    //_________________________________________________________________________________
    std::vector<T> Derive(std::vector<T> const& Input) const
    {
      // Check that the size of the input vector is equal to the number of
      // input nodes.
      if ((int) Input.size() != _Arch[0])
        Error("Derive: the number of inputs does not match the number of input nodes.");

      // Initialise output vector
      std::vector<T> output((_Np + 1) * _Arch.back());

      // Element counter
      int count = 0;

      // Number of layers
      const int nl = (int) _Arch.size();

      // Compute NN recursively and save the quantities that will be
      // needed to compute the derivatives.
      std::map<int, Matrix<T>> y;
      std::map<int, Matrix<T>> z;
      y.insert({0, Matrix<T>{(int) Input.size(), 1, Input}});
      for (int l = 1; l < nl - 1; l++)
        {
          // NTK_FLAG
          if (_NTKScaling)
              y.at(l - 1) = y.at(l - 1) * T(T(1) / T(sqrt(_Arch[l - 1])));

          const Matrix<T> M = _Links.at(l) * y.at(l - 1) + _Biases.at(l);
          y.insert({l, Matrix<T>{M, _ActFun}});
          z.insert({l, Matrix<T>{M, _dActFun}});
        }

      // Now take care of the output layer according to the selected
      // output function.

      // NTK_FLAG
      if (_NTKScaling)
          y.at(nl - 2) = y.at(nl - 2) * T(T(1) / T(sqrt(_Arch[nl - 2])));

      const Matrix<T> M = _Links.at(nl - 1) * y.at(nl - 2) + _Biases.at(nl - 1);
      y.insert({nl - 1, Matrix<T>{M, _OutputActFun}});
      z.insert({nl - 1, Matrix<T>{M, _dOutputActFun}});

      // Output vector of vectors. Push back NN itself as a first
      // element.
      for (auto const& e : y.at(nl - 1).GetVector())
        output[count++] = e;

      // First compute the Sigma matrix on the output layer that is just
      // the unity matrix.
      Matrix<T> Sigma(_Arch[nl - 1], _Arch[nl - 1], std::vector<T>(_Arch[nl - 1] * _Arch[nl - 1], T(0.)));
      for (int k = 0; k < _Arch[nl - 1]; k++)
        Sigma.SetElement(k, k, T(1));

      // Now run backwards on the layers to compute the derivatives
      for (int l = nl - 1; l > 0; l--)
        {
          const std::vector<T> zl = z.at(l).GetVector();
          const std::vector<T> ylmo = y.at(l - 1).GetVector();
          for (int i = 0; i < _Arch[l]; i++)
            {
              // Compute derivatives w.r.t. the biases
              for (int k = 0; k < _Arch[nl - 1]; k++)
                output[count++] = Sigma.GetElement(k, i) * zl[i];

              // Compute derivatives w.r.t. the links
              for (int j = 0; j < _Arch[l - 1]; j++)
                for (int k = 0; k < _Arch[nl - 1]; k++)
                  output[count++] = Sigma.GetElement(k, i) * zl[i] * ylmo[j];
            }

          // Compute Matrix "S" on this layer
          std::vector<T> entries;
          for (int i = 0; i < _Arch[l]; i++)
            {
              for (int j = 0; j < _Arch[l - 1]; j++)
                {
                  // NTK_FLAG
                  if (_NTKScaling)
                    entries.push_back(zl[i] * _Links.at(l).GetElement(i, j) / sqrt(_Arch[l - 1]));
                  else
                    entries.push_back(zl[i] * _Links.at(l).GetElement(i, j));
                }
            }
          const Matrix<T> S{_Arch[l], _Arch[l - 1], entries};

          // Update Matrix "Sigma" for the next step
          Sigma = Sigma * S;
        }
      return output;
    }

    //_________________________________________________________________________________
    Matrix<T> NTK (std::vector<T> const& Input_a, std::vector<T> const& Input_b) const
    {
      // Number of layers
      const int nl = (int) _Arch.size();

      // Compute the NTK in the first layer

      // Initialise the NTK with zeros
      Matrix<T> H_ab {_Arch[1], _Arch[1], std::vector<T>(_Arch[1] * _Arch[1], T(0.))}; // H^{1}

      // Compute the diagonal element
      Matrix<T> ya{(int) Input_b.size(), 1, Input_a};
      Matrix<T> yb{(int) Input_b.size(), 1, Input_b};
      Matrix<T> za{(int) Input_b.size(), 1, std::vector<T>((int) Input_a.size(), T(0.))};
      Matrix<T> zb{(int) Input_b.size(), 1, std::vector<T>((int) Input_b.size(), T(0.))};

      T D = std::inner_product(std::begin(Input_a), std::end(Input_a), std::begin(Input_b), T(0.0));

      // NTK_FLAG
      D = T(1.) + D / ( _NTKScaling ? T(_Arch[0]) : T(1.));

      // Fill the diagonal entries of the NTK
      for (int i = 0; i < H_ab.GetLines(); i++)
          H_ab.SetElement(i,i, D);

      // Apply the forward equation to compute the
      // NTK in the last layer
      for (int l = 1; l < nl - 1; l++)
        {
          // NTK_FLAG
          if (_NTKScaling) {
            ya = ya * T(T(1) / T(sqrt(_Arch[l-1])));
            yb = yb * T(T(1) / T(sqrt(_Arch[l-1])));

          }

          const Matrix<T> phi_l_a = _Links.at(l) * ya + _Biases.at(l);
          const Matrix<T> phi_l_b = _Links.at(l) * yb + _Biases.at(l);

          ya = Matrix<T> {phi_l_a, _ActFun}; // rho_l_a
          yb = Matrix<T> {phi_l_b, _ActFun}; // rho_l_b
          za = Matrix<T> {phi_l_a, _dActFun};
          zb = Matrix<T> {phi_l_b, _dActFun};

          // Compute the diagonal part of the equation
          std::vector<T> yav = ya.GetVector();
          std::vector<T> ybv = yb.GetVector();
          D = std::inner_product(std::begin(yav), std::end(yav), std::begin(ybv), T(0.0));

          // NTK_FLAG
          D = T(1.) + D / ( _NTKScaling ? T(_Arch[l]) : T(1.));

          // Compute the non-diagonal contribution
          for (int ia = 0; ia < H_ab.GetLines(); ia++){
              for (int ib = 0; ib < H_ab.GetColumns(); ib++){
                  T Hbar = za.GetElement(ia,0) * zb.GetElement(ib,0) * H_ab.GetElement(ia,ib);
                  H_ab.SetElement(ia, ib, Hbar);
                }
            }

          Matrix<T> WT{_Links.at(l+1)};
          WT.Transpose();
          Matrix<T> Maux = H_ab * WT;
          H_ab = Matrix<T> {_Links.at(l+1) * Maux};

          // NTK_FLAG
          if (_NTKScaling)
            H_ab = H_ab * T(T(1) / T(_Arch[l]));

          // Fill the diagonal entries of the NTK
          for (int i = 0; i < H_ab.GetLines(); i++){
            T aux = H_ab.GetElement(i,i) + D;
            H_ab.SetElement(i,i, aux);
          }
        }

      return H_ab;
    }

  Matrix<T> NTK_2 (std::vector<T> const& Input_a, std::vector<T> const& Input_b) const
  {
    const std::vector<double> dnnx_a = Derive(Input_a);
    const std::vector<double> dnnx_b = Derive(Input_b);

    const int Nout = _Arch.back();
    Matrix<T> H_ab {Nout, Nout, std::vector<T>(Nout *  Nout, T(0.))};

    //std::cout << "DEBUG NTK 2" << std::endl;

    for (int i = 0; i < Nout; i++){
      for (int j = 0; j < Nout; j++){
        T aux = T(0);
        for (int k = 1; k < _Np + 1; k++) {
          aux += dnnx_a[i + k * Nout] * dnnx_b[j + k * Nout];
        }
        std::cout << std::endl;
        H_ab.SetElement(i,j,aux);
      }
    }

    return H_ab;
  }

  private:
    const std::vector<int>             _Arch;
    const std::function<T(T const&)>   _ActFun;
    const std::function<T(T const&)>   _dActFun;
    const OutputFunction               _OutputFunc;
    std::function<T(T const&)>         _OutputActFun;
    std::function<T(T const&)>         _dOutputActFun;
    int                                _Np;
    std::map<int, Matrix<T>>           _Links;
    std::map<int, Matrix<T>>           _Biases;
    std::map<int, coordinates>         _IntMap;
    std::map<std::string, coordinates> _StrMap;
    std::map<std::string, int>         _StrIntMap;
    InitDistribution                   _InitDist;
    std::vector<std::vector<T>>        _DistParams;
    bool                               _NTKScaling;
  };
}
