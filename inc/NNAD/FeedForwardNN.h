//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//

#pragma once

#include "Matrix.h"

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
   *
   * More recently, we have introduced the possibility to provide the
   * NN with an arbitrary preprocessing function that can depend on
   * any number of parameters. The resulting derivative also accouns
   * for this additional function. This new functionality is backward
   * compatible.
   */
  template<class T>
  class FeedForwardNN
  {
  public:
    // Typedef for the preprocessing function. It is a std::function
    // that returns a vector of T with "( _NpPrep + 1 ) * _Nout"
    // outputs, where is "_NpPrep" the number of free parameters of
    // the preprocessing function and "_Nout" is the number of outputs
    // of the NN. The preprocessing function is a function of another
    // vector of T with as many entries as inputs of the NN, and a
    // vector of T containing the "_NpPrep" parameters.
    typedef std::function<std::vector<T>(std::vector<T> const&, std::vector<T> const&)> Preprocessing;

    //_________________________________________________________________________________
    FeedForwardNN(std::vector<int>           const& Arch,
                  int                        const& RandomSeed,
                  bool                       const& Report = false,
                  std::function<T(T const&)> const& ActFun = Sigmoid<T>,
                  std::function<T(T const&)> const& dActFun = dSigmoid<T>,
                  OutputFunction             const& OutputFunc = LINEAR,
                  Preprocessing              const& Preproc = nullptr,
                  std::vector<T>             const& PreprocPars = {}):
      _Arch(Arch),
      _ActFun(ActFun),
      _dActFun(dActFun),
      _OutputFunc(OutputFunc),
      _Preproc(Preproc),
      _PreprocPars(_Preproc == nullptr ? std::vector<T> {} : PreprocPars),
      _NpPrep((int) _PreprocPars.size())
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

      // Number of parameters of the NN
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
          _Links.insert({l, Matrix<T>{_Arch[l], _Arch[l - 1], sub_seeds.at(iseed++)}});
          _Biases.insert({l, Matrix<T>{_Arch[l], 1, sub_seeds.at(iseed++)}});
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

      // Check sanity of preprocessing function, if present
      if (_Preproc != nullptr)
        {
          // Generate random inputs between 0 and 1 but recasted as T
          std::vector<T> InPrep;
          for (int i = 0; i < _Arch.front(); i++)
            InPrep.push_back(T{1e-2 * ( rand() % 100 )});

          // Test preprocessing function
          const std::vector<T> OutPrep = _Preproc(InPrep, _PreprocPars);

          // Check that the number of outputs of the preprocessing
          // function is the same as that of the NN.
          if (OutPrep.size() != ( _NpPrep + 1 ) * _Arch.back())
            Error("FeedForwardNN: preprocessing function not well-formed.");
        }

      // Report NN parameters
      if (Report)
        {
          std::cout << "\nFeed-forward neural network initialised:" << std::endl;
          std::cout << "- architecture = [ ";
          for (auto const& n : _Arch)
            std::cout << n << " ";
          std::cout << "]" << std::endl;
          if (_Preproc != nullptr)
            {
              std::cout << "- preprocessing function provided" << std::endl;
              std::cout << "- number of parameters (NN + preprocessing) = " << _Np << " + " << _NpPrep << " (" << _Np + _NpPrep << ")" << std::endl;
            }
          else
            std::cout << "- number of parameters = " << _Np << std::endl;
          std::cout << "- NN output function: ";
          switch (_OutputFunc)
            {
            case ACTIVATION:
              std::cout << "activation-like" << std::endl;
              break;
            case LINEAR:
              std::cout << "linear" << std::endl;
              break;
            case QUADRATIC:
              std::cout << "quadratic" << std::endl;
              break;
            default:
              std::cout << "unknown" << std::endl;
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
                  Preprocessing              const& Preproc = nullptr,
                  std::vector<T>             const& PreprocPars = {},
                  std::function<T(T const&)> const& ActFun = Sigmoid<T>,
                  std::function<T(T const&)> const& dActFun = dSigmoid<T>):
      FeedForwardNN(Arch, RandomSeed, Report, ActFun, dActFun, OutputFunc, Preproc, PreprocPars)
    {
    }

    //_________________________________________________________________________________
    FeedForwardNN(std::vector<int>           const& Arch,
                  std::vector<T>             const& Pars,
                  std::function<T(T const&)> const& ActFun = Sigmoid<T>,
                  std::function<T(T const&)> const& dActFun = dSigmoid<T>,
                  OutputFunction             const& OutputFunc = LINEAR,
                  bool                       const& Report = false,
                  Preprocessing              const& Preproc = nullptr,
                  std::vector<T>             const& PreprocPars = {}):
      FeedForwardNN(Arch, 0, Report, ActFun, dActFun, OutputFunc, Preproc, PreprocPars)
    {
      SetParameters(Pars);
    }

    //_________________________________________________________________________________
    void SetParameters(std::vector<T> const& Pars)
    {
      // Check that the size of the parameter vector is equal to the
      // number of parameters of the current NN plus the number of
      // parameters of the preprocessing function.
      if ((int) Pars.size() != _Np + _NpPrep)
        Error("SetParameters: the number of parameters does not match that of the NN plus that of the preprocessing function.");

      // Set links and biases
      for (int ip = 0; ip < _Np; ip++)
        {
          const coordinates coord = _IntMap.at(ip);
          if (std::get<0>(coord))
            _Links.at(std::get<1>(coord)).SetElement(std::get<2>(coord), std::get<3>(coord), Pars[ip]);
          else
            _Biases.at(std::get<1>(coord)).SetElement(std::get<2>(coord), std::get<3>(coord), Pars[ip]);
        }

      // Set preprocessing parameters
      for (int ip = 0; ip < _NpPrep; ip++)
        _PreprocPars[ip] = Pars[_Np + ip];
    }

    //_________________________________________________________________________________
    void SetParameter(int const& ip, T const& Par)
    {
      // Check that the index "ip" is within the allowed bounds
      if (ip < 0 || ip >= _Np + _NpPrep)
        Error("SetParameter: index 'ip' out of range.");

      // If ip < _Np the parameter corresponds to a NN parameter...
      if (ip < _Np)
        {
          // Get coordinates
          const coordinates coord = _IntMap.at(ip);

          if (std::get<0>(coord))
            _Links.at(std::get<1>(coord)).SetElement(std::get<2>(coord), std::get<3>(coord), Par);
          else
            _Biases.at(std::get<1>(coord)).SetElement(std::get<2>(coord), std::get<3>(coord), Par);
        }
      // ...otherwise it's a preprocessing parameter
      else
        _PreprocPars[ip - _Np] = Par;
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
    void SetPreprocessingParameters(std::vector<T> const& ProprocPars)
    {
      // Check that the size of the parameter vector is equal to the
      // number of parameters of the preprocessing function.
      if ((int) ProprocPars.size() != _NpPrep)
        Error("SetPreprocessingParameters: the number of parameters does not match that of the preprocessing function.");

      // Set parameters
      _PreprocPars = ProprocPars;
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
    Preprocessing PreprocessingFunction() const
    {
      return _Preproc;
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
      return _Np + _NpPrep;
    }

    //_________________________________________________________________________________
    int GetNeuralNetworkParameterNumber() const
    {
      return _Np;
    }

    //_________________________________________________________________________________
    int GetPreprocessingParameterNumber() const
    {
      return _NpPrep;
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
      // Initialise output vector
      std::vector<T> output(_Np + _NpPrep);

      // Element counter
      int count = 0;

      // Number of layers
      const int nl = (int) _Arch.size();

      // Neural network parameters
      for (int l = nl - 1; l > 0; l--)
        for (int i = 0; i < _Arch[l]; i++)
          {
            // Biases
            output[count++] = _Biases.at(l).GetElement(i, 0);

            // Links
            for (int j = 0; j < _Arch[l - 1]; j++)
              output[count++] = _Links.at(l).GetElement(i, j);
          }

      // Preprocessing parameters
      for (int ip = 0; ip < _NpPrep; ip++)
        output[count++] = _PreprocPars[ip];

      return output;
    }

    //_________________________________________________________________________________
    std::vector<T> GetNueralNetworkParameters() const
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
    std::vector<T> GetPreprocessingParameters() const
    {
      return _PreprocPars;
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
        y = Matrix<T> {_Links.at(l) * y + _Biases.at(l), _ActFun};

      // Compute NN
      std::vector<T> NN = (Matrix<T> {_Links.at(nl - 1) * y + _Biases.at(nl - 1), _OutputActFun}).GetVector();

      // Include preprocessing function if necessary
      if (_Preproc != nullptr)
        {
          const std::vector<T> prep = _Preproc(Input, _PreprocPars);
          std::transform(NN.begin(), NN.end(), prep.begin(), NN.begin(), std::multiplies<T>());
        }

      return NN;
    }

    //_________________________________________________________________________________
    std::vector<T> Derive(std::vector<T> const& Input) const
    {
      // Check that the size of the input vector is equal to the number of
      // input nodes.
      if ((int) Input.size() != _Arch[0])
        Error("Derive: the number of inputs does not match the number of input nodes.");

      // Initialise dNN vector
      std::vector<T> dNN((_Np + 1) * _Arch.back());

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
          const Matrix<T> M = _Links.at(l) * y.at(l - 1) + _Biases.at(l);
          y.insert({l, Matrix<T>{M, _ActFun}});
          z.insert({l, Matrix<T>{M, _dActFun}});
        }

      // Now take care of the output layer according to the selected
      // output function.
      const Matrix<T> M = _Links.at(nl - 1) * y.at(nl - 2) + _Biases.at(nl - 1);
      y.insert({nl - 1, Matrix<T>{M, _OutputActFun}});
      z.insert({nl - 1, Matrix<T>{M, _dOutputActFun}});

      // Output vector of vectors. Push back NN itself as a first
      // element.
      for (auto const& e : y.at(nl - 1).GetVector())
        dNN[count++] = e;

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
                dNN[count++] = Sigma.GetElement(k, i) * zl[i];

              // Compute derivatives w.r.t. the links
              for (int j = 0; j < _Arch[l - 1]; j++)
                for (int k = 0; k < _Arch[nl - 1]; k++)
                  dNN[count++] = Sigma.GetElement(k, i) * zl[i] * ylmo[j];
            }

          // Compute Matrix "S" on this layer
          std::vector<T> entries;
          for (int i = 0; i < _Arch[l]; i++)
            for (int j = 0; j < _Arch[l - 1]; j++)
              entries.push_back(zl[i] * _Links.at(l).GetElement(i, j));
          const Matrix<T> S{_Arch[l], _Arch[l - 1], entries};

          // Update Matrix "Sigma" for the next step
          Sigma = Sigma * S;
        }

      // Include preprocessing function if necessary
      if (_Preproc != nullptr)
        {
          // Get number of outputs
          const int Nout = _Arch.back();

          // Append to the vector dNN the NN mupliplied by the
          // derivative of the preprocessing function.
          for (int ip = 0; ip < _NpPrep; ip++)
            dNN.insert(dNN.end(), dNN.begin(), dNN.begin() + Nout);

          // Get preprocessing function
          const std::vector<T> prep = _Preproc(Input, _PreprocPars);

          // Multiply dNN by the preprocessing function, i.e. the
          // first Nout outputs of prep.
          for (int ip = 0; ip < _Np + 1; ip++)
            std::transform(dNN.begin() + Nout * ip, dNN.begin() + Nout * ( ip + 1 ), prep.begin(), dNN.begin() + Nout * ip, std::multiplies<T>());

          // Now multiply the last _NpPrep bunches of Nout entries of
          // dNN (that for now only contain _NpPrep copies the NN
          // itself) by the _NpPrep derivatives of the preprocessing
          // function.
          for (int ip = _Np + 1; ip < _Np + _NpPrep + 1; ip++)
            std::transform(dNN.begin() + Nout * ip, dNN.begin() + Nout * ( ip + 1 ), prep.begin() + Nout * ( ip - _Np ), dNN.begin() + Nout * ip, std::multiplies<T>());
        }

      return dNN;
    }

  private:
    const std::vector<int>             _Arch;
    const std::function<T(T const&)>   _ActFun;
    const std::function<T(T const&)>   _dActFun;
    const OutputFunction               _OutputFunc;
    const Preprocessing                _Preproc;
    std::vector<T>                     _PreprocPars;
    const int                          _NpPrep;
    std::function<T(T const&)>         _OutputActFun;
    std::function<T(T const&)>         _dOutputActFun;
    int                                _Np;
    std::map<int, Matrix<T>>           _Links;
    std::map<int, Matrix<T>>           _Biases;
    std::map<int, coordinates>         _IntMap;
    std::map<std::string, coordinates> _StrMap;
    std::map<std::string, int>         _StrIntMap;
  };
}
