#include "Matrix.h"
#include "Globals.h"

template <class T>
Matrix<T>::Matrix(int const &Lines, int const &Columns, int const &RandomSeed) : _Lines(Lines),
                                                                                 _Columns(Columns),
                                                                                 _Matrix(_Lines * _Columns)
{
  // Initialise random number generator.
  //srand(RandomSeed);
  // Fill in the matrix with random numbers distributed in [-1:1].
  for (int i = 0; i < _Lines * _Columns; i++)
    _Matrix[i] = T(2e-2 * (rand() % 100) - 1);
  /*
  std::default_random_engine generator(RandomSeed);
  double sd = sqrt(2. / Columns); //He-et-al Initialization [see:https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e]
  std::normal_distribution<double> dist(0.0, sd);

  for (int i = 0; i < _Lines * _Columns; i++)
  {
    double init = dist(generator);

    while (init > 2. * sd || init < -2. * sd)
    {
      init = dist(generator);
    }

    _Matrix[i] = T(init);
  }
  */
  }

  template <class T>
  Matrix<T>::Matrix(int const &Lines, int const &Columns, std::vector<T> const &Entries) : _Lines(Lines),
                                                                                        _Columns(Columns),
                                                                                        _Matrix(Entries)
  {
    // Check that the size of the Entries match the size of the matrix
    if (_Lines * _Columns != (int)Entries.size())
      Error("Matrix: the size of the input vector does not match the size of the matrix.");
  }

  template <class T>
  Matrix<T>::Matrix(Matrix const &x, std::function<T(T const &)> f) : _Lines(x.GetLines()),
                                                                                                       _Columns(x.GetColumns()),
                                                                                                       _Matrix(_Lines * _Columns)
  {
    const std::vector<T> xv = x.GetVector();
    const int n = (int)xv.size();
    for (int i = 0; i < n; i++)
      _Matrix[i] = f(xv[i]);
  }

  // Setter
  template <class T>
  void Matrix<T>::SetElement(int const &i, int const &j, T const &value)
  {
    if (i < 0 || i > _Lines)
      Error("SetElement: line index out of range.");

    if (j < 0 || j > _Columns)
      Error("SetElement: column index out of range.");

    _Matrix[i * _Columns + j] = value;
  }

  // Getters
  template <class T>
  T Matrix<T>::GetElement(int const &i, int const &j) const
  {
    if (i < 0 || i > _Lines)
      Error("GetElement: line index out of range.");

    if (j < 0 || j > _Columns)
      Error("GetElement: column index out of range.");

    return _Matrix[i * _Columns + j];
  }

  // Operators
  template <class T>
  void Matrix<T>::operator=(Matrix const &term)
  {
    _Lines = term.GetLines();
    _Columns = term.GetColumns();
    _Matrix = term.GetVector();
  }

  template <class T>
  Matrix<T> Matrix<T>::operator+(Matrix const &term)
  {
    const int l = term.GetLines();
    const int c = term.GetColumns();
    if (_Lines != l || _Columns != c)
      Error("Lines or Columns don't match adding the two matrices.");

    Matrix result(l, c);
    for (int i = 0; i < l; i++)
      for (int j = 0; j < c; j++)
        result.SetElement(i, j, _Matrix[i * _Columns + j] + term.GetElement(i, j));

    return result;
  }

  template <class T>
  Matrix<T> Matrix<T>::operator-(Matrix const &term)
  {
    const int l = term.GetLines();
    const int c = term.GetColumns();
    if (_Lines != l || _Columns != c)
      Error("Lines or Columns don't match adding the two matrices.");

    Matrix result(l, c);
    for (int i = 0; i < l; i++)
      for (int j = 0; j < c; j++)
        result.SetElement(i, j, _Matrix[i * _Columns + j] - term.GetElement(i, j));

    return result;
  }

  template <class T>
  void Matrix<T>::operator+=(Matrix const &term)
  {
    const int l = term.GetLines();
    const int c = term.GetColumns();
    if (_Lines != l || _Columns != c)
      Error("Lines or Columns don't match adding the two matrices.");

    for (int i = 0; i < l; i++)
      for (int j = 0; j < c; j++)
        _Matrix[i * _Columns + j] += term.GetElement(i, j);
  }

  template <class T>
  void Matrix<T>::operator-=(Matrix const &term)
  {
    const int l = term.GetLines();
    const int c = term.GetColumns();
    if (_Lines != l || _Columns != c)
      Error("Lines or Columns don't match adding the two matrices.");

    for (int i = 0; i < l; i++)
      for (int j = 0; j < c; j++)
        _Matrix[i * _Columns + j] -= term.GetElement(i, j);
  }

  template <class T>
  Matrix<T> Matrix<T>::operator*(Matrix const &term) const
  {
    const int l1 = _Lines;
    const int c1 = _Columns;
    const int l2 = term.GetLines();
    const int c2 = term.GetColumns();
    if (c1 != l2)
      Error("Lines or Columns don't match multiplying the two matrices.");

    Matrix result(l1, c2);
    for (int i = 0; i < c2; i++)
      for (int j = 0; j < l1; j++)
      {
        T value=T(0);
        for (int k = 0; k < c1; k++)
          value += _Matrix[j * _Columns + k] * term.GetElement(k, i);

        result.SetElement(j, i, value);
      }

    return result;
  }

  template <class T>
  void Matrix<T>::operator*=(Matrix const &term)
  {
    const int l1 = _Lines;
    const int c1 = _Columns;
    const int l2 = term.GetLines();
    const int c2 = term.GetColumns();
    if (c1 != l2)
      Error("Lines or Columns don't match multiplying the two matrices.");

    for (int i = 0; i < c2; i++)
      for (int j = 0; j < l1; j++)
      {
        T value = T(0);
        for (int k = 0; k < c1; k++)
          value += _Matrix[j * _Columns + k] * term.GetElement(k, i);

        _Matrix[i * _Columns + j] = value;
      }
  }

  template <class T>
  Matrix<T> Matrix<T>::operator*(T const &coef)
  {
    Matrix result(_Lines, _Columns);
    for (int i = 0; i < _Lines; i++)
      for (int j = 0; j < _Columns; j++)
        result.SetElement(i, j, coef * _Matrix[i * _Columns + j]);

    return result;
  }

  // template fixed types
  template class Matrix<double>; //<! for numeric and analytic
  template class Matrix<ceres::Jet<double, GLOBALS::kStride>>; //<! for automatic
