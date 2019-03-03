#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <random>
#include <stdlib.h>

#include "Utilities.h"
#include "ceres/jet.h"

template <class T>
class Matrix
{
public:
  // Constructors

  Matrix(int const &Lines, int const &Columns, int const &RandomSeed = -1);
  Matrix(int const &, int const &, std::vector<T> const &);

  Matrix(Matrix const &x, std::function<T(T const &)> f );

  // Setter
  void SetElement(int const &, int const &, T const &);

  // Getters
  T GetElement(int const &, int const &) const;

  int GetLines() const { return _Lines; };
  int GetColumns() const { return _Columns; };
  std::vector<T> GetVector() const { return _Matrix; };

  // Operators
  void operator=(Matrix const &);

  Matrix operator+(Matrix const &);

  Matrix operator-(Matrix const &);

  void operator+=(Matrix const &);

  void operator-=(Matrix const &);

  Matrix operator*(Matrix const &) const;

  void operator*=(Matrix const &);

  Matrix operator*(T const &);

private:
  int _Lines;
  int _Columns;
  std::vector<T> _Matrix;
};
