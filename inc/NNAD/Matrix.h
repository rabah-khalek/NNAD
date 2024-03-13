//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//

#pragma once


#include "Utilities.h"
#include "Distributions.h"

#include <vector>
#include <functional>
#include <random>

namespace nnad
{
  template <class T>
  class Matrix
  {
  public:
    //_________________________________________________________________________________
    Matrix()
    {
    }

    //_________________________________________________________________________________
    Matrix(Matrix<T> const& M)
    {
      _Lines   = M.GetLines();
      _Columns = M.GetColumns();
      _Matrix  = M.GetVector();
    }

    //_________________________________________________________________________________
    Matrix(int              const& Lines,
           int              const& Columns,
           int              const& RandomSeed = -1,
           InitDistribution const& InitDist = UNIFORM,
           std::vector<T>   const& Params = {}
          ):
      _Lines(Lines),
      _Columns(Columns),
      _Matrix(_Lines * _Columns)
    {
      // Initialise the distribution for the entries of the matrix
      Distribution<T> *d = DistributionFactory<T>(InitDist, Params, RandomSeed);

      // Fill the elements of the matrix
      for (int i = 0; i < _Lines * _Columns; i++)
        _Matrix[i] = d->operator()();
    }

    //_________________________________________________________________________________
    Matrix(int const& Lines, int const& Columns, std::vector<T> const& Entries):
      _Lines(Lines),
      _Columns(Columns),
      _Matrix(Entries)
    {
      // Check that the size of the Entries match the size of the
      // matrix.
      if (_Lines * _Columns != (int) Entries.size())
        Error("Matrix: the size of the input vector does not match the size of the matrix.");
    }

    //_________________________________________________________________________________
    Matrix(Matrix const& x, std::function<T(T const&)> f):
      _Lines(x.GetLines()),
      _Columns(x.GetColumns()),
      _Matrix(_Lines * _Columns)
    {
      const std::vector<T> xv = x.GetVector();
      for (int i = 0; i < (int) xv.size(); i++)
        _Matrix[i] = f(xv[i]);
    }

    //_________________________________________________________________________________
    void Transpose()
    {
      std::vector<T> new_Matrix(_Columns * _Lines);
      for (int i = 0; i < _Lines; i++)
        for (int j = 0; j < _Columns; j++)
          new_Matrix[j*_Lines + i] = _Matrix[i * _Columns + j];

      const int temp = _Lines;
      _Lines   = _Columns;
      _Columns = temp;
      _Matrix  = new_Matrix;
    }

    // Function to get cofactor matrix
    //_________________________________________________________________________________
    Matrix<T> GetCofactor(int p, int q)
    {
      if (_Lines != _Columns)
        Error("GetCofactor: matrix must be square.");

      Matrix<T> output{_Lines-1, _Columns-1};
      int i = 0, j = 0;

      // Looping for each element of the matrix
      for (int row = 0; row < _Lines; row++)
        for (int col = 0; col < _Columns; col++)
          // Copying into temporary matrix only those element which
          // are not in given row and column.
          if (row != p && col != q)
            {
              output.SetElement(i,j++,  this->GetElement(row,col));

              // Row is filled, so increase row index and reset col
              // index.
              if (j == _Lines - 1)
                {
                  j = 0;
                  i++;
                }
            }
      return output;
    }

    // Recursive function for finding Determinant of matrix
    //_________________________________________________________________________________
    T Determinant()
    {
      if (_Lines != _Columns)
        Error("Determinant: matrix must be square.");

      T D = 0; // Initialize result

      //  Base case : if matrix contains single element
      if (_Lines == 1)
        return this->GetElement(0,0);

      // To store sign multiplier
      int sign = 1;

      // Iterate for each element of first row
      for (int f = 0; f < _Lines; f++)
        {
          // Getting Cofactor
          Matrix<T> Cofactor = this->GetCofactor(0, f);
          D += sign * this->GetElement(0, f) * Cofactor.Determinant();

          // terms are to be added with alternate sign
          sign = -sign;
        }
      return D;
    }

    // Function to get Adjoint matrix
    //_________________________________________________________________________________
    Matrix<T> Adjoint()
    {
      if (_Lines != _Columns)
        Error("Adjoint: matrix must be square.");

      Matrix<T> output{_Lines, _Columns};

      if (_Lines == 1)
        {
          output.SetElement(0,0, T{1});
          return output;
        }
      int sign = 1;

      for (int i = 0; i < _Lines; i++)
        for (int j = 0; j < _Columns; j++)
          {
            // Get cofactor
            Matrix<T> Cofactor = this->GetCofactor(i, j);

            // sign of adj[j][i] positive if sum of row
            // and column indexes is even.
            sign = ((i + j) % 2 == 0) ? 1 : -1;

            // Interchanging rows and columns to get the
            // transpose of the cofactor matrix
            output.SetElement(j, i, (sign) * Cofactor.Determinant());
          }
      return output;
    }

    // Function to calculate inverse, breaks if matrix is singular
    // _________________________________________________________________________________
    Matrix<T> Inverse()
    {
      if (_Lines != _Columns)
        Error("inverse: matrix must be square.");

      Matrix<T> output{_Lines, _Columns};

      // Find Determinant
      T det = this->Determinant();
      if (det == 0)
        Error("inverse: Singular matrix, can't find its inverse");

      // Find Adjoint
      Matrix<T> adj = this->Adjoint();

      // Find Inverse using formula "inverse = adj/det"
      for (int i = 0; i < _Lines; i++)
        for (int j = 0; j < _Columns; j++)
          output.SetElement(i,j, adj.GetElement(i,j) / T{det});

      return output;
    }

    // Function to calculate (Moore-Penrose) pseudo-inverse, breaks if
    // matrix is singular for Linearly independent rows Matrix.
    // _________________________________________________________________________________
    Matrix<T> PseudoInverse_LLC()
    {
      Matrix<T> transpose = (*this);
      transpose.Transpose();
      Matrix<T> temp = transpose * (*this);
      Matrix<T> temp_inverse = temp.Inverse();
      Matrix<T> output = temp_inverse * transpose;
      return output;
    }

    // Function to calculate (Moore-Penrose) pseudo-inverse, breaks if
    // matrix is singular for Linearly independent rows Matrix.
    // _________________________________________________________________________________
    Matrix<T> PseudoInverse_LLR()
    {
      Matrix<T> transpose = (*this);
      transpose.Transpose();
      Matrix<T> temp = (*this) * transpose;
      Matrix<T> temp_inverse = temp.Inverse();
      Matrix<T> output = transpose * temp_inverse;
      return output;
    }

    //_________________________________________________________________________________
    void SetElement(int const& i, int const& j, T const& value)
    {
      if (i < 0 || i > _Lines)
        Error("SetElement: line index out of range.");

      if (j < 0 || j > _Columns)
        Error("SetElement: column index out of range.");

      _Matrix[i * _Columns + j] = value;
    }

    //_________________________________________________________________________________
    T GetElement(int const& i, int const& j) const
    {
      if (i < 0 || i > _Lines)
        Error("GetElement: line index out of range.");

      if (j < 0 || j > _Columns)
        Error("GetElement: column index out of range.");

      return _Matrix[i * _Columns + j];
    }

    //_________________________________________________________________________________
    int GetLines() const
    {
      return _Lines;
    }

    //_________________________________________________________________________________
    int GetColumns() const
    {
      return _Columns;
    }

    //_________________________________________________________________________________
    std::vector<T> GetVector() const
    {
      return _Matrix;
    }

    //_________________________________________________________________________________
    std::vector<T> GetLine(int const& i) const
    {
      std::vector<T> line{_Columns};
      for (int j = 0; j < _Columns; j++)
        line[j] = GetElement(i, j);
      return line;
    }

    //_________________________________________________________________________________
    std::vector<T> GetColumn(int const& j) const
    {
      std::vector<T> column{_Lines};
      for (int i = 0; i < _Lines; i++)
        column[i] = GetElement(i, j);
      return column;
    }

    //_________________________________________________________________________________
    void operator = (Matrix<T> const& term)
    {
      _Lines   = term.GetLines();
      _Columns = term.GetColumns();
      _Matrix  = term.GetVector();
    }

    //_________________________________________________________________________________
    Matrix<T> operator + (Matrix<T> const& term)
    {
      const int l = term.GetLines();
      const int c = term.GetColumns();
      if (_Lines != l || _Columns != c)
        Error("Lines or Columns don't match adding the two matrices.");

      Matrix result{l, c};
      for (int i = 0; i < l; i++)
        for (int j = 0; j < c; j++)
          result.SetElement(i, j, _Matrix[i * _Columns + j] + term.GetElement(i, j));

      return result;
    }

    //_________________________________________________________________________________
    Matrix<T> operator - (Matrix<T> const& term)
    {
      const int l = term.GetLines();
      const int c = term.GetColumns();
      if (_Lines != l || _Columns != c)
        Error("Lines or Columns don't match adding the two matrices.");

      Matrix result{l, c};
      for (int i = 0; i < l; i++)
        for (int j = 0; j < c; j++)
          result.SetElement(i, j, _Matrix[i * _Columns + j] - term.GetElement(i, j));

      return result;
    }

    //_________________________________________________________________________________
    void operator += (Matrix<T> const& term)
    {
      const int l = term.GetLines();
      const int c = term.GetColumns();
      if (_Lines != l || _Columns != c)
        Error("Lines or Columns don't match adding the two matrices.");

      for (int i = 0; i < l; i++)
        for (int j = 0; j < c; j++)
          _Matrix[i * _Columns + j] += term.GetElement(i, j);
    }

    //_________________________________________________________________________________
    void operator -= (Matrix<T> const& term)
    {
      const int l = term.GetLines();
      const int c = term.GetColumns();
      if (_Lines != l || _Columns != c)
        Error("Lines or Columns don't match adding the two matrices.");

      for (int i = 0; i < l; i++)
        for (int j = 0; j < c; j++)
          _Matrix[i * _Columns + j] -= term.GetElement(i, j);
    }

    //_________________________________________________________________________________
    Matrix<T> operator * (Matrix<T> const& term) const
    {
      const int l1 = _Lines;
      const int c1 = _Columns;
      const int l2 = term.GetLines();
      const int c2 = term.GetColumns();
      if (c1 != l2)
        Error("Lines or Columns don't match multiplying the two matrices.");

      Matrix result{l1, c2};
      for (int i = 0; i < c2; i++)
        for (int j = 0; j < l1; j++)
          {
            T value = T(0);
            for (int k = 0; k < c1; k++)
              value += _Matrix[j * _Columns + k] * term.GetElement(k, i);

            result.SetElement(j, i, value);
          }

      return result;
    }

    //_________________________________________________________________________________
    void operator *= (Matrix<T> const& term)
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

    //_________________________________________________________________________________
    Matrix<T> operator * (T const& coef)
    {
      Matrix result{_Lines, _Columns};
      for (int i = 0; i < _Lines; i++)
        for (int j = 0; j < _Columns; j++)
          result.SetElement(i, j, coef * _Matrix[i * _Columns + j]);

      return result;
    }

    //_________________________________________________________________________________
    void Display()
    {
      for (int i = 0; i < _Lines; i++)
        {
          for (int j = 0; j < _Columns; j++)
            std::cout << this->GetElement(i,j) << " ";
          std::cout << std::endl;
        }
      std::cout << std::endl;
    }

  private:
    int                               _Lines;
    int                               _Columns;
    std::vector<T>                    _Matrix;
  };
}
