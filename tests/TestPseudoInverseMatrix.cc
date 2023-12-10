//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//

#include <iostream>

#include "NNAD/FeedForwardNN.h"

int main()
{
  nnad::Matrix<double> X{3, 3, std::vector<double>{1., 2., 3., 0., 4., 5., 1., 0., 6.}};
  std::cout << "Matrix = " << std::endl;
  X.Display();

  nnad::Matrix<double> CoFactor = X.GetCofactor(0, 0);
  std::cout << "Cofactor(0,0) = " << std::endl;
  CoFactor.Display();

  double determinant = X.Determinant();
  std::cout << "Determinant = " << determinant << std::endl;

  nnad::Matrix<double> adjoint = X.Adjoint();
  std::cout << "Adjoint = " << std::endl;
  adjoint.Display();

  nnad::Matrix<double> inverse = X.Inverse();
  std::cout << "Inverse = " << std::endl;
  inverse.Display();

  nnad::Matrix<double> pseudoinverse = X.PseudoInverse_LLR();
  std::cout << "Pseudoinverse = " << std::endl;
  pseudoinverse.Display();

  nnad::Matrix<double> X2{5, 13, std::vector<double>{0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,
                                                     0,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  0,  0,
                                                     0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,
                                                     0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,
                                                     0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0
                                                    }};

  std::cout << "Matrix 2 = " << std::endl;
  X2.Display();

  nnad::Matrix<double> pseudoinverse2 = X2.PseudoInverse_LLR();
  std::cout << "Pseudoinverse 2 = " << std::endl;
  pseudoinverse2.Display();

  return 0;
}
