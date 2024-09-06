//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//

#include "NNAD/FeedForwardNN.h"

#include <iostream>

int main()
{
  // Define architecture
  const std::vector<int> arch{3, 5, 5, 4};

  // Initialise NN
  const nnad::FeedForwardNN<double> nn{arch, 0, nnad::OutputFunction::QUADRATIC, true};

  // Input vector
  const std::vector<double> x{0.1, 2.3, 4.5};
  std::cout << "Input: x = { ";
  for (auto const& e : x)
    std::cout << e << " ";
  std::cout << "}\n" << std::endl;

  // Get derivatives of the NN at x
  const nnad::Matrix<double> anders = nn.EvaluatePrime(x);

  // Compute the derivatives numerically as incremental ratios
  const double eps = 1e-5;
  nnad::Matrix<double> numders{arch.back(), arch.front()};
  for (int p = 0; p < (int) arch.front(); p++)
    {
      std::vector<double> xp = x;
      std::vector<double> xm = x;
      xp[p] += eps;
      xm[p] -= eps;
      const std::vector<double> nnp = nn.Evaluate(xp);
      const std::vector<double> nnm = nn.Evaluate(xm);
      for (int k = 0; k < (int) arch.back(); k++)
        numders.SetElement(k, p, ( nnp[k] - nnm[k] ) / 2 / eps);
    }

  // Compare derivatives
  std::cout << std::scientific;
  std::cout << "dN_k/dx_p:" << std::endl;
  std::cout << "k\tp\tnum. ders.\tan. ders.\tratio" << std::endl;
  for (int k = 0; k < (int) arch.back(); k++)
    for (int p = 0; p < (int) arch.front(); p++)
      std::cout << k << "\t" << p << "\t" << numders.GetElement(k, p) << "\t" << anders.GetElement(k, p) << "\t" << numders.GetElement(k, p) / anders.GetElement(k, p) << std::endl;
  std::cout << "\n";

  return 0;
}
