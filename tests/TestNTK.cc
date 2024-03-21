//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//

#include "NNAD/FeedForwardNN.h"

#include <iostream>
#include <time.h>

int main()
{
  // Define architecture
  const std::vector<int> arch{30, 1000, 1000, 1};
  const std::vector<double> Cb{0, 0, 0};
  const std::vector<double> Cw{1., 1., 1.};
  std::vector<std::vector<double>> C;

  time_t timer;

  std::default_random_engine g ( time(&timer));

  for (int i = 0; i < (int) Cb.size(); i++)
    C.push_back(std::vector<double> {Cb[i], Cw[i]});

  // Initialise NN
  const nnad::FeedForwardNN<double> nn{arch, int(g()), true, nnad::Tanh<double>, nnad::dTanh<double>,
                                        nnad::OutputFunction::LINEAR,
                                        nnad::InitDistribution::GAUSSIAN, C, true};


  // Input vector
  std::vector<double> x1 (arch[0], 0.);
  for (auto& x : x1)
    {
      std::uniform_real_distribution<double> d (0., 1.0);
      x = d(g);
    }

  std::vector<double> x2 (arch[0], 0.);
  for (auto& x : x1)
    {
      std::uniform_real_distribution<double> d (0., 1.0);
      x = d(g);
    }
  std::cout << "Input: x = { ";
  for (auto const& e : x1)
    std::cout << e << " ";
  std::cout << "}\n" << std::endl;

  // Get NN at x
  std::cout << "Starting evaluation" << std::endl;
  std::vector<double> ders = nn.Evaluate(x1);

  std::cout << "Printing values" << std::endl;
  for (auto& d : ders)
    std::cout << d << std::endl;
  std::cout << "_________________________" << std::endl;

  // Compute the derivatives numerically as incremental ratios
  const double eps = 1e-5;
  const std::vector<double> pars = nn.GetParameters();
  const int np = (int) pars.size();

  nnad::Matrix<double> result = nn.NTK(x1, x2);
  result.Display();

  return 0;
}
