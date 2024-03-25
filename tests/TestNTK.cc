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
  const std::vector<int> arch{30, 100, 100, 1};
  const std::vector<double> Cb{0, 0, 0};
  const std::vector<double> Cw{1., 1., 1.};
  std::vector<std::vector<double>> C;

  time_t timer;

  std::default_random_engine g ( time(&timer));

  for (int i = 0; i < (int) Cb.size(); i++)
    C.push_back(std::vector<double> {Cb[i], Cw[i]});

  // Initialise NN
  const nnad::FeedForwardNN<double> nn{arch, int(g()), true, nnad::Sigmoid<double>, nnad::dSigmoid<double>,
                                        nnad::OutputFunction::LINEAR,
                                        nnad::InitDistribution::GAUSSIAN, {}, true};


  // Input vector
  std::vector<double> x1 (arch[0], 0.);
  for (auto& x : x1) {
      std::uniform_real_distribution<double> d (0., 1.0);
      x = d(g);
    }

  std::vector<double> x2 (arch[0], 0.);
  for (auto& x : x2) {
      std::uniform_real_distribution<double> d (0., 1.0);
      x = d(g);
    }

  std::cout << "Input: x_1 = { ";
  for (auto const& e : x1)
    std::cout << e << " ";
  std::cout << "}\n" << std::endl;

    std::cout << "Input: x_2 = { ";
  for (auto const& e : x2)
    std::cout << e << " ";
  std::cout << "}\n" << std::endl;

  nnad::Matrix<double> result = nn.NTK(x1, x2);
  nnad::Matrix<double> result2 = nn.NTK_2(x1, x2);
  std::cout << "Printing method 1" << std::endl;
  std::cout << "_________________________" << std::endl;
  result.Display();
  std::cout << "Printing method 2" << std::endl;
  std::cout << "_________________________" << std::endl;
  result2.Display();

  return 0;
}
