//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//

#include "NNAD/FeedForwardNN.h"

#include <iostream>

int main()
{
  // Define architecture
  const std::vector<int> arch{3, 5, 5, 3};

  // Initialise NN
  const nnad::FeedForwardNN<double> nn{arch, 0, nnad::OutputFunction::QUADRATIC, true};
  nnad::FeedForwardNN<double> nn_copy{nn};

  // Input vector
  std::vector<double> x{0.1, 2.3, 4.5};
  std::cout << "Input: x = { ";
  for (auto const& e : x)
    std::cout << e << " ";
  std::cout << "}\n" << std::endl;

  // Get NN at x
  std::vector<double> res = nn.Evaluate(x);
  std::vector<double> res_copy = nn_copy.Evaluate(x);

  // Check size 
  if (res.size() != res_copy.size())
    nnad::Error("Vector do not have the same size");

  // Compare results
  std::cout << std::scientific;
  std::cout << "id\t NN.\t NN copy." << std::endl;
  for (int id = 0; id < (int) res.size(); id++)
    std::cout << id << "\t" << res[id] << "\t" << res_copy[id] << std::endl;
  std::cout << "\n";

  // Make results different
  std::vector<double> parameters = nn.GetParameters();
  parameters[2] = 0.5;
  nn_copy.SetParameters(parameters);
  res_copy = nn_copy.Evaluate(x);

  // Compare results
  std::cout << "Are they different?" << std::endl;
  std::cout << std::scientific;
  std::cout << "id\t NN.\t NN copy." << std::endl;
  for (int id = 0; id < (int) res.size(); id++)
    std::cout << id << "\t" << res[id] << "\t" << res_copy[id] << std::endl;
  std::cout << "\n";


  return 0;
}
