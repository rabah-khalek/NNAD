//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//

#include "NNAD/FeedForwardNN.h"

#include <iostream>
#include <chrono>

int main()
{
  // Define architecture
  const std::vector<int> arch{3, 5, 5, 3};

  // Initialise NN
  const nnad::FeedForwardNN<double> nn{arch, 0, nnad::OutputFunction::ACTIVATION, true};

  // Input vector
  std::vector<double> x{0.1, 2.3, 4.5};

  // Number of calls
  const int ncalls = 500000;

  // Start timer
  auto start{std::chrono::steady_clock::now()};

  // Get NN at x
  for (int i = 0; i < ncalls; i++)
    nn.Evaluate(x);

  // Stop timer
  std::chrono::duration<double> elapsed_seconds{std::chrono::steady_clock::now() - start};

  // Report time
  std::cout << "Time elepsed taken for " << ncalls << " evaluations of the NN: " << elapsed_seconds.count() << " seconds" << std::endl;

  // Start timer
  start = std::chrono::steady_clock::now();

  // Compare NN derivatives at x
  for (int i = 0; i < ncalls; i++)
    nn.Derive(x);

  // Stop timer
  elapsed_seconds = std::chrono::steady_clock::now() - start;

  // Report time
  std::cout << "Time elepsed taken for " << ncalls << " evaluations of the NN and its derivatives: " << elapsed_seconds.count() << " seconds\n" << std::endl;

  return 0;
}
