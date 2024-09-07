//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//

#include "NNAD/FeedForwardNN.h"
#include "NNAD/Utilities.h"

int main()
{
  // Define architecture
  const std::vector<int> arch{2, 10, 1};

  // Initialise NN
  nnad::FeedForwardNN<double> nn{arch, 0, nnad::OutputFunction::QUADRATIC, true};

  // Set exponential link function
  nn.SetLinkFunctions(nnad::Exp<double>, nnad::dExp<double>, {false, true});

  // Value of x
  const std::vector<double> xv = {0.001, 0.01, 0.1, 0.5, 1, 5, 10, 20};

  // Tabulation in b
  const int nb = 100;
  const double bmin = 0;
  const double bmax = 10;
  const double bstp = ( bmax - bmin ) / ( nb - 1 );

  std::cout << std::scientific;
  for (double b = bmin; b <= 1.00001 * bmax; b += bstp)
    {
      std::cout << b << "\t";
      for (double x : xv)
        {
          const double fb = nn.Evaluate({x, b})[0];
          const double f0 = nn.Evaluate({x, bmin})[0];
          const double f8 = nn.Evaluate({x, bmax})[0];
          const double fNP = ( fb - f8 ) / ( f0 - f8 );
          std::cout << fNP << "\t";
        }
      std::cout << std::endl;
    }
  return 0;
}
