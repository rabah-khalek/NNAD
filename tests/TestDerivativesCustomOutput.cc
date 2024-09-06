//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//

#include "NNAD/FeedForwardNN.h"
#include "NNAD/Utilities.h"

#include <iostream>

int main()
{
  // Define architecture
  const std::vector<int> arch{3, 5, 5, 3};

  // Initialise NN with custom output function
  const nnad::FeedForwardNN<double> nn{arch, 0, true, nnad::Sigmoid<double>, nnad::dSigmoid<double>,
                                       [] (double const& x) -> double{ return 1 / ( 1 + x ); },
                                       [] (double const& x) -> double{ return - 1 / pow(1 + x, 2); }};

  // Linear output NN needed for the numerical derivative
  const nnad::FeedForwardNN<double> nnl{arch, 0, nnad::OutputFunction::LINEAR, false};

  // Input vector
  std::vector<double> x{0.1, 2.3, 4.5};
  std::cout << "Input: x = { ";
  for (auto const& e : x)
    std::cout << e << " ";
  std::cout << "}\n" << std::endl;

  // Print parameters along with their label
  const std::vector<double> pars = nn.GetParameters();
  const std::map<int, std::string> strmap = nn.GetIntStrMap();
  std::cout << std::scientific;
  std::cout << "Parameters index - name - value:\n";
  for (int i = 0; i < (int) pars.size(); i++)
    std::cout << i << "\t" << strmap.at(i) << "\t\t" << pars[i] << std::endl;
  std::cout << "\n";

  // Get NN at x
  std::vector<double> ders = nn.Evaluate(x);
  const std::vector<double> lin = nnl.Evaluate(x);

  // Compute the derivatives numerically as incremental ratios
  const double eps = 1e-5;
  const int np = (int) pars.size();

  // Loop over all derivatives
  for (int ip = 0; ip < np; ip++)
    {
      // Displace ip-th parameter by "eps" up and down
      std::vector<double> parsp(pars.size());
      std::vector<double> parsm(pars.size());
      for (int jp = 0; jp < np; jp++)
        {
          parsp[jp] = (jp == ip ? pars[jp] * ( 1 + eps ) : pars[jp]);
          parsm[jp] = (jp == ip ? pars[jp] * ( 1 - eps ) : pars[jp]);
        }

      // Define NNs with the displaced parameters.
      const nnad::FeedForwardNN<double> nnp{arch, parsp};
      const nnad::FeedForwardNN<double> nnm{arch, parsm};

      // Get outputs and compute the derivative numerically
      const std::vector<double> vp = nnp.Evaluate(x);
      const std::vector<double> vm = nnm.Evaluate(x);

      for (int io = 0; io < arch.back(); io++)
        ders.push_back(nn.GetDerOutputFunction()(lin[io]) * ( vp[io] - vm[io] ) / 2 / eps / pars[ip]);
    }

  // Compare derivatives
  const std::vector<double> anders = nn.Derive(x);
  const int nd = (int) anders.size();
  std::cout << "id\t num. ders.\t an. ders.\t   ratio" << std::endl;
  for (int id = 0; id < nd; id++)
    std::cout << id << "\t" << ders[id] << "\t" << anders[id] << "\t" << ( anders[id] == 0 ? 0 : ders[id] / anders[id] ) << std::endl;
  std::cout << "\n";

  return 0;
}
