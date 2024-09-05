//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//

#include "NNAD/FeedForwardNN.h"

#include <iostream>

int main()
{
  // Define architecture
  const std::vector<int> arch{1, 5, 5, 2};

  // Define preprocessing function
  nnad::FeedForwardNN<double>::Preprocessing Preproc = [] (std::vector<double> const& in, std::vector<double> const& par) -> std::vector<double>
  {
    const double x = in[0];
    const double f1 = pow(x, par[0]) * pow(1 - x, par[1]);
    const double f2 = pow(x, par[2]) * pow(1 - x, par[3]);
    return {f1, f2, log(x) * f1, 0, log(1 - x) * f1, 0, 0, log(x) * f2, 0, log(1 - x) * f2};
  };

  // Preprocessing parameters
  const std::vector<double> parsprep{-1, 1, 2, 3};

  // Initialise NN needed for the numerical derivative
  const nnad::FeedForwardNN<double> nn{arch, 0, nnad::OutputFunction::LINEAR, true};

  // Initialise NN with preprocessing
  const nnad::FeedForwardNN<double> nnp{arch, 0, nnad::OutputFunction::QUADRATIC, true, Preproc, parsprep};

  // Input vector
  std::vector<double> x{0.001};
  std::cout << "Input: x = { ";
  for (auto const& e : x)
    std::cout << e << " ";
  std::cout << "}\n" << std::endl;

  // Get NNs at x
  const std::vector<double> nnl = nn.Evaluate(x);

  // Get preprocessing at x
  const std::vector<double> prepf = Preproc(x, parsprep);

  // Compute the derivatives numerically as incremental ratios
  const double eps = 1e-5;
  const std::vector<double> pars = nn.GetParameters();
  const int np = (int) pars.size();

  // Square and include preprocessing function
  std::vector<double> ders;
  for (int io = 0; io < arch.back(); io++)
    ders.push_back(nnp.GetOutputFunction()(nnl[io]) * prepf[io]);

  // Loop over all derivatives w.r.t the NN parameters
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
      const nnad::FeedForwardNN<double> nnpl{arch, parsp};
      const nnad::FeedForwardNN<double> nnmn{arch, parsm};

      // Get outputs and compute the derivative numerically
      const std::vector<double> vp = nnpl.Evaluate(x);
      const std::vector<double> vm = nnmn.Evaluate(x);

      for (int io = 0; io < arch.back(); io++)
        ders.push_back(nnp.GetDerOutputFunction()(nnl[io]) * ( vp[io] - vm[io] ) / 2 / eps / pars[ip] * prepf[io]);
    }

  // Now include derivatives w.r.t. the preprocessing parameters
  for (int ip = 0; ip < (int) parsprep.size(); ip++)
    {
      // Displace ip-th parameter by "eps" up and down
      std::vector<double> parsp(parsprep.size());
      std::vector<double> parsm(parsprep.size());
      for (int jp = 0; jp < parsprep.size(); jp++)
        {
          parsp[jp] = (jp == ip ? parsprep[jp] * ( 1 + eps ) : parsprep[jp]);
          parsm[jp] = (jp == ip ? parsprep[jp] * ( 1 - eps ) : parsprep[jp]);
        }

      // Define preprocessing function with the displaced parameters.
      const std::vector<double> vp = Preproc(x, parsp);
      const std::vector<double> vm = Preproc(x, parsm);

      for (int io = 0; io < arch.back(); io++)
        ders.push_back(nnp.GetOutputFunction()(nnl[io]) * ( vp[io] - vm[io] ) / 2 / eps / parsprep[ip]);
    }

  // Compare derivatives
  const std::vector<double> anders = nnp.Derive(x);
  const int nd = (int) anders.size();
  std::cout << std::scientific;
  std::cout << "id\t num. ders.\t an. ders.\t   ratio" << std::endl;
  for (int id = 0; id < nd; id++)
    std::cout << id << "\t" << ders[id] << "\t" << anders[id] << "\t" << ( anders[id] == 0 ? 0 : ders[id] / anders[id] ) << std::endl;
  std::cout << "\n";

  return 0;
}
