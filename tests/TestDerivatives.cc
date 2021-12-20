//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//

#include <iostream>

#include "NNAD/FeedForwardNN.h"

int main()
{
  // Define architecture
  const std::vector<int> arch{1, 1};

  // Initialise NN
  const nnad::FeedForwardNN<double> nn{arch, 0, nnad::OutputFunction::QUADRATIC, true};

  // Linear output NN needed for the numerical derivative
  const nnad::FeedForwardNN<double> nnl{arch, 0, nnad::OutputFunction::LINEAR, false};

  // Input vector
  std::vector<double> x{0.2};
  std::cout << "Input: x = { ";
  for (auto const& e : x)
    std::cout << e << " ";
  std::cout << "}\n" << std::endl;

  // Get NN at x
  std::vector<double> ders = nn.Evaluate(x);
  const std::vector<double> lin = nnl.Evaluate(x);
  const std::vector<double> anders = nn.Derive(x);

  // Compute the derivatives numerically as incremental ratios
  const double eps = 1e-5;
  const std::vector<double> pars = nn.GetParameters();
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
    std::cout<<"ders.size()="<<ders.size()<<std::endl;
    for (int io = 0; io < arch.back(); io++)
      if (nn.OutputFunctionType() == nnad::OutputFunction::LINEAR)
        ders.push_back(( vp[io] - vm[io] ) / 2 / eps / pars[ip]);
      else if (nn.OutputFunctionType() == nnad::OutputFunction::QUADRATIC)
        ders.push_back(2 * lin[io] * ( vp[io] - vm[io] ) / 2 / eps / pars[ip]);
      else if (nn.OutputFunctionType() == nnad::OutputFunction::ACTIVATION)
        ders.push_back(nn.GetDerActivationFunction()(lin[io]) * ( vp[io] - vm[io] ) / 2 / eps / pars[ip]);
  }
  
    const int nd = (int)anders.size();
    std::cout << std::scientific;
    std::cout << "id\t num. ders.\t an. ders.\t   ratio" << std::endl;
    for (int id = 0; id < nd; id++)
      std::cout << id << "\t" << ders[id] << "\t" << anders[id] << "\t" << (anders[id] == 0 ? 0 : ders[id] / anders[id]) << std::endl;
    std::cout << "\n";

    std::vector<double> xp;
    std::vector<double> xm;

    xp.push_back(x[0] + eps);
    xm.push_back(x[0]-eps);

    const std::vector<double> vp_in = nn.Evaluate(xp);
    const std::vector<double> vm_in = nn.Evaluate(xm);

    double num_der = 2 * lin[0] * (vp_in[0] - vm_in[0]) / 2 / eps / x[0];
    double ana_der = anders.at(np+1);
    std::cout<< "num_der=" << num_der << std::endl;
    std::cout << "ana_der=" << ana_der << std::endl;

    return 0;
}
