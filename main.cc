#include "tFeedForwardNN.h"
#include "Timer.h"
//#include "Chi2CostFunction.h"
#include "my_AutoDiffCostFunctor.h"

// Standard libs
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <fstream>
#include <string>

// ROOT
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TMultiGraph.h>
#include <TCanvas.h>
#include <TLegend.h>



int main()
{
  // Timer
  Timer t;

  // ============================================================
  // Initialise NN to be fitted to data.
  // ============================================================
  //FeedForwardNN* nn = new FeedForwardNN{{2, 10, 3}, time(NULL)};
  FeedForwardNN<double> *nn = new FeedForwardNN<double>({1, 5, 1}, time(NULL));

  // Generate pseudo data
  //std::vector<std::pair<double, double>> Data = GenerateData(Preds, 0.005, 0.01);
  // ============================================================

  // ============================================================
  // Prepare the model
  // ============================================================


  const int n=100;
  vectdata Data;
  double xmin=0;
  double xmax=6.28;
  for(int i=0;i<n;i++)
  {
    Datapoint tuple;
    double x = xmin + i * xmax / n;
    double y = sin(x);
    double sd = 1e-2 * (rand() % 100) + 0.001;
    std::get<0>(tuple) = x;
    std::get<1>(tuple) = y;
    std::get<2>(tuple) = 0.1;
    Data.push_back(tuple);
  }
  

      // Put initial parameters in a std::vector<double*> for initialising
      // the ceres solver.
      const int np = nn->GetParameterNumber();
  const std::vector<double> pars = nn->GetParameters();
  std::vector<double *> initPars(np);
  for (int ip = 0; ip < np; ip++)
    initPars[ip] = new double(pars[ip]);

  // Allocate a "Chi2CostFunction" instance to be fed to ceres for
  // minimisation.
  //ceres::CostFunction *chi2cf = new Chi2CostFunction{np, Predictions, Derivatives, Data};

ceres::DynamicAutoDiffCostFunction<my_AutoDiffCostFunctor, 4> *chi2cf = new ceres::DynamicAutoDiffCostFunction<my_AutoDiffCostFunctor, 4>( new my_AutoDiffCostFunctor(np, Data));

  //TODO: Add the numeric cost function.
  //ceres::DynamicNumericDiffCostFunction<NumericCostFunctor> *chi2cf = new ceres::DynamicNumericDiffCostFunction<NumericCostFunctor>(new NumericCostFunctor(np + npp, Predictions, expdata));

for (int i = 0; i < np; i++)
  chi2cf->AddParameterBlock(1);

chi2cf->SetNumResiduals(Data.size());

// Allocate "Problem" instance
ceres::Problem problem;
problem.AddResidualBlock(chi2cf, NULL, initPars);
// ============================================================

// ============================================================
// Run the solver with some options.
// ============================================================
// Compute initial chi2
double chi2 = 0;
std::vector<std::vector<double>> Predictions;
for (int i = 0; i < n; i++)
{
  std::vector<double> x;
  x.push_back(std::get<0>(Data[i]));
  std::vector<double>
      v = nn->Evaluate(x);
  Predictions.push_back(v);
  //std::cout << "Predictions[id][0] = " << Predictions[i][0]<<std::endl;
  //std::cout << "std::get<0>(Data[id]) = " << std::get<0>(Data[i]) << std::endl;
  //std::cout << "std::get<1>(Data[id]) = " << std::get<1>(Data[i]) << std::endl;
  //std::cout << "std::get<2>(Data[id]) = " << std::get<2>(Data[i]) << std::endl;
  }
  //exit(1);
  for (int id = 0; id < n; id++)
    chi2 += pow((Predictions[id][0] - std::get<1>(Data[id])) / std::get<2>(Data[id]), 2);
  chi2 /= n;
  std::cout << "Initial chi2 = " << chi2 << std::endl;
  std::cout << "\n";
  
  ceres::Solver::Options options;
  options.max_num_iterations = 1000;
  options.minimizer_progress_to_stdout = true;
  options.function_tolerance = 1e-10;
  options.parameter_tolerance = 1e-10;
  options.gradient_tolerance = 1e-10;
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  // Compute final chi2
  chi2 = 0;
  std::vector<double> final_pars;
  for (int i = 0; i < np; i++)
    final_pars.push_back(initPars[i][0]);
  nn->SetParameters(final_pars);

  for (int i = 0; i < n; i++)
  {
    std::vector<double> x;
    x.push_back(std::get<0>(Data[i]));
    std::vector<double>
        v = nn->Evaluate(x);
    Predictions.at(i)=v;
  }
  for (int id = 0; id < n; id++)
    chi2 += pow((Predictions[id][0] - std::get<1>(Data[id])) / std::get<2>(Data[id]), 2);
  chi2 /= n;
  std::cout << "Final chi2 = " << chi2 << std::endl;
  std::cout << "\n";
  t.stop();
  // ============================================================

  // Plots results (taken from Rabah's code).
  
  TGraphErrors *exp_graph = new TGraphErrors();
  TGraph *NN_graph = new TGraph();
  //for (int i = 0; i < (int) Data.size(); i++)
  for (int i = 0; i < n; i++)
  {
    const double x = std::get<0>(Data[i]);
    const double y = std::get<1>(Data[i]);
    const double dy = std::get<2>(Data[i]);
    const double yn = Predictions[i][0];
    exp_graph->SetPoint(i, x, y);
    exp_graph->SetPointError(i, 0, dy);
    NN_graph->SetPoint(i, x, yn);
  }
  exp_graph->SetLineColor(1);
  exp_graph->SetMarkerStyle(20);
  NN_graph->SetMarkerColor(2);
  NN_graph->SetMarkerStyle(20);

  TLegend *leg = new TLegend(0.7, 0.89, 0.89, 0.75);
  leg->SetFillColor(0);
  leg->AddEntry(exp_graph, "Exp", "lp");
  leg->AddEntry(NN_graph, "NN", "lp");
  leg->AddEntry((TObject *)0, ("#chi^{2} = " + std::to_string(chi2)).c_str(), "");

  TMultiGraph *mg = new TMultiGraph();
  TCanvas *c = new TCanvas();
  mg->Add(NN_graph, "AP");
  //mg->Add(exp_graph, "AP");
  mg->SetTitle("Title");
  mg->Draw("AP");
  leg->Draw("SAME");

  c->SaveAs("output.pdf");
  system("open output.pdf");

  delete nn;
  return 0;
}