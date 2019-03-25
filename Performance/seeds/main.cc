#include "FeedForwardNN.h"
#include "Timer.h"

#include "AnalyticCostFunction.h"
#include "AutoDiffCostFunction.h"
#include "NumericCostFunction.h"
#include "Globals.h"

// YAML
#include "yaml-cpp/yaml.h"

// Standard libs
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <fstream>
#include <string>

#include <iomanip>

using namespace std;

int main(int argc, char *argv[])
{
  YAML::Node InputCard;
  string InputCardName;
  int Seed;

  switch (argc)
  {
  case 2: //if Seed is not given by the user
    InputCardName = argv[1];
    InputCard = YAML::LoadFile((InputCardName).c_str());
    if (InputCard["Seed"].as<int>() == -1)
    {
      srand(time(NULL));
      Seed = rand();
      srand(Seed);
    }
    else
    {
      Seed = InputCard["Seed"].as<int>();
      srand(Seed);
    }
    break;

  case 3: //if Seed is given by the user
    InputCardName = argv[1];
    Seed = atoi(argv[2]);
    srand(Seed);  
    break;

  default:
    cerr << "Usage: " << argv[0] << " <inputcardNAME.yaml> <Seed[optional]>" << endl;
    exit(-1);
  }

  InputCard = YAML::LoadFile((InputCardName).c_str());

  // Timer
  Timer t;

  // ============================================================
  // Initialise NN to be fitted to data.
  // ============================================================
  //FeedForwardNN* nn = new FeedForwardNN{{2, 10, 3}, time(NULL)};
  vector<int> NNarchitecture = InputCard["NNarchitecture"].as<vector<int>>();
  FeedForwardNN<double> *nn = new FeedForwardNN<double>(NNarchitecture, Seed);

  // Generate pseudo data
  //vector<pair<double, double>> Data = GenerateData(Preds, 0.005, 0.01);
  // ============================================================

  // ============================================================
  // Prepare the model
  // ============================================================

  const int n = 100;
  vectdata Data;
  double xmin = 0;
  double xmax = 6.28;
  for (int i = 0; i < n; i++)
  {
    Datapoint tuple;
    double x = xmin + i * xmax / n;
    double y = sin(x);
    double sd = 1e-2 * (rand() % 100) + 0.001;
    get<0>(tuple) = x;
    get<1>(tuple) = y;
    get<2>(tuple) = sd;
    Data.push_back(tuple);
  }

  // Put initial parameters in a vector<double*> for initialising
  // the ceres solver.
  const int np = nn->GetParameterNumber();
  const vector<double> pars = nn->GetParameters();
  vector<double *> initPars(np);
  for (int ip = 0; ip < np; ip++)
    initPars[ip] = new double(pars[ip]);

  // Allocate "Problem" instance
  ceres::Problem problem;

  // Allocate a "Chi2CostFunction" instance to be fed to ceres for
  // minimisation based on the choice from InputCard.yaml
  ceres::CostFunction *analytic_chi2cf = nullptr;
  ceres::DynamicAutoDiffCostFunction<AutoDiffCostFunction, GLOBALS::kStride> *automatic_chi2cf = nullptr;
  ceres::DynamicNumericDiffCostFunction<NumericCostFunction> *numeric_chi2cf = nullptr;

  string DerivativesChoice = InputCard["Derivatives"].as<string>();
  map<string, int> StrIntMapDerivatives;
  StrIntMapDerivatives["Analytic"] = 0;
  StrIntMapDerivatives["Automatic"] = 1;
  StrIntMapDerivatives["Numeric"] = 2;

  switch (StrIntMapDerivatives[DerivativesChoice])
  {
  //Analytic
  case 0:
    analytic_chi2cf = new AnalyticCostFunction(np, Data, NNarchitecture, Seed);
    delete automatic_chi2cf;
    delete numeric_chi2cf;
    problem.AddResidualBlock(analytic_chi2cf, NULL, initPars);
    break;

  //Automatic
  case 1:
    automatic_chi2cf = new ceres::DynamicAutoDiffCostFunction<AutoDiffCostFunction, GLOBALS::kStride>(new AutoDiffCostFunction(np, Data, NNarchitecture, Seed));
    delete analytic_chi2cf;
    delete numeric_chi2cf;
    for (int i = 0; i < np; i++)
      automatic_chi2cf->AddParameterBlock(1);
    automatic_chi2cf->SetNumResiduals(Data.size());
    problem.AddResidualBlock(automatic_chi2cf, NULL, initPars);

    break;

  //Numeric
  case 2:
    numeric_chi2cf = new ceres::DynamicNumericDiffCostFunction<NumericCostFunction>(new NumericCostFunction(np, Data, NNarchitecture, Seed));
    delete analytic_chi2cf;
    delete automatic_chi2cf;
    for (int i = 0; i < np; i++)
      numeric_chi2cf->AddParameterBlock(1);
    numeric_chi2cf->SetNumResiduals(Data.size());

    problem.AddResidualBlock(numeric_chi2cf, NULL, initPars);

    break;

  //Error
  default:
    cerr << "Check if \"" << InputCard["Derivatives"].as<string>() << "\" exists in the InputCard.yaml" << endl;
    exit(-1);
  }


  // ============================================================

  // ============================================================
  // Run the solver with some options.
  // ============================================================
  // Compute initial chi2
  double chi2 = 0;
  vector<vector<double>> Predictions;
  for (int i = 0; i < n; i++)
  {
    vector<double> x;
    x.push_back(get<0>(Data[i]));
    vector<double>
        v = nn->Evaluate(x);
    Predictions.push_back(v);
    //cout << "Predictions[id][0] = " << Predictions[i][0]<<endl;
    //cout << "get<0>(Data[id]) = " << get<0>(Data[i]) << endl;
    //cout << "get<1>(Data[id]) = " << get<1>(Data[i]) << endl;
    //cout << "get<2>(Data[id]) = " << get<2>(Data[i]) << endl;
  }
  //exit(1);
  for (int id = 0; id < n; id++)
    chi2 += pow((Predictions[id][0] - get<1>(Data[id])) / get<2>(Data[id]), 2);
  chi2 /= n;
  cout << "Initial chi2 = " << chi2 << endl;
  cout << "\n";

  ceres::Solver::Options options;
  options.max_num_iterations = 1000;
  options.minimizer_progress_to_stdout = true;
  options.function_tolerance = 1e-10;
  options.parameter_tolerance = 1e-10;
  options.gradient_tolerance = 1e-10;
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);
  cout << summary.FullReport() << "\n";

  // Compute final chi2
  chi2 = 0;
  vector<double> final_pars;
  for (int i = 0; i < np; i++)
    final_pars.push_back(initPars[i][0]);
  nn->SetParameters(final_pars);

  for (int i = 0; i < n; i++)
  {
    vector<double> x;
    x.push_back(get<0>(Data[i]));
    vector<double>
        v = nn->Evaluate(x);
    Predictions.at(i) = v;
  }
  for (int id = 0; id < n; id++)
    chi2 += pow((Predictions[id][0] - get<1>(Data[id])) / get<2>(Data[id]), 2);
  chi2 /= n;

  ofstream fseed("seeds.dat", ios::out | ios::app);
  fseed<<Seed<<" \t "<<chi2<<endl;
  cout << "Final chi2 = " << chi2 << endl;
  cout << "\n";
  cout << "Derivatives Choice was: " << DerivativesChoice<<endl;
  t.stop();
  // ============================================================

  delete nn;
  return 0;
}