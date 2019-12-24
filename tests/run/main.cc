#include "Timer.h"
#include "AnalyticCostFunction.h"
#include "AutoDiffCostFunction.h"
#include "NumericCostFunction.h"
#include "Globals.h"

// NNAD
#include <NNAD/FeedForwardNN.h>

// YAML
#include "yaml-cpp/yaml.h"

// CERES
#include "ceres/ceres.h"

// Standard libs
#include <iostream>
#include <math.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <random>

#include "gsl/gsl_sf_legendre.h"


using namespace std;

// n = 0
double P0(double x) { return 1.;}
double P1(double x) { return x;}
double P2(double x) { return ((3 * x * x) - 1)*1. / 2;}
double P3(double x) { return 0.5 * (5 * std::pow(x, 3) - 3 * x); }
double P4(double x) { return 0.125 * (35 * std::pow(x, 4) - 30 * x * x + 3); }
double P5(double x) { return (1. / 8) * (63 * pow(x, 5) - 70 * pow(x, 3) + 15 * x); }
double P6(double x) { return (1. / 16) * (231 * pow(x, 6) - 315 * pow(x, 4) + 105 * pow(x, 2) - 5); }
double P10(double x) { return (1. / 256) * (46189 * pow(x, 10) - 109395 * pow(x, 8) + 90090 * pow(x, 6) - 30030 * pow(x, 4) + 3465 * pow(x, 2) - 63); }
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
    }
    else
      Seed = InputCard["Seed"].as<int>();
    break;

  case 3: //if Seed is given by the user
    InputCardName = argv[1];
    Seed = atoi(argv[2]);
    break;

  default:
    cerr << "Usage: " << argv[0] << " <inputcardNAME.yaml> <Seed[optional]>" << endl;
    exit(-1);
  }

  InputCard = YAML::LoadFile((InputCardName).c_str());

  // ============================================================
  // Initialise NN to be fitted to data.
  // ============================================================
  vector<int> NNarchitecture = InputCard["NNarchitecture"].as<vector<int>>();
  nnad::FeedForwardNN<double> *nn = new nnad::FeedForwardNN<double>(NNarchitecture, Seed, true);

  // Generate pseudo data
  //vector<pair<double, double>> Data = GenerateData(Preds, 0.005, 0.01);
  // ============================================================

  // ============================================================
  // Prepare the model
  // ============================================================

  int n = InputCard["Ndata"].as<int>();
  vectdata Data;
  double xmin = InputCard["xmin"].as<double>();
  double xmax = InputCard["xmax"].as<double>();//0.4*3.14;
  double yshift = InputCard["yshift"].as<double>();

  //---- noise
  double noise_mean = InputCard["noise_mean"].as<double>();
  double noise_sd = InputCard["noise_sd"].as<double>();
  //std::random_device rd{};
  //std::mt19937 gen{InputCard["Seed"].as<int>()};
  std::default_random_engine gen(Seed);
  std::normal_distribution<> noise_gen{noise_mean, noise_sd};
  //----

  std::vector<double> truth;
  for (int i = 0; i < n; i++)
  {
    Datapoint tuple;
    double x = xmin + i * (xmax-xmin) / n;
    double y = P10(x) + yshift; //pow(sin(x),2)+1;//sd;

    truth.push_back(y);

    //---- noise
    double noise = 0;
    while(! noise)
      if(y!=0)
        noise = noise_gen(gen)*y;
      else
        noise = noise_gen(gen);
    //

    //double sd = (0.1e-2 * (rand() % 100)); //[0 to 10%]
    //while(!sd)
    //  sd = (0.1e-2 * (rand() % 100)); 
    //sd*= y;

    get<0>(tuple) = x;
    get<1>(tuple) = y+noise;
    get<2>(tuple) = std::abs(noise);
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
  options.max_num_iterations = InputCard["max_num_iterations"].as<int>();
  options.minimizer_progress_to_stdout = true;
  options.function_tolerance = InputCard["function_tolerance"].as<double>();
  options.parameter_tolerance = 1e-20;
  options.gradient_tolerance = 1e-20;
  ceres::Solver::Summary summary;
  // Timer
  Timer t;
  Solve(options, &problem, &summary);
  double duration = t.stop();
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
  ofstream test("test.dat");
  for (int id = 0; id < n; id++)
  {
    chi2 += pow((Predictions[id][0] - get<1>(Data[id])) / get<2>(Data[id]), 2);
    test << get<0>(Data[id]) << " " << Predictions[id][0] << " " << get<1>(Data[id]) << " " << truth[id] << " " << get<2>(Data[id]) << endl;
  }
  chi2 /= n;
  cout << "Final chi2 = " << chi2 << endl;
  cout << "Number of parameters = "<< np <<endl;
  cout << "\n";
  cout << "Derivatives Choice was: " << DerivativesChoice<<endl;

  // ============================================================

  //! testing
  std::vector<double> hidden;
  for (int i = 1; i < InputCard["NNarchitecture"].as<vector<double>>().size()-1;i++)
  {
    hidden.push_back(InputCard["NNarchitecture"].as<vector<double>>()[i]);
  }
  bool exist = false;
  string output_name = "output/" + DerivativesChoice+"/" + to_string(np)+"_";
  for (int i = 0; i < hidden.size(); i++)
    output_name += "_"+to_string(int(hidden.at(i)));
  output_name += ".dat";
  ifstream f(output_name.c_str());
  if(f.good())
    exist=true;

  ofstream out((output_name).c_str(), ios::out | ios::app);

  if(!exist)
  {
  out << "Seed "<<" "<<"np"<<" ";
  for (int i=0;i<hidden.size();i++)
    out << "arch"+to_string(i+1)<< " ";
  out << "chi2/dat duration" << endl;
  }
  out << Seed <<" "<<np<<" ";
  for (int i=0;i<hidden.size();i++)
    out << hidden[i] << " ";
  out << chi2 << " " << duration << endl;

      //! testing

      delete nn;
  return 0;
}
