#pragma once

#include "Utilities.h"
#include <random>

namespace nnad
{
  // Available distributions at initialisation
  enum InitDistribution: int {UNIFORM, GAUSSIAN};

  /**
   * @brief The "Distribution" class defines the distribution
   * interface. This serves as base class for an abstract
   * factory pattern.
   */
  template <typename T>
  class Distribution
  {
  public:
    /**
     * @brief The "Distribution" constructor
     * @param RandomSeed: Is the seed for the random number generator
    */
    Distribution() {};
    virtual ~Distribution() {};

    /**
     * @brief Function that returns the random number
     * drawn from the distribution resolved at initialisation.
     */
    T Generate() {return _distribution();}

  protected:
    std::function<T(void)>        _distribution;
    std::default_random_engine    _g;
    int                           _seed;
  };


  //_____________________________________________________
  /**
   * @brief The "UniformDistribution" inherits from the
   * "Distribution" class. This class makes use of the
   * std library to initialises the uniform distribution.
   */
  template <typename T>
  class UniformDistribution : public Distribution<T>
  {
  public:
    /**
     * @brief The default "UniformDistribution" constructor.
     * @param min: Is the minimum value of the uniform distribution.
     * @param max: Is the maximum value of the unform distribution.
    */
    UniformDistribution(std::vector<T> Params = {}, int const& RandomSeed = -1):
      _min((Params.empty()) ? T(-1) : Params[0]),
      _max((Params.empty()) ? T(1) : Params[1])
    {
      this->_seed = RandomSeed;
      this->_g = std::default_random_engine(this->_seed);
      std::uniform_real_distribution<T> _d (_min, _max);
      this->_distribution = std::function<T(void)> ( [&] () {return this->_d(this->_g);});
    }

  private:
    std::uniform_real_distribution<T> _d;
    T _max;
    T _min;
  };

  //_____________________________________________________
  /**
   * @brief The "GaussianDistribution" inherits from the
   * "Distribution" class. This class makes use of the
   * std library to initialises the uniform distribution.
   */
  template <typename T>
  class GaussianDistribution : public Distribution<T>
  {
  public:
    /**
     * @brief The default "GaussianDistribution" constructor.
     * @param min: Is the minimum value of the uniform distribution.
     * @param max: Is the maximum value of the unform distribution.
    */
    GaussianDistribution(std::vector<T> Params = {}, int const& RandomSeed = -1):
      _mean((Params.empty()) ? T(0) : Params[0]),
      _stddev((Params.empty()) ? T(1) : Params[1])
    {
      this->_seed = RandomSeed;
      this->_g = std::default_random_engine(this->_seed);
      std::normal_distribution<T> _d (_mean, _stddev);
      this->_distribution = std::function<T(void)> ( [&] () {return this->_d(this->_g);});
    }

  private:
    std::normal_distribution<T> _d;
    T _mean;
    T _stddev;
  };


  /**
   * @brief The (abstract) factory method for the distributions.
  */
  template <typename T>
  Distribution<T>* DistributionFactory(InitDistribution const& DistName = UNIFORM,
                                       std::vector<T>          Params = {},
                                       int const&              RandomSeed = -1)
  {
    Distribution<T>* dist;
    switch (DistName)
      {
      case 0:
        dist = new UniformDistribution<T> {Params, RandomSeed};
        break;
      case 1:
        dist = new GaussianDistribution<T> {Params, RandomSeed};
        break;
      default:
        Error("The selected distribution is not implemented.");
      }
    return dist;
  }
}