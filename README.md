# NNAD
Neural Network library with Analytical Derivatives.

## Prerequisites (to run tests)
- cmake
- pkg-config
- ceres-solver
- glog
- gflags
- eigen3
- yaml-cpp

## Installation
```
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=SOME_PATH
make
```

## Conda Installation
```
conda create -n nnad
conda install gxx_linux-64 (see https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html)
cd NNAD
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make && make install
```

Code documentation generated with Doxygen can be found [here](https://vbertone.github.io/NNAD/html/index.html).

## About the library
NNAD provides a Neural network library with analytical derivatives with a simple example script where the library is interfaced with ceres-solver to fit a sin(x) in `tests/main.cc`.  
More elaborate examples, like fitting convoluted functions can be found in `https://github.com/rabah-khalek/NNAD-Interface`.

## Time Performance
### Linear
![](https://github.com/rabah-khalek/NNAD/blob/master/plots/time_linear.png)
### Logarithmic
![](https://github.com/rabah-khalek/NNAD/blob/master/plots/time_log.png)

## Seeds Dependency
A table is needed here to compare He-et-al parameter initialization versus others.

## Output fitted function
Fitting sin(x) with n=100 linear steps in x:
![](https://github.com/rabah-khalek/NNAD/blob/master/plots/output.png)
