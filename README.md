[![DOI](https://zenodo.org/badge/170917214.svg)](https://zenodo.org/badge/latestdoi/170917214)
# NNAD

`NNAD` stands for Neural Network Analytic Derivatives and is a C++ implementation of the analytic derivatives of a feed-forward neural network with arbitrary architecture with respect to its free parameters. We adopeted the back-propagation method that makes the computation of derivatives in the context of minimisation problems particularly performing.

## Installation

The `NNAD` library only relies on `cmake` for configuration and installation. This is done by following the standar procedure:
```
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=SOME_PATH
make
```

## Conda Installation

Alternatively, one can use `Conda`:
```
conda create -n nnad
conda install gxx_linux-64 (see https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html)
cd NNAD
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make && make install
```

## Documentation

A detailed documentation of the code generated with Doxygen can be found [here](https://vbertone.github.io/NNAD/html/index.html).

## Examples

A simple example of the usage of `NNAD`, where analytic derivatives are compared to a numerical evaluation, can be found in `tests/main.cc`. More elaborate examples, where `NNAD` is used in minimisation problems, are instead collected [here](https://github.com/rabah-khalek/NNAD-Interface).

## Reference

- Rabah Abdul Khalek, Valerio Bertone, *On the derivatives of feed-forward neural networks*, arXiv:2005.07039

## Contacts

- Rabah Abdul Khalek: rabah.khalek@gmail.com
- Valerio Bertone: valerio.bertone@cern.ch
