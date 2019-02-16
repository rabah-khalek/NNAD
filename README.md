# NNAGD
Neural Network library for Analytical Gradient Descent interfaced with ceres-solver.
  
## Comparison
Fitting sin(x) with n=100 linear steps in x:  
- **analytic**:   time[1.218678 s]	iterations[1000] 	chi2/ndat[7.28241e-09].  
- **automatic**:  time[1.435668 s]	iterations[1000]	chi2/ndat[7.28241e-09].  
- **numeric**:    time[7.904501 s]	iterations[1000] 	chi2/ndat[7.28893e-09].  

![](https://github.com/rabah-khalek/NNAGD/blob/master/output.png)
