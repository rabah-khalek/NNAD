import numpy as nump
import matplotlib.pyplot as py
import sys,os
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Times']})
rc('text', usetex=True)


txt = open("test.dat", "r")
contents = txt.readlines()

x = []
pred = []
data = []
truth = []
err = []
for i, line in enumerate(contents):

    x.append(float(line.split()[0]))
    pred.append(float(line.split()[1]))
    data.append(float(line.split()[2]))
    truth.append(float(line.split()[3]))
    err.append(float(line.split()[4]))

ax = py.subplot(111)

ax.plot(x, pred, ls='-', color="b", label="Predictions", lw=2)
ax.plot(x, truth, ls='--', color="r", label="Truth ($P_{10}$)", lw=2)
ax.scatter(x, data, color="k", lw=0.5)
ax.errorbar(x, data, yerr=err, fmt='k.', label="Data")

#ax.text(0.72, 0.78, A_ref[A], fontsize=40, transform=ax.transAxes)
ax.set_xlabel(r'x', fontsize=12)
ax.set_ylabel(r'f(x)', fontsize=12)
ax.legend(loc='best')

py.savefig('test.pdf')
py.cla()
py.clf()
