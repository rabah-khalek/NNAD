import numpy as nump
import matplotlib.pyplot as py
import sys,os
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

strategies=["Analytic","Automatic","Numeric"]
colors={}
colors["Analytic"]="blue"
colors["Automatic"]="red"
colors["Numeric"]="green"

output={}

#loading
for strategy in strategies:
    output[strategy]={}
    strategy_path = "output/"+strategy
    if not os.path.isdir(strategy_path): 
        continue
    
    all_txts=[f for f in os.listdir(strategy_path) if os.path.isfile(os.path.join(strategy_path,f))]
    all_txts.sort()
    for np_str in all_txts:
        np = int(np_str.split(".dat")[0])
        output[strategy][np]={}
        np_txt = open("output/"+strategy+"/"+np_str, "r")
        np_contents = np_txt.readlines()

        for i, np_line in enumerate(np_contents):
            if i==0: continue
            Seed = float(np_line.split()[0])
            output[strategy][np][Seed] = {}
            output[strategy][np][Seed]["hid1"]=float(np_line.split()[2])
            output[strategy][np][Seed]["chi2"]=float(np_line.split()[3])
            output[strategy][np][Seed]["time"]=float(np_line.split()[4])

ax = py.subplot(111)

for strategy in strategies:
    strategy_path = "output/"+strategy
    if not os.path.isdir(strategy_path):
        continue

    nps = []
    avg_times = []
    std_times = []

    for np in sorted(output[strategy].keys()):
        times = []

        for Seed in output[strategy][np].keys():
            times.append(output[strategy][np][Seed]["time"])
        
        avg_time = nump.mean(times)
        std_time = nump.std(times)

        nps.append(np)
        avg_times.append(avg_time)
        std_times.append(std_time)

    ax.plot(nps, avg_times, ls='-', color=colors[strategy],
            label=strategy, lw=3)

    ax.fill_between(nps, list(nump.array(avg_times)+nump.array(std_times)), list(nump.array(avg_times)-nump.array(std_times)),facecolor=colors[strategy], alpha=0.25, edgecolor=None, lw=1)

#ax.text(0.72, 0.78, A_ref[A], fontsize=40, transform=ax.transAxes)
ax.set_xlabel(r'number of parameters', fontsize=12)
ax.set_ylabel(r'$\langle T \rangle_{Seeds} \pm 1\sigma$ [s]', fontsize=12)
ax.legend(loc='upper left')

py.savefig('time_comparison.pdf')
py.cla()
py.clf()
