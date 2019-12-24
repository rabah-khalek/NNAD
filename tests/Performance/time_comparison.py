import numpy as nump
import matplotlib.pyplot as py
import sys,os
from matplotlib import rc
#rc('font', **{'family': 'sans-serif', 'sans-serif': ['Times']})
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
            output[strategy][np][Seed]["hid1"]=int(np_line.split()[2])
            output[strategy][np][Seed]["chi2"]=float(np_line.split()[3])
            output[strategy][np][Seed]["time"]=float(np_line.split()[4])

ax = py.subplot(111)
ax2 = ax.twiny()

for strategy in strategies:
    strategy_path = "output/"+strategy
    if not os.path.isdir(strategy_path):
        continue

    nps = []
    neurones = []
    avg_times = []
    med_times = []

    std_times = []
    up95_times = []
    low95_times = []

    up68_times = []
    low68_times = []

    for i, np in enumerate(sorted(output[strategy].keys())):
        times = []

        for Seed in output[strategy][np].keys():
            if output[strategy][np][Seed]["chi2"] < 1.2:
                times.append(output[strategy][np][Seed]["time"])
        if not times:
            continue

        avg_time = nump.mean(times)
        med_time = nump.median(times, axis=0)
        std_time = nump.std(times)

        if np==22 or i == len(output[strategy].keys())-1:
            xp = nump.linspace(0,np,100)
            yp = nump.zeros(100)+avg_time
            ax.plot(xp, yp, ls='--', color=colors[strategy], lw=1, alpha=0.5)

        up95_time = nump.nanpercentile(times, 95., axis=0)
        low95_time = nump.nanpercentile(times, 5., axis=0)

        up68_time = nump.nanpercentile(times,  68., axis=0)
        low68_time = nump.nanpercentile(times, 32., axis=0)

        nps.append(np)
        neurones.append(output[strategy][np][Seed]["hid1"])
        avg_times.append(avg_time)
        med_times.append(med_time)

        std_times.append(std_time)
        up95_times.append(up95_time)
        low95_times.append(low95_time)

        up68_times.append(up68_time)
        low68_times.append(low68_time)

    ax.plot(nps, avg_times, ls='-', color=colors[strategy],label=strategy, lw=3)
    #ax.plot(nps, med_times, ls='-', color=colors[strategy],label=strategy, lw=3)

    ax.fill_between(nps, list(nump.array(avg_times)+nump.array(std_times)), list(nump.array(avg_times)-nump.array(std_times)),facecolor=colors[strategy], alpha=0.25, edgecolor=None, lw=1)
    #ax.fill_between(nps, up68_times, low68_times, facecolor=colors[strategy], alpha=0.25, edgecolor=None, lw=1)

#ax.text(0.72, 0.78, A_ref[A], fontsize=40, transform=ax.transAxes)
ax.set_xlabel('Parameters', fontsize=12)
ax.set_ylabel(r'Time $\left(\rm \frac{s}{1k\,\,iterations}\right)$', fontsize=12)
ax.set_xlim(left=10)
ax.set_ylim(bottom=1, top=400)

xticks = []
new_xticks = []


for i,np in enumerate(nps):
    if not i%3:
        xticks.append(np)
        new_xticks.append(neurones[i])

ax.set_xticks(xticks)
ax.tick_params(which='both', direction='in', labelsize=12)
#ax.set_xticklabels(nps, rotation=90)

ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(xticks)
ax2.set_xticklabels(new_xticks)
ax2.tick_params(which='both', direction='in', labelsize=12)
ax2.set_xlabel(r"Middle neurones", fontsize=12)

ax.set_yscale('log')

ax.legend(loc='best')

py.savefig('time_comparison.pdf')
py.cla()
py.clf()
