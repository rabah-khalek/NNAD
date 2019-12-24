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

nrows, ncols = 3, 1
py.figure(figsize=(ncols*5, nrows*3.5))

count=0

for strategy in strategies:

    ax = py.subplot(311+count)

    strategy_path = "output/"+strategy
    if not os.path.isdir(strategy_path):
        continue

    nps = []
    avg_chi2s = []
    med_chi2s = []
    std_chi2s = []

    up95_chi2s = []
    low95_chi2s = []
    

    for np in sorted(output[strategy].keys()):
        chi2s = []

        for Seed in output[strategy][np].keys():
            chi2s.append(output[strategy][np][Seed]["chi2"])
        
        avg_chi2 = nump.mean(chi2s)
        med_chi2 = nump.median(chi2s, axis=0)

        std_chi2 = nump.std(chi2s)

        up95_chi2 = nump.nanpercentile(chi2s, 95., axis=0)
        low95_chi2 = nump.nanpercentile(chi2s, 5., axis=0)

        nps.append(np)
        avg_chi2s.append(avg_chi2)
        med_chi2s.append(med_chi2)
        std_chi2s.append(std_chi2)

        up95_chi2s.append(up95_chi2)
        low95_chi2s.append(low95_chi2)

    #ax.set_yscale('log')

    #ax.plot(nps, avg_chi2s, ls='-', color=colors[strategy], label=strategy, lw=3)
    ax.plot(nps, med_chi2s, ls='-', color=colors[strategy], label=strategy, lw=3)

    #ax.fill_between(nps, list(nump.array(avg_chi2s)+nump.array(std_chi2s)), list(nump.array(avg_chi2s)-nump.array(std_chi2s)),facecolor=colors[strategy], alpha=0.25, edgecolor=None, lw=1)
    ax.fill_between(nps, up95_chi2s, low95_chi2s,
                    facecolor=colors[strategy], alpha=0.25, edgecolor=None, lw=1)

    ax.set_ylabel(r'$\widetilde{\chi^2}_{Seeds}\,\, [95\% CL]$', fontsize=12)
    ax.legend(loc='upper left')

    count+=1
#ax.text(0.72, 0.78, A_ref[A], fontsize=40, transform=ax.transAxes)
ax.set_xlabel(r'number of parameters', fontsize=12)



py.tight_layout()
py.savefig('chi2_comparison.pdf')
py.cla()
py.clf()
