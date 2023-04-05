#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import seaborn as sns
import matplotlib.transforms as mtransforms

sns.set_theme(context = "notebook",style="ticks", palette='bright')
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["figure.autolayout"] = False
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["font.size"] = 20
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["lines.markersize"] = 0
plt.rcParams["legend.fontsize"] = 20
plt.rcParams['grid.color'] = "#949292"
plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['font.family'] = "sans-serif"
legend_properties = {'weight':'bold'}

fig = plt.figure(constrained_layout=False,figsize=(8,5))
gs= GridSpec(1, 1, figure=fig, wspace=0.1, hspace = 0.3)
ax0 = plt.subplot(gs.new_subplotspec((0, 0)))

axList = [[ax0]]
ax0.set_xlabel('Episodes (e)')

ax0.set_ylabel('Convergence (ΔQ)')
labelList = ['(a)']

episodeNum = int(sys.argv[1])
convGap = int(sys.argv[2])
envType = sys.argv[3].lower()


x3 = list(np.linspace(10,episodeNum+10,int(episodeNum/convGap), endpoint=False))

## Make sure path exists for this one line of code #################################
df = pd.read_csv("./ProcessedOutput/ProcessedConvergence.csv", delimiter=',')
################################################################################

#### Edit these lines according to your experiments and Graph generation preferences #################
frameworkName = ['PeLPA']
AttackPercentage=['0','20', '40']
attackerNo = ['No-Attack (0%)', 'Attack (20%)', 'Attack (40%)']
lineStyle = ['s-','v-','o-','^-.','*--','s-']
color = ['red','cyan','black','magenta','blue','green']
markerFace = ['none','cyan','none','magenta','blue','green']
markerEdge = ["black","black","black","black","black","black"]
legend = ['PeLPA']
################################################################################

### Convergence #####
for i in range(len(AttackPercentage)):
    for j in range(len(frameworkName)):
        print(str(frameworkName[j])+str(AttackPercentage[i]))
        axList[j][0].plot(x3, df[str(frameworkName[j])+str(AttackPercentage[i])],
                          lineStyle[i], color = color[i], markerfacecolor=markerFace[i],
		                              markeredgecolor=markerEdge[i], label = attackerNo[i])
        axList[j][0].set_title(str(envType), loc="right" )
        axList[j][0].legend(loc ='upper right',# bbox_to_anchor=(0.8, 0.5), #0.25,-1,
		          ncol=1, fancybox=True, shadow=True,
		           borderpad=0.2,labelspacing=0.2, handlelength=0.9,columnspacing=0.8,handletextpad=0.3)
        axList[j][0].tick_params(axis='x')
        axList[j][0].tick_params(axis='y')
        trans = mtransforms.ScaledTranslation(-1/72, 1/72, fig.dpi_scale_trans)
        axList[j][0].text(0.5, -0.35, labelList[j], transform=axList[j][0].transAxes + trans)
        
        axList[j][0].ticklabel_format(axis='y', style='sci', scilimits=(0,1))
        axList[j][0].yaxis.set_minor_formatter(mticker.NullFormatter())
        axList[j][0].set_title(str(envType), loc="right" )
        




## Make sure path exists for this one line of code #################################
plt.savefig('./Convergence.pdf',bbox_inches='tight') 
################################################################################

plt.show()







