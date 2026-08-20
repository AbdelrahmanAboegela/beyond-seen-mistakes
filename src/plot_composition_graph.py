import json,math
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from alexgym_data import load_exercise,CRITERIA

DATA='data'; OUT=Path('paper/figures/composition_graph.png')
protocol=json.load(open('configs/final_protocol.json'))
fig,axs=plt.subplots(1,3,figsize=(7.2,2.2))
for ax,(ex,criteria) in zip(axs,CRITERIA.items()):
    X,Y,co,g,df=load_exercise(DATA,ex,T=16)
    u,c=np.unique(co,return_counts=True); bits=np.array([[int(ch) for ch in s] for s in u])
    if bits.shape[1]>=2: xy=PCA(n_components=2,random_state=0).fit_transform(bits)
    else: xy=np.c_[bits[:,0],np.zeros(len(bits))]
    # Hamming-1 edges
    for i in range(len(u)):
        for j in range(i+1,len(u)):
            if np.sum(bits[i]!=bits[j])==1:
                ax.plot([xy[i,0],xy[j,0]],[xy[i,1],xy[j,1]],color='0.82',lw=.6,zorder=1)
    eligible=set(protocol['default_targets'][ex])
    for i,s in enumerate(u):
        isel=s in eligible
        ax.scatter(xy[i,0],xy[i,1],s=10+2.4*c[i],facecolors='0.15' if isel else 'white',edgecolors='0.2',linewidths=.6,zorder=3)
    ax.set_title(f'{ex.capitalize()} ({len(u)} comps.)',fontsize=8)
    ax.set_xticks([]);ax.set_yticks([]);ax.spines[:].set_visible(False)
fig.text(.5,.01,'Node area = diagnosis frequency; filled nodes are the 12 final eligible LOCO targets; edges connect Hamming-distance-one diagnoses.',ha='center',fontsize=6.5)
fig.tight_layout(rect=[0,.08,1,1],w_pad=1.0)
OUT.parent.mkdir(parents=True,exist_ok=True);fig.savefig(OUT,dpi=300,bbox_inches='tight');print(OUT)
