from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS={"nonstructural":"#2f6f8f","curvature":"#c4513e","case":"#246b45"}

def style():
    mpl.rcParams.update({"font.family":"DejaVu Serif","font.size":8,"axes.labelsize":9,"axes.titlesize":9,"legend.fontsize":7,"savefig.dpi":350,"axes.linewidth":.75})

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); style()
    c=pd.read_csv(a.input/"case_controlled.csv"); l=pd.read_csv(a.input/"case_lanl.csv")
    fig,ax=plt.subplots(2,2,figsize=(7.15,5.15))
    for beta,color in [(0,"#4c78a8"),(1,"#e28e28"),(2,"#b73c45")]:
        z=c[c.beta==beta]
        ax[0,0].hist(z.case_mrr,bins=np.linspace(0,1,19),histtype="step",lw=1.6,color=color,label=rf"$\beta={beta}$",density=True)
    ax[0,0].set(xlabel="CASE reciprocal rank",ylabel="density",title="(a) Controlled rank spectrum");ax[0,0].legend(frameon=False);ax[0,0].grid(alpha=.18)
    for beta,color in [(0,"#4c78a8"),(1,"#e28e28"),(2,"#b73c45")]:
        z=c[c.beta==beta].assign(selected=(c[c.beta==beta].case_action=="curvature").astype(float)).groupby("budget").selected.agg(["mean","std","count"]); se=z["std"].fillna(0)/np.sqrt(z["count"]); x=z.index*100
        ax[0,1].plot(x,z["mean"],"o-",lw=1.7,color=color,label=rf"$\beta={beta}$");ax[0,1].fill_between(x,z["mean"]-1.96*se,z["mean"]+1.96*se,color=color,alpha=.14)
    ax[0,1].set(xlabel="telemetry budget (percent)",ylabel="structural-action rate",title="(b) Gate action spectrum",ylim=(-.05,1.05));ax[0,1].legend(frameon=False);ax[0,1].grid(alpha=.18)
    g=c.groupby(["beta","m"]).apply(lambda x:(x.case_mrr-x.random).mean()).unstack(); im=ax[1,0].imshow(g,aspect="auto",cmap="RdYlGn",vmin=-.2,vmax=.7)
    ax[1,0].set(xticks=range(len(g.columns)),xticklabels=[f"m={i}" for i in g.columns],yticks=range(len(g.index)),yticklabels=[rf"$\beta={i}$" for i in g.index],title="(c) Selective-gain phase diagram");
    for i in range(g.shape[0]):
        for j in range(g.shape[1]): ax[1,0].text(j,i,f"{g.iloc[i,j]:.2f}",ha="center",va="center",fontsize=7)
    fig.colorbar(im,ax=ax[1,0],label="mean CASE minus nonstructural MRR")
    d=l.case_mrr-l.entropy; ax[1,1].hist(d,bins=23,color="#9ca5ac",alpha=.72,density=True);ax[1,1].axvline(0,color="black",lw=.8);ax[1,1].axvline(d.mean(),color="#246b45",lw=1.6,label=f"mean={d.mean():.2e}")
    ax[1,1].set(xlabel="CASE minus temporal MRR",ylabel="density",title="(d) LANL residual-gain distribution");ax[1,1].legend(frameon=False);ax[1,1].grid(alpha=.18)
    fig.tight_layout();fig.savefig(a.output/"figure_case_spectra.png",bbox_inches="tight");plt.close(fig)
    # Two manuscript-ready full-width data tables.
    s=c.groupby(["beta","m","budget"]).agg(case_mrr=("case_mrr","mean"),curvature_mrr=("curvature","mean"),nonstructural_mrr=("random","mean"),structural_action=("case_action",lambda x:float((x=="curvature").mean())),fisher=("information","mean"),beta_hat=("beta_hat","mean")).reset_index();s.to_csv(a.output/"table_case_controlled_grid.csv",index=False)
    q=l.groupby("budget").agg(case_mrr=("case_mrr","mean"),curvature_mrr=("curvature","mean"),temporal_mrr=("entropy","mean"),structural_action=("case_action",lambda x:float((x=="curvature").mean())),reliability=("reliability","mean"),fisher=("information","mean")).reset_index();q.to_csv(a.output/"table_case_lanl_grid.csv",index=False)
if __name__=="__main__": main()
