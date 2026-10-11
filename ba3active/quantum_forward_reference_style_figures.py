"""Publication-grade reference-style figures drawn only from quantum-routing results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress


QMODEL = "quantum_moe_selected"
ORDER = ("complete", "random50", "hub50", "auth_only", "dns_only", "history_random50")
COLORS = {"complete":"#15568b", "random50":"#c94046", "hub50":"#d18b21", "auth_only":"#118b7a", "dns_only":"#72549b", "history_random50":"#50585f"}
LABELS = {"complete":"Complete", "random50":"Random query loss", "hub50":"Hub query loss", "auth_only":"Authentication only", "dns_only":"Name resolution only", "history_random50":"Random history loss"}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2, "axes.labelsize": 8.7,
        "axes.titlesize": 9.2, "axes.linewidth": .72, "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4, "legend.fontsize": 7.3, "figure.dpi": 180,
        "savefig.dpi": 420, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel(ax, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#d8dddf", linewidth=.5, alpha=.86)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def qrows(results: pd.DataFrame) -> pd.DataFrame:
    return results[results.model.eq(QMODEL)].copy()


def joined(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["fold", "seed", "episode", "timestamp", "condition"]
    q = qrows(results)[keys + ["alpha", "temporal_entropy", "graph_entropy", "top10_overlap"]]
    t = results[results.model.eq("temporal")][keys + ["rank", "mrr"]].rename(columns={"rank":"rt", "mrr":"mt"})
    g = results[results.model.eq("graph_only")][keys + ["rank", "mrr"]].rename(columns={"rank":"rg", "mrr":"mg"})
    frame = q.merge(t, on=keys).merge(g, on=keys)
    frame["regret"] = np.log1p(frame.rt) - np.log1p(frame.rg)
    frame["graph_gain"] = frame.mg - frame.mt
    return frame


def figure_one(results: pd.DataFrame, output: Path) -> None:
    frame = joined(results)
    fig = plt.figure(figsize=(7.32, 7.64), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, height_ratios=(1.07, .95, 1.16))

    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(-.2, 5.05)
    ax.text(.08, 4.65, "a  Chronological construction of a routing episode", fontsize=10, fontweight="bold")
    for y in (3.8, 2.75, 1.7, .65):
        ax.plot([.6, 15.25], [y, y], color="#adb4b8", lw=.85)
        for x in np.arange(1.25, 15.1, 1.05):
            ax.add_patch(Circle((x, y), .075, facecolor="#c4c8ca", edgecolor="none"))
    past = FancyBboxPatch((2.08, 2.12), 4.55, 1.64, boxstyle="round,pad=.02,rounding_size=.03", facecolor="#6faea6", edgecolor="#0a6865", lw=1.35, alpha=.83)
    query = FancyBboxPatch((6.62, 1.04), 4.38, 1.64, boxstyle="round,pad=.02,rounding_size=.03", facecolor="#f0af69", edgecolor="#a45918", lw=1.35, alpha=.83)
    ax.add_patch(past); ax.add_patch(query)
    for x, y in ((2.1,2.75),(2.63,3.8),(2.63,2.75),(3.45,2.75),(4.32,2.75),(5.35,3.8),(5.35,2.75),(6.15,2.75),(6.62,1.7),(7.18,1.7),(7.78,1.7),(8.42,1.7),(9.4,1.7),(10.0,1.7),(10.48,1.7)):
        ax.add_patch(Circle((x, y), .095, facecolor="#242729", edgecolor="none", zorder=3))
    ax.text(4.37, 4.02, "strictly earlier edge ledger", ha="center", color="#075d5a", fontsize=8)
    ax.plot([3.05, 5.95], [2.28, 2.28], color="#4f575b", lw=.9, ls="--")
    ax.text(4.5, 1.97, "history stress removes half of prior edges before diffusion", ha="center", color="#4f575b", fontsize=6.8)
    ax.text(8.82, .75, "next authentication destination", ha="center", color="#964a10", fontsize=8)
    ax.text(2.0, 4.35, "earlier windows", fontsize=7.5, color="#596166")
    ax.text(7.28, 4.35, "observation window", fontsize=7.5, color="#596166")
    ax.text(11.8, 4.35, "held out target bucket", fontsize=7.5, color="#596166")
    ax.add_patch(FancyArrowPatch((1.2,.13), (14.0,.13), arrowstyle="-|>", mutation_scale=13, lw=1.0, color="#3d4549"))
    ax.text(7.6, -.06, "chronological time", ha="center", va="top", fontsize=7.8)

    ax = fig.add_subplot(gs[1, :]); ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 4.1)
    ax.text(.08, 3.78, "b  Constant-size quantum authority decision", fontsize=10, fontweight="bold")
    feature_labels = ("missingness", "history retention", "source history", "temporal entropy", "graph entropy", "expert overlap")
    for i, label in enumerate(feature_labels):
        y = 3.08 - i*.53
        ax.add_patch(FancyBboxPatch((.58, y-.18), 2.42, .33, boxstyle="round,pad=.02", facecolor="#f8f8f8", edgecolor="#8d969a", lw=.6))
        ax.text(.72, y-.02, label, fontsize=7.6, va="center")
        for j in range(6):
            ax.add_patch(Rectangle((2.06+.14*j, y-.125), .085, .19, facecolor=("#168c85" if (i+j)%3 else "#ec985e"), edgecolor="none"))
    ax.add_patch(FancyArrowPatch((3.22, 2.0), (4.45, 2.0), arrowstyle="->", mutation_scale=11, lw=1.0, color="#59636a"))
    ax.text(3.83, 2.23, "episode context", ha="center", fontsize=7.5)
    for wire in range(4):
        y = 3.13-wire*.62
        ax.plot([4.55, 10.15], [y, y], color="#3e4649", lw=.75)
        for x, color, label in ((5.02,"#73b6ad","R"),(6.02,"#efb06f","Y"),(7.22,"#73b6ad","R"),(8.22,"#efb06f","Y")):
            ax.add_patch(Rectangle((x, y-.155), .38, .31, facecolor=color, edgecolor="#343a3e", lw=.55))
            ax.text(x+.19, y, label, ha="center", va="center", fontsize=6.8, fontweight="bold")
        ax.add_patch(Circle((9.35, y), .065, facecolor="#2c3134", edgecolor="none"))
    ax.text(7.35,.2, "data reuploading circuit with local and parity observables", ha="center", fontsize=7.15)
    ax.add_patch(FancyArrowPatch((10.3, 2.0), (11.18, 2.0), arrowstyle="->", mutation_scale=11, lw=1.0, color="#59636a"))
    ax.add_patch(FancyBboxPatch((11.28,.84), 3.63, 2.38, boxstyle="round,pad=.02", facecolor="#f8fafb", edgecolor="#8d969a", lw=.75))
    ax.text(13.1, 2.8, "bounded gate", ha="center", fontsize=8, fontweight="bold")
    curve_x = np.linspace(11.62, 14.38, 120)
    ax.plot(curve_x, 1.56 + .72/(1+np.exp(-3.2*(curve_x-13.02))), color="#c43c47", lw=1.7)
    ax.text(13.1, 1.08, r"$\alpha_e$ mixes temporal and graph posteriors", ha="center", fontsize=7.15)

    ax = fig.add_subplot(gs[2, 0]); panel(ax, "c", "Cross fitted authority response")
    for condition in ORDER:
        part = frame[frame.condition.eq(condition)]
        ax.scatter(part.regret, part.alpha, s=8, color=COLORS[condition], alpha=.16, edgecolor="none")
        part = part.copy(); part["bin"] = pd.qcut(part.regret, q=min(8, part.regret.nunique()), duplicates="drop")
        agg = part.groupby("bin", observed=True).agg(x=("regret","mean"), y=("alpha","mean")).reset_index(drop=True)
        ax.plot(agg.x, agg.y, marker="o", ms=2.6, lw=1.2, color=COLORS[condition])
    ax.axvline(0, ls="--", lw=.75, color="#72797d")
    ax.set_xlabel("Cross fitted log rank regret")
    ax.set_ylabel("Quantum authority")
    ax.set_ylim(0, 1.04)
    handles = [plt.Line2D([],[],color=COLORS[c],marker="o",lw=1.2,ms=3,label=LABELS[c]) for c in ORDER]
    ax.legend(handles=handles, frameon=False, loc="lower right", ncol=1, handlelength=1.1)

    ax = fig.add_subplot(gs[2, 1]); panel(ax, "d", "Authority distribution by telemetry regime")
    values = [frame[frame.condition.eq(c)].alpha.to_numpy() for c in ORDER]
    v = ax.violinplot(values, positions=np.arange(len(ORDER)), widths=.70, showmeans=True, showextrema=False)
    for body, condition in zip(v["bodies"], ORDER):
        body.set_facecolor(COLORS[condition]); body.set_edgecolor="#394145"; body.set_alpha(.52)
    v["cmeans"].set_color("#22282b"); v["cmeans"].set_linewidth(1.25)
    rng = np.random.default_rng(21)
    for i, (condition, values_i) in enumerate(zip(ORDER, values)):
        chosen = rng.choice(values_i, size=min(80, len(values_i)), replace=False)
        ax.scatter(i+rng.normal(0,.075,len(chosen)), chosen, s=4.5, color=COLORS[condition], alpha=.2, edgecolor="none")
    short_labels = {"complete":"Full", "random50":"Query", "hub50":"Hub", "auth_only":"Auth", "dns_only":"DNS", "history_random50":"History"}
    ax.set_xticks(range(len(ORDER)), [short_labels[condition] for condition in ORDER], rotation=22, ha="right")
    ax.set_ylabel("Quantum authority")
    ax.set_ylim(0, 1.05)
    save(fig, output, "figure1_protocol_reference_style")


def figure_two(results: pd.DataFrame, output: Path) -> None:
    frame = joined(results)
    fig = plt.figure(figsize=(7.32, 7.20))
    gs = GridSpec(2, 2, figure=fig, height_ratios=(1.04, 1.0), hspace=.54, wspace=.56)
    ax = fig.add_subplot(gs[0, :], projection="3d")
    ax.set_title("a  Smoothed observed authority surface", loc="left", fontweight="bold", pad=7)
    x, y, z = frame.temporal_entropy.to_numpy(), frame.graph_entropy.to_numpy(), frame.alpha.to_numpy()
    xb = np.linspace(np.quantile(x,.02), np.quantile(x,.98), 18)
    yb = np.linspace(np.quantile(y,.02), np.quantile(y,.98), 18)
    rows=[]
    for i in range(len(xb)-1):
        for j in range(len(yb)-1):
            mask=(x>=xb[i])&(x<xb[i+1])&(y>=yb[j])&(y<yb[j+1])
            if mask.sum()>=4:
                rows.append(((xb[i]+xb[i+1])/2,(yb[j]+yb[j+1])/2,z[mask].mean()))
    points=np.asarray(rows); gx, gy=np.meshgrid(np.linspace(points[:,0].min(),points[:,0].max(),45),np.linspace(points[:,1].min(),points[:,1].max(),45))
    gz=griddata(points[:,:2],points[:,2],(gx,gy),method="linear")
    nearest=griddata(points[:,:2],points[:,2],(gx,gy),method="nearest")
    gz=np.where(np.isnan(gz),nearest,gz); gz=gaussian_filter(gz,1.0)
    surface=ax.plot_surface(gx,gy,gz,cmap="RdBu",edgecolor="#f2f2f2",linewidth=.12,antialiased=True,alpha=.94)
    ax.contour(gx,gy,gz,zdir="z",offset=.48,levels=5,colors="#353b3e",linewidths=.5)
    ax.set_xlabel("Temporal entropy",labelpad=5); ax.set_ylabel("Graph entropy",labelpad=5); ax.set_zlabel("Authority",labelpad=4); ax.set_zlim(.48,1.0); ax.set_box_aspect((1.42, 1.0, .58)); ax.view_init(elev=27,azim=-129)
    cb=fig.colorbar(surface,ax=ax,shrink=.57,pad=.075); cb.set_label("Smoothed mean authority",fontsize=7.2); cb.ax.tick_params(labelsize=6.8)

    ax=fig.add_subplot(gs[1,0]); panel(ax,"b","Regret calibration across data regimes")
    for condition in ORDER:
        part=frame[frame.condition.eq(condition)].copy(); part["bin"]=pd.qcut(part.regret,q=min(10,part.regret.nunique()),duplicates="drop")
        agg=part.groupby("bin",observed=True).agg(x=("regret","mean"),y=("alpha","mean"),s=("alpha","std"),n=("alpha","size")).reset_index(drop=True)
        se=agg.s.fillna(0)/np.sqrt(agg.n)
        ax.fill_between(agg.x,agg.y-se,agg.y+se,color=COLORS[condition],alpha=.09,linewidth=0)
        ax.plot(agg.x,agg.y,color=COLORS[condition],marker="o",ms=2.5,lw=1.25)
    ax.axvline(0,color="#767d81",lw=.75,ls="--")
    ax.set_xlabel("Cross fitted log rank regret"); ax.set_ylabel("Mean authority"); ax.set_ylim(.45,1.02)

    ax=fig.add_subplot(gs[1,1]); panel(ax,"c","Authority sensitivity to graph utility")
    rows=[]
    for c in ORDER:
        part=frame[frame.condition.eq(c)]; fit=linregress(part.regret,part.alpha)
        rows.append((c,part.graph_gain.mean(),fit.slope,fit.stderr))
    for c,gain,slope,se in rows:
        ax.errorbar(gain,slope,yerr=se,fmt="o",ms=6,capsize=2.5,color=COLORS[c],markeredgecolor="#263034",markeredgewidth=.4)
        off=(5,-9) if c=="auth_only" else (5,4)
        ax.annotate(LABELS[c],(gain,slope),xytext=off,textcoords="offset points",fontsize=7.0)
    ax.axvline(0,color="#747b7f",ls="--",lw=.75); ax.axhline(0,color="#747b7f",ls="--",lw=.75)
    ax.set_xlabel("Graph MRR minus temporal MRR"); ax.set_ylabel("Authority slope per regret unit")
    fig.subplots_adjust(left=.10, right=.94, top=.95, bottom=.095)
    save(fig,output,"figure2_landscape_reference_style")


def figure_three(results: pd.DataFrame, output: Path) -> None:
    q=qrows(results); keys=["fold","seed","episode","timestamp","condition"]
    exact=q.set_index(keys)
    fig=plt.figure(figsize=(7.32,6.6),constrained_layout=True)
    gs=GridSpec(2,2,figure=fig,height_ratios=(1.08,.92))
    shots=(32,64,128,256,512,1024)
    ax=fig.add_subplot(gs[0,:]); panel(ax,"a","Finite shot convergence of the routed endpoint posterior")
    for c in ORDER:
        seed_lines=[]
        means=[]
        for shot in shots:
            work=results[(results.condition.eq(c))&(results.model.eq(f"quantum_shot_{shot}"))].set_index(keys)
            base=exact.reindex(work.index)
            drift=np.abs(work.mrr-base.mrr)
            means.append(drift.mean())
            seed_lines.append(drift.groupby(level="seed").mean().to_numpy())
        seed_lines=np.asarray(seed_lines).T
        for line in seed_lines:
            ax.plot(shots,line,color=COLORS[c],alpha=.12,lw=.65)
        ax.plot(shots,means,color=COLORS[c],marker="o",ms=3.25,lw=1.5,label=LABELS[c])
    ax.set_xscale("log",base=2); ax.set_yscale("log"); ax.set_xticks(shots,[str(v) for v in shots])
    ax.set_xlabel("Simulated measurement shots"); ax.set_ylabel("Absolute MRR deviation from exact inference")
    ax.legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(.5,1.25),columnspacing=.95,handlelength=1.4)

    ax=fig.add_subplot(gs[1,0]); panel(ax,"b","Shot based authority estimate at 256 shots")
    shot=results[results.model.eq("quantum_shot_256")].set_index(keys)
    base=exact.reindex(shot.index)
    work=shot.reset_index().copy(); work["exact_alpha"]=base.alpha.to_numpy()
    for c in ORDER:
        part=work[work.condition.eq(c)]
        ax.scatter(part.exact_alpha,part.alpha,s=7,color=COLORS[c],alpha=.22,edgecolor="none")
    ax.plot([0,1],[0,1],color="#30373a",lw=.9)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xlabel("Exact authority"); ax.set_ylabel("Finite shot authority")

    ax=fig.add_subplot(gs[1,1]); panel(ax,"c","Noise response remains localized to the gate")
    for c in ORDER:
        xs=[]; ys=[]
        for p,r in ((0.,0.),(.005,.010),(.010,.020),(.020,.030),(.050,.050)):
            work=results[(results.condition.eq(c))&(results.model.eq(f"quantum_noise_p{p:.3f}_r{r:.3f}"))].set_index(keys)
            base=exact.reindex(work.index)
            xs.append(p); ys.append(np.abs(work.alpha-base.alpha).mean())
        ax.plot(xs,ys,marker="o",ms=3,lw=1.35,color=COLORS[c])
    ax.set_xlabel("Depolarizing probability"); ax.set_ylabel("Mean absolute authority drift")
    save(fig,output,"figure3_measurement_reference_style")


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); style()
    results=pd.read_csv(args.results)
    global ORDER
    ORDER = tuple(condition for condition in ORDER if condition in set(results.condition))
    if not ORDER:
        raise ValueError("results contain no recognized evaluation conditions")
    figure_one(results,args.output); figure_two(results,args.output); figure_three(results,args.output)


if __name__ == "__main__":
    main()
