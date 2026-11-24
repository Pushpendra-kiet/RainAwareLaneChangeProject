"""run_hf.py - zero-shot evaluation of all controllers on the high-fidelity plant (plant_hf.Plant14)."""
import json, os, time, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import rainlc as R, plant as P, plant_hf as H, run_extension as X

OUT = "results"
COL = X.COL
SEED_HF = 31337


def run_hf(factory, sc, hf, **kw):
    t = time.time()
    m = P.run_policy(factory(), sc, mismatch=hf, plant_cls=H.Plant14, **kw)
    return m, P.cost_of(m), time.time() - t


def main(extra=None):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 130})
    ctrl, _ = X.build_controllers()
    if extra: ctrl.update(extra)
    # ---------------------------------------------------------------- A. main zero-shot test (level 1) + structure-only test (level 0)
    rng = np.random.default_rng(2026)
    ri = rng.uniform(0, 1, 300); v = rng.uniform(18, 32, 300)
    sc = R.make_scenarios(ri, v, rng, mu_err_std=0.12)
    hf1 = H.make_hf_params(300, np.random.default_rng(SEED_HF), 1.0)
    hf0 = H.make_hf_params(300, np.random.default_rng(SEED_HF), 0.0)
    res1 = {n: run_hf(f, sc, hf1) for n, f in ctrl.items()}
    res0 = {n: run_hf(f, sc, hf0) for n, f in ctrl.items()}
    ref = "GA-RainAware-DR"
    for tag, res in [("hf_level1", res1), ("hf_level0", res0)]:
        rows = X.table(res, ref=ref)
        for r in rows: r["p_vs_ref"] = r.pop("p_vs_GA")
        print(f"== {tag} (ref = {ref})"); print(X.write_table(rows, f"{OUT}/hf_table_{tag}"), flush=True)
    # paired comparison Fixed-DR vs RainAware-DR
    Ja, Jb = res1["GA-Fixed-DR"][1], res1["GA-RainAware-DR"][1]
    print("paired Fixed-DR vs RainAware-DR (HF level1): mean J %.3f vs %.3f ; p=%.3g ; Fixed better in %d/300" % (Ja.mean(), Jb.mean(), wilcoxon(Ja, Jb).pvalue, (Ja < Jb).sum()), flush=True)
    # ---------------------------------------------------------------- B. rain sweep
    lv = np.linspace(0, 1, 11); sw = {n: dict(J=[], fail=[], rmse=[], util=[], T=[]) for n in ctrl}
    for i, r_ in enumerate(lv):
        rr = np.repeat(r_, 60); vv = np.repeat([20., 25., 30.], 20)
        scw = R.make_scenarios(rr, vv, np.random.default_rng(5000 + i), mu_err_std=0.10)
        hfw = H.make_hf_params(60, np.random.default_rng(900 + i), 1.0)
        for n, f in ctrl.items():
            m, J, _ = run_hf(f, scw, hfw); ok = m["fail"][0] < .5
            sw[n]["J"].append(float(J.mean())); sw[n]["fail"].append(float(100 * m["fail"].mean()))
            sw[n]["rmse"].append(float(m["rmse"][0][ok].mean()) if ok.any() else float("nan"))
            sw[n]["util"].append(float(m["util"].mean())); sw[n]["T"].append(float(m["T"].mean()))
    json.dump(dict(levels=lv.tolist(), **sw), open(f"{OUT}/hf_sweep.json", "w"), indent=1)
    # ---------------------------------------------------------------- C. parameter-spread sweep
    lvls = [0.0, 0.5, 1.0, 1.5, 2.0]; sub = np.arange(150)
    scs = {k: (v_[sub] if k != "z" else v_[:, sub, :]) for k, v_ in sc.items()}
    spJ = {n: [] for n in ctrl}; spF = {n: [] for n in ctrl}
    for l in lvls:
        hfl = H.make_hf_params(150, np.random.default_rng(SEED_HF + 1), l)
        for n, f in ctrl.items():
            m, J, _ = run_hf(f, scs, hfl); spJ[n].append(float(J.mean())); spF[n].append(float(100 * m["fail"].mean()))
    json.dump(dict(levels=lvls, J=spJ, fail=spF), open(f"{OUT}/hf_spread_sweep.json", "w"), indent=1)
    # ---------------------------------------------------------------- D. failure analysis (level 1)
    fa = {}
    bins_r = [(0, .33, "light"), (.33, .66, "moderate"), (.66, 1.01, "heavy")]; bins_v = [(18, 25, "18-25 m/s"), (25, 32.1, "25-32 m/s")]
    for n in ctrl:
        f = res1[n][0]["fail"][0] > .5
        fa[n] = {f"{rn}/{vn}": (float(100 * f[(ri >= r0) & (ri < r1) & (v >= v0) & (v < v1)].mean()), int(((ri >= r0) & (ri < r1) & (v >= v0) & (v < v1)).sum()))
                 for r0, r1, rn in bins_r for v0, v1, vn in bins_v}
    json.dump(fa, open(f"{OUT}/hf_failure_breakdown.json", "w"), indent=1)
    print("== failure rate % by rain/speed bin (HF level 1)")
    for n in ctrl:
        if any(x[0] > 0 for x in fa[n].values()): print(" ", n.ljust(16), {k: round(x[0], 1) for k, x in fa[n].items()})
    # ---------------------------------------------------------------- figures
    show = ["Hand-tuned", "GA-RainAware", "GA-RainAware-DR", "GA-Fixed-DR", "MPC", "DQN"] + [k for k in (extra or {})]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for a, (k, lab) in zip(ax, [("rmse", "Tracking RMSE [m]"), ("util", "Peak friction utilisation"), ("J", "Mean cost J"), ("fail", "Failure rate [%]")]):
        for n in show: a.plot(lv, sw[n][k], "-o", ms=3, color=COL.get(n, None), label=n)
        a.set_xlabel("rain intensity"); a.set_ylabel(lab)
    ax[2].set_yscale("log"); ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT}/hf_fig1_rain_sweep.png"); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for n in show:
        ax[0].plot(lvls, spJ[n], "-o", ms=3, color=COL.get(n, None), label=n); ax[1].plot(lvls, spF[n], "-o", ms=3, color=COL.get(n, None))
    ax[0].set_yscale("log")
    for a, t_ in zip(ax, ["Mean cost J", "Failure rate [%]"]): a.set_xlabel("HF parameter spread (0 = nominal vehicle)"); a.set_title(t_)
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT}/hf_fig2_spread.png"); plt.close()
    scT = R.make_scenarios([0.85], [27.], np.random.default_rng(3), mu_err_std=0.0)
    hfT = H.make_hf_params(1, np.random.default_rng(5), 0.0)
    fig, ax = plt.subplots(1, 5, figsize=(19, 3.5))
    for n in ["Hand-tuned", "MPC", "DQN", "GA-RainAware-DR"] + [k for k in (extra or {})][:1]:
        m = P.run_policy(ctrl[n](), scT, mismatch=hfT, plant_cls=H.Plant14, traj=True); tr = m["traj"]
        ax[0].plot(tr["t"], tr["y"][:, 0], color=COL.get(n), label=n); ax[1].plot(tr["t"], np.degrees(tr["delta"][:, 0]), color=COL.get(n))
        ax[2].plot(tr["t"], tr["ay"][:, 0] / (m["mu"][0, 0] * R.G), color=COL.get(n)); ax[3].plot(tr["t"], np.degrees(tr["phi"][:, 0]), color=COL.get(n))
        ax[4].plot(tr["t"], tr["vx"][:, 0], color=COL.get(n))
    for a, t_ in zip(ax, ["Lateral position y [m]", "Steering angle [deg]", "ay / (μ g)", "Roll angle [deg]", "Forward speed [m/s]"]): a.set_title(t_); a.set_xlabel("time [s]"); a.set_xlim(0, 10)
    ax[0].legend(fontsize=7); fig.suptitle("High-fidelity plant: heavy rain (0.85), v0 = 27 m/s, nominal parameters", y=1.03)
    plt.tight_layout(); plt.savefig(f"{OUT}/hf_fig3_heavy_rain.png", bbox_inches="tight"); plt.close()
    print("all done", flush=True)
    return ctrl


if __name__ == "__main__":
    main()
