"""run_extension.py - baselines (MPC, DQN), sim-to-sim validation, optimiser-study analysis, figures and tables."""
import json, os, pickle, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu
import rainlc as R, plant as P, policies as PL, dqn

OUT = "results"
COL = {"Hand-tuned": "#8a8a8a", "GA-Dry": "#d95f02", "GA-Fixed": "#7570b3", "GA-RainAware": "#1b9e77",
       "GA-RainAware-DR": "#0b6b4f", "GA-Fixed-DR": "#4b4685", "MPC": "#e7298a", "MPC-adhesion": "#e6ab02", "MPC-dry": "#a6761d", "DQN": "#1f78b4"}
OPT_COL = {"GA": "#1b9e77", "PSO": "#d95f02", "CMA-ES": "#7570b3", "NSGA-II": "#e7298a"}


def load_dqn():
    best = None
    for s, r in enumerate(json.load(open(f"{OUT}/dqn_training.json"))):
        if best is None or r["val_J"] < best[0]: best = (r["val_J"], r["seed"])
    d = pickle.load(open(f"{OUT}/dqn_seed{best[1]}.pkl", "rb"))
    q = dqn.QNet(d["sizes"]); q.W, q.b = d["W"], d["b"]
    return q, best[1]


def build_controllers():
    params = json.load(open(f"{OUT}/best_parameters.json"))
    arr = lambda k: np.array(list(params[k].values()))
    tm = json.load(open(f"{OUT}/mpc_tuned.json"))
    q, qs = load_dqn()
    dr = json.load(open(f"{OUT}/best_parameters_dr.json"))
    c = {
        "Hand-tuned": lambda: PL.ParamPolicy(arr("Hand-tuned")),
        "GA-Dry": lambda: PL.ParamPolicy(arr("GA-Dry")),
        "GA-Fixed": lambda: PL.ParamPolicy(arr("GA-Fixed")),
        "GA-RainAware": lambda: PL.ParamPolicy(arr("GA-RainAware")),
        "GA-Fixed-DR": lambda: PL.ParamPolicy(np.array(list(dr["GA-Fixed-DR"].values()))),
        "GA-RainAware-DR": lambda: PL.ParamPolicy(np.array(list(dr["GA-RainAware-DR"].values()))),
        "MPC": lambda: PL.MPCPolicy(True, T0=tm["MPC"]["T0"], T1=tm["MPC"]["T1"], r_rate=tm["MPC"]["r_rate"], stiff_exp=0.0),
        "MPC-adhesion": lambda: PL.MPCPolicy(True, T0=tm["MPC-adhesion"]["T0"], T1=tm["MPC-adhesion"]["T1"],
                                              r_rate=tm["MPC-adhesion"]["r_rate"], stiff_exp=1.0),
        "MPC-dry": lambda: PL.MPCPolicy(False, T0=3.0, r_rate=tm["MPC"]["r_rate"], stiff_exp=0.0),
        "DQN": lambda: PL.DQNPolicy(q),
    }
    return c, qs


def run(factory, sc, mismatch=None, **kw):
    t = time.time()
    m = P.run_policy(factory(), sc, mismatch=mismatch, **kw)
    return m, P.cost_of(m), time.time() - t


def table(res, ref="GA-RainAware"):
    rows = []
    for n, (m, J, dt) in res.items():
        ok = m["fail"][0] < 0.5
        row = dict(controller=n, J=f"{J.mean():.2f} ± {J.std():.2f}", fail_pct=f"{100*m['fail'].mean():.1f}",
                   rmse_m=f"{m['rmse'][0][ok].mean():.3f}", final_err_m=f"{m['final'][0][ok].mean():.3f}",
                   friction_util=f"{m['util'].mean():.2f}", duration_s=f"{m['T'].mean():.2f}",
                   jerk=f"{m['jerk'][0][ok].mean():.1f}", sim_s_per_300=f"{dt:.1f}")
        if n != ref:
            try: row["p_vs_GA"] = f"{wilcoxon(J, res[ref][1]).pvalue:.1e}"
            except ValueError: row["p_vs_GA"] = "n/a"
        else: row["p_vs_GA"] = "-"
        rows.append(row)
    return rows


def write_table(rows, path):
    cols = list(rows[0].keys())
    open(path + ".csv", "w").write(",".join(cols) + "\n" + "\n".join(",".join(str(r[c]) for c in cols) for r in rows) + "\n")
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rows)
    open(path + ".md", "w").write(md); return md


def main():
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 130})
    ctrl, dqn_seed = build_controllers()
    print("DQN agent used: seed", dqn_seed, flush=True)
    summary = {}
    # ------------------------------------------------------------------ A. nominal random test
    rng = np.random.default_rng(2026)
    ri = rng.uniform(0, 1, 300); v = rng.uniform(18, 32, 300)
    sc = R.make_scenarios(ri, v, rng, mu_err_std=0.12)
    resA = {n: run(f, sc) for n, f in ctrl.items()}
    print("== A. nominal plant, random held-out test (N=300)"); print(write_table(table(resA), f"{OUT}/ext_table_nominal"), flush=True)
    # ------------------------------------------------------------------ B. sim-to-sim: mismatched higher-fidelity plant
    mm = P.make_mismatch(300, np.random.default_rng(99), 1.0)
    resB = {n: run(f, sc, mismatch=mm) for n, f in ctrl.items()}
    print("== B. mismatched plant (Magic-Formula tyres + relaxation + parameter errors), same 300 scenarios")
    print(write_table(table(resB), f"{OUT}/ext_table_mismatch"), flush=True)
    levels = [0.0, 0.5, 1.0, 1.5, 2.0]; sub = slice(0, 150)
    scs = {k: (v_[sub] if hasattr(v_, "__getitem__") and k != "z" else v_) for k, v_ in sc.items()}
    scs["z"] = sc["z"][:, sub, :]
    mmJ = {n: [] for n in ctrl}; mmF = {n: [] for n in ctrl}
    for lv in levels:
        mml = P.make_mismatch(150, np.random.default_rng(7), lv)
        for n, f in ctrl.items():
            m, J, _ = run(f, scs, mismatch=mml)
            mmJ[n].append(float(J.mean())); mmF[n].append(float(100 * m["fail"].mean()))
    json.dump(dict(levels=levels, J=mmJ, fail=mmF), open(f"{OUT}/ext_mismatch_levels.json", "w"), indent=1)
    # ------------------------------------------------------------------ C. rain sweep (nominal plant)
    lv = np.linspace(0, 1, 11); sw = {n: dict(J=[], fail=[], rmse=[], T=[], util=[]) for n in ctrl}
    for i, r_ in enumerate(lv):
        rr = np.repeat(r_, 60); vv = np.repeat([20., 25., 30.], 20)
        scw = R.make_scenarios(rr, vv, np.random.default_rng(5000 + i), mu_err_std=0.10)
        for n, f in ctrl.items():
            m, J, _ = run(f, scw); ok = m["fail"][0] < .5
            sw[n]["J"].append(float(J.mean())); sw[n]["fail"].append(float(100 * m["fail"].mean()))
            sw[n]["rmse"].append(float(m["rmse"][0][ok].mean()) if ok.any() else np.nan)
            sw[n]["T"].append(float(m["T"].mean())); sw[n]["util"].append(float(m["util"].mean()))
    json.dump(dict(levels=lv.tolist(), **sw), open(f"{OUT}/ext_sweep.json", "w"), indent=1)
    # ------------------------------------------------------------------ figures: baselines
    main_set = ["Hand-tuned", "MPC", "MPC-dry", "DQN", "GA-RainAware", "GA-RainAware-DR"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for a, (k, lab) in zip(ax, [("rmse", "Tracking RMSE [m]"), ("util", "Peak friction utilisation"), ("T", "Lane-change duration [s]"), ("fail", "Failure rate [%]")]):
        for n in main_set: a.plot(lv, sw[n][k], "-o", ms=3, color=COL[n], label=n)
        a.set_xlabel("rain intensity"); a.set_ylabel(lab)
    ax[0].legend(); plt.tight_layout(); plt.savefig(f"{OUT}/ext_fig1_baselines_vs_rain.png"); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for n in ctrl:
        ax[0].plot(levels, mmJ[n], "-o", ms=3, color=COL[n], label=n); ax[1].plot(levels, mmF[n], "-o", ms=3, color=COL[n])
    ax[0].set_yscale("log")
    for a, t_ in zip(ax, ["Mean cost J", "Failure rate [%]"]): a.set_xlabel("plant-mismatch level (0 = nominal model; 1 = moderate)"); a.set_title(t_)
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT}/ext_fig2_mismatch_robustness.png"); plt.close()
    scT = R.make_scenarios([0.85], [27.], np.random.default_rng(3), mu_err_std=0.0)
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for n in ["Hand-tuned", "MPC", "DQN", "GA-RainAware-DR"]:
        m = P.run_policy(ctrl[n](), scT, traj=True); tr = m["traj"]
        ax[0].plot(tr["t"], tr["y"][:, 0], color=COL[n], label=n); ax[1].plot(tr["t"], np.degrees(tr["delta"][:, 0]), color=COL[n])
        ax[2].plot(tr["t"], tr["ay"][:, 0] / (m["mu"][0, 0] * R.G), color=COL[n]); ax[3].plot(tr["t"], np.degrees(tr["beta"][:, 0]), color=COL[n])
    for a, t_ in zip(ax, ["Lateral position y [m]", "Steering angle [deg]", "ay / (μ g)", "Sideslip β [deg]"]): a.set_title(t_); a.set_xlabel("time [s]"); a.set_xlim(0, 10)
    ax[0].legend(); fig.suptitle("Heavy rain (0.85), v0 = 27 m/s", y=1.03); plt.tight_layout(); plt.savefig(f"{OUT}/ext_fig3_heavy_rain_baselines.png", bbox_inches="tight"); plt.close()
    d = json.load(open(f"{OUT}/dqn_training.json"))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for r in d:
        c = np.array(r["curve"]); ax.plot(c[:, 0], np.minimum(c[:, 2], 200), label=f"seed {r['seed']} (best @ batch {r['best_batch']})")
    ax.set_yscale("log"); ax.set_xlabel("training batch (128 episodes each)"); ax.set_ylabel("validation cost (lower better)"); ax.set_title("DQN: validation cost during training")
    ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/ext_fig4_dqn_training.png"); plt.close()
    # ------------------------------------------------------------------ D. optimiser study
    st = json.load(open(f"{OUT}/optimizer_study.json"))
    names = ["GA", "PSO", "CMA-ES", "NSGA-II"]
    rows = []; byopt = {n: [r for r in st if r["name"] == n] for n in names}
    for n in names:
        tj = np.array([r["test_J"] for r in byopt[n]]); ff = np.array([r["test_fail"] for r in byopt[n]]) * 100
        p = "-" if n == "GA" else f"{mannwhitneyu(tj, [r['test_J'] for r in byopt['GA']]).pvalue:.2f}"
        rows.append(dict(optimiser=n, test_J_mean_sd=f"{tj.mean():.2f} ± {tj.std(ddof=1):.2f}", best_seed=f"{tj.min():.2f}",
                         worst_seed=f"{tj.max():.2f}", fail_pct_mean=f"{ff.mean():.2f}", n_seeds_with_failures=int((ff > 0).sum()),
                         wallclock_s=f"{np.mean([r['time'] for r in byopt[n]]):.0f}", p_vs_GA_MannWhitney=p))
    print("== D. optimiser comparison (5 seeds, 6400 evals each)"); print(write_table(rows, f"{OUT}/ext_table_optimisers"), flush=True)
    G = R.GENES
    prow = [dict(optimiser=n, **{g: f"{np.median([r['x'][i] for r in byopt[n]]):.2f}" for i, g in enumerate(G)}) for n in names]
    print(write_table(prow, f"{OUT}/ext_table_optimiser_params"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for n in names:
        ev_ = np.array([[c[0] for c in r["conv"]] for r in byopt[n]]).mean(0)
        cv = np.array([[c[1] for c in r["conv"]] for r in byopt[n]])
        ax[0].plot(ev_, cv.mean(0), "-o", ms=3, color=OPT_COL[n], label=n); ax[0].fill_between(ev_, cv.min(0), cv.max(0), color=OPT_COL[n], alpha=.15)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("fitness evaluations"); ax[0].set_ylabel("validation fitness of best-so-far (common set)")
    ax[0].set_title("Convergence (mean, min–max over 5 seeds)"); ax[0].legend()
    ax[1].boxplot([[r["test_J"] for r in byopt[n]] for n in names], tick_labels=names)
    for i, n in enumerate(names): ax[1].scatter(np.full(5, i + 1) + np.random.default_rng(i).uniform(-.08, .08, 5), [r["test_J"] for r in byopt[n]], color=OPT_COL[n], zorder=3, s=18)
    ax[1].set_ylabel("held-out test cost J"); ax[1].set_title("Final quality across seeds (N=300 test)")
    best_nsga = min(byopt["NSGA-II"], key=lambda r: r["test_J"]); F = np.array(best_nsga["front_f"])
    sc_ = ax[2].scatter(F[:, 0], F[:, 1], c=F[:, 2], cmap="viridis", s=22)
    ax[2].set_xlabel("tracking objective"); ax[2].set_ylabel("safety objective"); ax[2].set_title("NSGA-II Pareto front (colour = efficiency/comfort)")
    plt.colorbar(sc_, ax=ax[2]); plt.tight_layout(); plt.savefig(f"{OUT}/ext_fig5_optimisers.png"); plt.close()
    # Pareto-front trade-off in terms of physical quantities (decode front solutions on test set)
    X = np.array(best_nsga["front_x"]); ts = R.make_scenarios(*(lambda g: (g.uniform(0, 1, 150), g.uniform(18, 32, 150), g))(np.random.default_rng(11)), mu_err_std=0.12)
    mt = R.simulate(X, ts)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    pairs = [(mt["rmse"].mean(1), mt["T"].mean(1), "mean RMSE [m]", "mean duration [s]"),
             (mt["util"].max(1), mt["rmse"].mean(1), "max friction util", "mean RMSE [m]"),
             (mt["T"].mean(1), mt["util"].mean(1), "mean duration [s]", "mean friction util")]
    for a, (xx, yy, xl, yl) in zip(ax, pairs): a.scatter(xx, yy, s=20, color=OPT_COL["NSGA-II"]); a.set_xlabel(xl); a.set_ylabel(yl)
    fig.suptitle("NSGA-II front decoded into physical trade-offs (150 unseen scenarios)", y=1.03)
    plt.tight_layout(); plt.savefig(f"{OUT}/ext_fig6_pareto_physical.png", bbox_inches="tight"); plt.close()
    # NSGA-II knee candidates summary
    sumf = X.shape[0]
    json.dump(dict(dqn_seed=dqn_seed, nsga_front_size=sumf), open(f"{OUT}/ext_misc.json", "w"))
    print("all done", flush=True)


if __name__ == "__main__":
    main()
