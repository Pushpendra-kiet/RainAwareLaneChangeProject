"""run_experiments.py - train GA controllers, evaluate on unseen rain scenarios, write figures/tables."""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.stats import wilcoxon
import rainlc as R
from ga import GeneticAlgorithm

OUT = "results"; os.makedirs(OUT, exist_ok=True)
POP, GENS = 80, 80
TRAIN_RI = np.array([0, .25, .5, .75, 1.0]); TRAIN_V = np.array([20., 24., 28.])

# name -> (training rain levels, locked genes, mu-estimator error std in training, number of GA seeds)
CONFIGS = {
    "GA-RainAware": dict(ri=TRAIN_RI, lock=[],                    err=0.10, seeds=5),
    "GA-Fixed":     dict(ri=TRAIN_RI, lock=["T1", "g", "kv"],     err=0.10, seeds=3),   # ablation: no rain adaptation
    "GA-Dry":       dict(ri=np.array([0.]), lock=["T1", "g", "kv"], err=0.0,  seeds=3),  # tuned on dry road only
}


def grid(ri_levels):
    ri, v = np.meshgrid(ri_levels, TRAIN_V)
    return ri.ravel(), v.ravel()


def bounds(lock):
    lb, ub = R.LB.copy(), R.UB.copy()
    for g in lock:
        i = R.GENES.index(g); lb[i] = ub[i] = 0.0
    return lb, ub


def train(job):
    name, seed = job
    c = CONFIGS[name]
    lb, ub = bounds(c["lock"])
    ri, v = grid(c["ri"])
    def ev(pop, gen):  # fresh noise + estimator errors every generation -> no over-fitting to one noise draw
        sc = R.make_scenarios(ri, v, np.random.default_rng(10_000 * seed + gen), mu_err_std=c["err"])
        return R.fitness(pop, sc)[0]
    ga = GeneticAlgorithm(lb, ub, POP, GENS, seed=seed)
    t = time.time()
    best, f, hist = ga.run(ev)
    # validation of the final-population top-5 on fresh scenarios -> choose the most robust
    top = hist["final_pop"][np.argsort(hist["final_fit"])[:5]]
    vr = np.random.default_rng(777 + seed)
    rv = vr.choice(c["ri"], 60) if len(c["ri"]) > 1 else np.zeros(60)
    rv = np.clip(rv + (vr.uniform(-.08, .08, 60) if len(c["ri"]) > 1 else 0), 0, 1)
    sc = R.make_scenarios(rv, vr.uniform(18, 30, 60), vr, mu_err_std=c["err"])
    vf = R.fitness(top, sc)[0]
    best = top[np.argmin(vf)]
    print(f"[{name} seed {seed}] done in {time.time()-t:.0f}s  train-best {f:.3f}  val {vf.min():.3f}", flush=True)
    return name, seed, best, float(vf.min()), hist["best"], hist["mean"]


def run_training():
    jobs = [(n, s) for n, c in CONFIGS.items() for s in range(c["seeds"])]
    with Pool(2) as p:
        res = p.map(train, jobs, chunksize=1)
    ctrl, runs = {}, {n: [] for n in CONFIGS}
    for name, seed, best, vf, hb, hm in res:
        runs[name].append(dict(seed=seed, x=best, val=vf, best=hb, mean=hm))
    for name, rr in runs.items():
        b = min(rr, key=lambda r: r["val"])
        ctrl[name] = b["x"]
    ctrl = {"Hand-tuned": R.HAND_TUNED, **ctrl}
    json.dump({k: R.unpack(v) for k, v in ctrl.items()}, open(f"{OUT}/best_parameters.json", "w"), indent=2)
    json.dump({n: [dict(seed=r["seed"], val=r["val"], params=R.unpack(r["x"])) for r in rr] for n, rr in runs.items()},
              open(f"{OUT}/all_ga_runs.json", "w"), indent=2)
    return ctrl, runs


# ------------------------------------------------------------------ evaluation
def eval_ctrl(x, sc):
    met = R.simulate(x, sc)
    J = R.cost(met)[0]
    return {k: v[0] for k, v in met.items()}, J


def test_random(ctrl, N=300, seed=2026, mu_err=0.12):
    rng = np.random.default_rng(seed)
    ri = rng.uniform(0, 1, N); v = rng.uniform(18, 32, N)
    sc = R.make_scenarios(ri, v, rng, mu_err_std=mu_err)
    return sc, {k: eval_ctrl(x, sc) for k, x in ctrl.items()}


def table(res, sc):
    names = list(res)
    rows, ref = [], res["GA-RainAware"][1]
    for n in names:
        m, J = res[n]
        ok = m["fail"] < 0.5
        row = dict(controller=n, J=f"{J.mean():.2f} ± {J.std():.2f}", fail_pct=f"{100*m['fail'].mean():.1f}",
                   rmse_m=f"{m['rmse'][ok].mean():.3f}", final_err_m=f"{m['final'][ok].mean():.3f}",
                   friction_util=f"{m['util'].mean():.2f}", duration_s=f"{m['T'].mean():.2f}",
                   speed_loss_pct=f"{100*m['speed_loss'].mean():.1f}", jerk=f"{m['jerk'][ok].mean():.1f}")
        if n != "GA-RainAware":
            try: row["p_vs_RainAware"] = f"{wilcoxon(J, ref).pvalue:.1e}"
            except ValueError: row["p_vs_RainAware"] = "n/a"
        else: row["p_vs_RainAware"] = "-"
        rows.append(row)
    return rows


def write_table(rows, path):
    cols = list(rows[0].keys())
    with open(path + ".csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows: f.write(",".join(str(r[c]) for c in cols) + "\n")
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows: md += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    open(path + ".md", "w").write(md)
    return md


def sweep(ctrl, levels=np.linspace(0, 1, 11), speeds=(20, 25, 30), reps=20, mu_err=0.10):
    out = {k: {m: [] for m in ["rmse", "util", "T", "fail", "J", "speed_loss"]} for k in ctrl}
    for li, ri in enumerate(levels):
        rr, vv = np.repeat(ri, len(speeds) * reps), np.repeat(speeds, reps).tolist() * 1
        vv = np.tile(np.repeat(speeds, reps), 1)
        sc = R.make_scenarios(rr, vv, np.random.default_rng(5000 + li), mu_err_std=mu_err)
        for k, x in ctrl.items():
            m, J = eval_ctrl(x, sc)
            ok = m["fail"] < 0.5
            out[k]["rmse"].append(float(m["rmse"][ok].mean()) if ok.any() else np.nan)
            out[k]["util"].append(float(m["util"].mean())); out[k]["T"].append(float(m["T"].mean()))
            out[k]["fail"].append(float(100 * m["fail"].mean())); out[k]["J"].append(float(J.mean()))
            out[k]["speed_loss"].append(float(100 * m["speed_loss"].mean()))
    return levels, out


def mu_error_sensitivity(ctrl, errs=(0, .1, .2, .3)):
    res = {k: [] for k in ctrl}
    for e in errs:
        rng = np.random.default_rng(31 + int(100 * e))
        ri = rng.uniform(0, 1, 200); v = rng.uniform(18, 32, 200)
        sc = R.make_scenarios(ri, v, rng, mu_err_std=e)
        for k, x in ctrl.items():
            m, J = eval_ctrl(x, sc)
            res[k].append((float(J.mean()), float(100 * m["fail"].mean())))
    return errs, res


# ------------------------------------------------------------------ plots
COL = {"Hand-tuned": "#8a8a8a", "GA-Dry": "#d95f02", "GA-Fixed": "#7570b3", "GA-RainAware": "#1b9e77"}


def plot_all(ctrl, runs, levels, sw, errs, es, res):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 130})
    # 1 convergence
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for n, rr in runs.items():
        B = np.minimum.accumulate(np.array([r["best"] for r in rr]), axis=1)
        M = np.array([r["mean"] for r in rr])
        g = np.arange(B.shape[1])
        ax[0].plot(g, B.mean(0), color=COL[n], label=n); ax[0].fill_between(g, B.min(0), B.max(0), color=COL[n], alpha=.2)
        ax[1].plot(g, np.median(M, 0), color=COL[n], label=n)
    ax[0].set_title("Best-so-far fitness (mean, min–max over GA seeds)"); ax[1].set_title("Population-mean fitness (median)")
    ax[1].set_yscale("log")
    for a in ax: a.set_xlabel("generation"); a.set_ylabel("fitness (lower = better)")
    ax[0].legend(); plt.tight_layout(); plt.savefig(f"{OUT}/fig1_convergence.png"); plt.close()
    # 2 rain sweep
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
    items = [("rmse", "Tracking RMSE [m]"), ("util", "Peak friction utilisation |ay|/(μg)"), ("T", "Lane-change duration [s]"),
             ("fail", "Failure rate [%]"), ("speed_loss", "Speed reduction [%]"), ("J", "Mean cost J")]
    for a, (k, lab) in zip(ax.ravel(), items):
        for n in ctrl: a.plot(levels, sw[n][k], "-o", ms=3, color=COL[n], label=n)
        a.set_xlabel("rain intensity"); a.set_ylabel(lab)
    ax[0, 1].axhline(1.0, color="k", ls="--", lw=.8); ax[1, 2].set_yscale("log")
    ax[0, 0].legend(); plt.tight_layout(); plt.savefig(f"{OUT}/fig2_rain_sweep.png"); plt.close()
    # 3 trajectories in heavy rain
    sc = R.make_scenarios([0.85], [27.], np.random.default_rng(3), mu_err_std=0.0)
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for n, x in ctrl.items():
        m = R.simulate(x, sc, traj=True); tr = m["traj"]
        ax[0].plot(tr["t"], tr["y"][:, 0], color=COL[n], label=n)
        ax[1].plot(tr["t"], np.degrees(tr["delta"][:, 0]), color=COL[n])
        ax[2].plot(tr["t"], tr["ay"][:, 0] / (m["mu"][0, 0] * R.G), color=COL[n])
        ax[3].plot(tr["t"], np.degrees(tr["beta"][:, 0]), color=COL[n])
    ax[0].axhline(R.LANE_W, color="k", ls=":", lw=.8); ax[2].axhline(1, color="k", ls="--", lw=.8); ax[2].axhline(-1, color="k", ls="--", lw=.8)
    for a, t_ in zip(ax, ["Lateral position y [m]", "Steering angle [deg]", "ay / (μ g)  (±1 = grip limit)", "Body sideslip β [deg]"]):
        a.set_title(t_); a.set_xlabel("time [s]"); a.set_xlim(0, 10)
    ax[0].legend(); fig.suptitle("Heavy rain (intensity 0.85, μ≈0.37), v0 = 27 m/s", y=1.03)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig3_trajectories_heavy_rain.png", bbox_inches="tight"); plt.close()
    # 4 evolved schedule
    ri = np.linspace(0, 1, 50); mu = R.rain_to_mu(ri); d = np.clip(1 - mu / R.MU_DRY, 0, 1)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    for n in ["GA-RainAware", "GA-Fixed", "Hand-tuned"]:
        p = R.unpack(ctrl[n])
        ax[0].plot(ri, p["T0"] + p["T1"] * d, color=COL[n], label=n)
        ax[1].plot(ri, np.clip(mu / R.MU_DRY, .1, 1.1) ** p["g"], color=COL[n])
        ax[2].plot(ri, 25 * (1 - p["kv"] * d), color=COL[n])
    for a, t_ in zip(ax, ["Evolved lane-change duration T [s]", "Feedback gain scale (μ/μ_dry)^g", "Target speed [m/s] (v0 = 25)"]):
        a.set_title(t_); a.set_xlabel("rain intensity")
    ax[0].legend(); plt.tight_layout(); plt.savefig(f"{OUT}/fig4_evolved_schedule.png"); plt.close()
    # 5 friction-estimator robustness
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6)); w = .2
    for i, n in enumerate(ctrl):
        ax[0].bar(np.arange(len(errs)) + i * w, [e[0] for e in es[n]], w, color=COL[n], label=n)
        ax[1].bar(np.arange(len(errs)) + i * w, [e[1] for e in es[n]], w, color=COL[n])
    for a, t_ in zip(ax, ["Mean cost J", "Failure rate [%]"]):
        a.set_xticks(np.arange(len(errs)) + 1.5 * w); a.set_xticklabels([f"{int(100*e)}%" for e in errs])
        a.set_xlabel("friction-estimator error (std)"); a.set_title(t_)
    ax[0].set_yscale("log"); ax[0].legend(); plt.tight_layout(); plt.savefig(f"{OUT}/fig5_mu_estimation_error.png"); plt.close()
    # 6 distribution of cost on random test set
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.boxplot([res[n][1] for n in ctrl], tick_labels=list(ctrl), showfliers=False)
    ax.set_ylabel("cost J per test scenario"); ax.set_title("Random held-out test set (N=300)")
    plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(f"{OUT}/fig6_test_cost_boxplot.png"); plt.close()


if __name__ == "__main__":
    t0 = time.time()
    ctrl, runs = run_training()
    print("training done", f"{time.time()-t0:.0f}s", flush=True)
    sc, res = test_random(ctrl)
    rows = table(res, sc)
    print(write_table(rows, f"{OUT}/table_random_test"))
    levels, sw = sweep(ctrl)
    json.dump({k: v for k, v in sw.items()}, open(f"{OUT}/sweep_vs_rain.json", "w"), indent=1)
    errs, es = mu_error_sensitivity(ctrl)
    json.dump({k: v for k, v in es.items()}, open(f"{OUT}/mu_error_sensitivity.json", "w"), indent=1)
    plot_all(ctrl, runs, levels, sw, errs, es, res)
    print("params:"); print(json.dumps({k: {a: round(b, 3) for a, b in R.unpack(v).items()} for k, v in ctrl.items()}, indent=1))
    print("all done", f"{time.time()-t0:.0f}s")
