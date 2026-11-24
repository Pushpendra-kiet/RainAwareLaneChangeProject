"""run_optimizers.py - GA vs PSO vs CMA-ES vs NSGA-II on the rain-aware lane-change tuning problem (5 seeds each)."""
import json, os, time, pickle
import numpy as np
from multiprocessing import Pool
import rainlc as R
from ga import GeneticAlgorithm
import optimizers as O

OUT = "results"; os.makedirs(OUT, exist_ok=True)
TRAIN_RI = np.array([0, .25, .5, .75, 1.0]); TRAIN_V = np.array([20., 24., 28.])
RI, V = [a.ravel() for a in np.meshgrid(TRAIN_RI, TRAIN_V)]
SEEDS = range(5)
BUDGET = 6400


def val_set(seed):
    rng = np.random.default_rng(777 + seed)
    return R.make_scenarios(np.clip(rng.choice(TRAIN_RI, 60) + rng.uniform(-.08, .08, 60), 0, 1),
                            rng.uniform(18, 30, 60), rng, mu_err_std=0.10)


def common_val():
    rng = np.random.default_rng(4242)
    return R.make_scenarios(rng.uniform(0, 1, 60), rng.uniform(18, 30, 60), rng, mu_err_std=0.10)


def test_set():
    rng = np.random.default_rng(2026)
    ri = rng.uniform(0, 1, 300); v = rng.uniform(18, 32, 300)
    return R.make_scenarios(ri, v, rng, mu_err_std=0.12)


def job(args):
    name, seed = args
    mk = lambda g: R.make_scenarios(RI, V, np.random.default_rng(10_000 * seed + g), mu_err_std=0.10)
    ev = lambda pop, g: R.fitness(pop, mk(g))[0]
    evm = lambda pop, g: O.fitness_groups(pop, mk(g))
    t0 = time.time(); extras = {}
    if name == "GA":
        ga = GeneticAlgorithm(R.LB, R.UB, 80, 80, seed=seed)
        _, _, h = ga.run(ev)
        cands = h["final_pop"][np.argsort(h["final_fit"])[:5]]
        hist = dict(best_x=h["gen_best_x"], evals_per_gen=80)
    elif name == "PSO":
        cands, hist, extras = O.pso(ev, R.LB, R.UB, swarm=40, iters=80, seed=seed)
    elif name == "CMA-ES":
        cands, hist, extras = O.cmaes(ev, R.LB, R.UB, lam=20, gens=320, seed=seed)
    elif name == "NSGA-II":
        cands, hist, extras = O.nsga2(evm, R.LB, R.UB, pop=40, gens=80, seed=seed)
    # same selection protocol for every optimiser: best of the (<=5) returned candidates on a fresh validation set
    vf = R.fitness(cands, val_set(seed))[0]
    best = cands[np.argmin(vf)]
    # convergence: evaluate generation-best solutions at 10 checkpoints on a common validation set
    cv = common_val(); G_ = len(hist["best_x"])
    cps = np.unique(np.linspace(0, G_ - 1, 10).astype(int))
    conv = [(int((c + 1) * hist["evals_per_gen"]), float(R.fitness(hist["best_x"][c][None], cv)[0][0])) for c in cps]
    # held-out test
    ts = test_set()
    met = R.simulate(best, ts); J = R.cost(met)[0]
    res = dict(name=name, seed=seed, x=best.tolist(), val=float(vf.min()), test_J=float(J.mean()),
               test_fail=float(met["fail"].mean()), test_rmse=float(met["rmse"].mean()), time=time.time() - t0,
               conv=conv, J_all=J.tolist())
    if name == "NSGA-II":
        res["front_x"] = extras["front_x"].tolist(); res["front_f"] = extras["front_f"].tolist()
    print(f"[{name} s{seed}] {time.time()-t0:.0f}s val {vf.min():.3f} test J {J.mean():.3f} fail {met['fail'].mean():.3f}", flush=True)
    return res


if __name__ == "__main__":
    t = time.time()
    jobs = [(n, s) for s in SEEDS for n in ["GA", "PSO", "CMA-ES", "NSGA-II"]]
    with Pool(2) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/optimizer_study.json", "w"))
    print("done", time.time() - t)
