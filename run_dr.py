"""run_dr.py - GA trained with domain randomisation (plant mismatch) vs the nominal-trained GA."""
import json, time
import numpy as np
from multiprocessing import Pool
import rainlc as R, plant as P, policies as PL, dr
from ga import GeneticAlgorithm

OUT = "results"
RI, V = [a.ravel() for a in np.meshgrid(np.array([0, .25, .5, .75, 1.0]), np.array([20., 24., 28.]))]
CONF = {"GA-RainAware-DR": [], "GA-Fixed-DR": ["T1", "g", "kv"]}


def job(args):
    name, seed = args
    lb, ub = R.LB.copy(), R.UB.copy()
    for g in CONF[name]:
        i = R.GENES.index(g); lb[i] = ub[i] = 0.0

    def ev(pop, gen):
        rng = np.random.default_rng(10_000 * seed + gen)
        sc = R.make_scenarios(RI, V, rng, mu_err_std=0.10)
        mm = P.make_mismatch(sc["ri"].size, rng, 1.0)
        return dr.fitness_dr(pop, sc, mm)[0]
    t = time.time()
    _, _, h = GeneticAlgorithm(lb, ub, 80, 80, seed=seed).run(ev)
    top = h["final_pop"][np.argsort(h["final_fit"])[:5]]
    vr = np.random.default_rng(777 + seed)
    rv = np.clip(vr.uniform(0, 1, 60), 0, 1)
    sc = R.make_scenarios(rv, vr.uniform(18, 30, 60), vr, mu_err_std=0.10)
    vf = dr.fitness_dr(top, sc, P.make_mismatch(60, vr, 1.0))[0]
    best = top[np.argmin(vf)]
    print(f"[{name} s{seed}] {time.time()-t:.0f}s val {vf.min():.3f}", flush=True)
    return name, seed, best.tolist(), float(vf.min())


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(n, s) for n in CONF for s in range(4)]
    with Pool(2) as p: res = p.map(job, jobs, chunksize=1)
    out = {}
    for n in CONF:
        b = min([r for r in res if r[0] == n], key=lambda r: r[3])
        out[n] = dict(zip(R.GENES, b[2]))
    json.dump(out, open(f"{OUT}/best_parameters_dr.json", "w"), indent=1)
    json.dump([dict(name=r[0], seed=r[1], val=r[3], x=r[2]) for r in res], open(f"{OUT}/dr_runs.json", "w"), indent=1)
    print(json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in out.items()}, indent=1)); print("done", time.time() - t0)
