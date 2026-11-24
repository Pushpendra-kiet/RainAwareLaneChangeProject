"""
optimizers.py - PSO, CMA-ES and NSGA-II (pure numpy) with the same interface/budget as ga.GeneticAlgorithm
====================================================================================================
Every optimiser minimises the same noisy simulation-based fitness; the scenario noise is re-drawn each generation
(common random numbers inside a generation). Budget: 6400 fitness evaluations per run for every optimiser.
Each run returns:  candidates (k,n), history {'best_x': [per-generation best], 'evals_per_gen': int}, extras.
"""
import numpy as np
from ga import GeneticAlgorithm
import rainlc as R


# ----------------------------------------------------------------------------- multi-objective fitness (NSGA-II)
def fitness_groups(pop, sc, w=R.W, worst_weight=0.5):
    """Three objectives whose sum equals the scalar cost: tracking, safety, efficiency/comfort."""
    met = R.simulate(pop, sc)
    track = w["rmse"] * met["rmse"] + w["maxe"] * met["maxe"] + w["final"] * met["final"] + w["over"] * met["over"]
    safety = w["util"] * np.maximum(met["util"] - w["util_thr"], 0) + w["beta"] * met["beta"] + w["fail"] * met["fail"]
    eff = w["jerk"] * met["jerk"] + w["srate"] * met["srate"] + w["T"] * met["T"] + w["speed"] * met["speed_loss"]
    return np.stack([g.mean(1) + worst_weight * g.max(1) for g in (track, safety, eff)], 1)


# ----------------------------------------------------------------------------- PSO
def pso(ev, lb, ub, swarm=40, iters=80, seed=0, c1=1.7, c2=1.7, w0=0.9, w1=0.4):
    """Global-best PSO with linearly decreasing inertia. Personal bests are re-evaluated together with the
    positions each iteration (same noise draw), so the budget is 2*swarm*iters evaluations."""
    rng = np.random.default_rng(seed)
    span = ub - lb
    x = lb + rng.random((swarm, lb.size)) * span
    v = rng.uniform(-0.2, 0.2, x.shape) * span
    pb = x.copy(); best_hist = []
    pbf = ev(np.vstack([x, pb]), 0)[swarm:]
    for it in range(iters):
        g = pb[np.argmin(pbf)]
        best_hist.append(g.copy())
        if it == iters - 1: break
        w = w0 - (w0 - w1) * it / (iters - 1)
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = np.clip(w * v + c1 * r1 * (pb - x) + c2 * r2 * (g - x), -0.5 * span, 0.5 * span)
        x = np.clip(x + v, lb, ub)
        f = ev(np.vstack([x, pb]), it + 1)
        fx, fpb = f[:swarm], f[swarm:]
        imp = fx < fpb
        pb = np.where(imp[:, None], x, pb); pbf = np.where(imp, fx, fpb)
    order = np.argsort(pbf)[:5]
    return pb[order], dict(best_x=best_hist, evals_per_gen=2 * swarm), {}


# ----------------------------------------------------------------------------- CMA-ES
def cmaes(ev, lb, ub, lam=20, gens=320, seed=0, sigma0=0.25):
    """(mu/mu_w, lambda)-CMA-ES on the unit hypercube (box constraints by clipping)."""
    rng = np.random.default_rng(seed)
    n = lb.size; mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum(); mueff = 1 / (w ** 2).sum()
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n); cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff); cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    mean = rng.random(n); sigma = sigma0
    pc = np.zeros(n); ps = np.zeros(n); C = np.eye(n); Bm = np.eye(n); D = np.ones(n)
    best_hist, last = [], []
    for g in range(gens):
        z = rng.standard_normal((lam, n))
        y = (Bm @ (D[:, None] * z.T)).T
        xs = np.clip(mean + sigma * y, 0, 1)
        f = ev(lb + xs * (ub - lb), g)
        idx = np.argsort(f)
        best_hist.append(lb + xs[idx[0]] * (ub - lb)); last.append(best_hist[-1])
        xsel = xs[idx[:mu]]; ysel = (xsel - mean) / sigma
        new_mean = w @ xsel; yw = (new_mean - mean) / sigma
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Bm @ ((Bm.T @ yw) / D))
        hsig = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chiN < 1.4 + 2 / (n + 1))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * yw
        C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (ysel.T * w) @ ysel)
        sigma = min(sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1)), 0.6)
        C = (C + C.T) / 2
        Dsq, Bm = np.linalg.eigh(C); D = np.sqrt(np.maximum(Dsq, 1e-20))
        mean = new_mean
    cands = np.vstack([lb + np.clip(mean, 0, 1) * (ub - lb)] + last[-4:])
    return cands, dict(best_x=best_hist, evals_per_gen=lam), dict(final_sigma=float(sigma))


# ----------------------------------------------------------------------------- NSGA-II
def _fast_nds(F):
    n = len(F)
    dom = (F[:, None, :] <= F[None, :, :]).all(2) & (F[:, None, :] < F[None, :, :]).any(2)   # dom[i,j]: i dominates j
    cnt = dom.sum(0); rank = np.full(n, -1); cur = np.where(cnt == 0)[0]; r = 0
    cnt = cnt.copy()
    while cur.size:
        rank[cur] = r
        nxt = []
        for i in cur:
            for j in np.where(dom[i])[0]:
                cnt[j] -= 1
                if cnt[j] == 0: nxt.append(j)
        cur = np.array(nxt, int); r += 1
    return rank


def _crowding(F, rank):
    cd = np.zeros(len(F))
    for r in np.unique(rank):
        ids = np.where(rank == r)[0]
        if ids.size <= 2: cd[ids] = np.inf; continue
        for m in range(F.shape[1]):
            o = ids[np.argsort(F[ids, m])]
            span = F[o[-1], m] - F[o[0], m] + 1e-12
            cd[o[0]] = cd[o[-1]] = np.inf
            cd[o[1:-1]] += (F[o[2:], m] - F[o[:-2], m]) / span
    return cd


def nsga2(ev_multi, lb, ub, pop=40, gens=80, seed=0):
    """NSGA-II (3 objectives). Parents are re-evaluated with the offspring each generation (same noise draw), so the
    budget is 2*pop*gens evaluations. Candidates = 5 members of the final first front with the lowest objective sum."""
    ga = GeneticAlgorithm(lb, ub, pop_size=pop, seed=seed)
    rng = ga.rng
    P = ga._init(); best_hist = []
    Pf = ev_multi(P, 0)
    for g in range(1, gens + 1):
        rank = _fast_nds(Pf); cd = _crowding(Pf, rank)
        a, b = rng.integers(0, pop, pop), rng.integers(0, pop, pop)
        better = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (cd[a] > cd[b]))
        sel = P[np.where(better, a, b)]
        Q = np.clip(ga.mutate(ga.crossover(sel)), lb, ub)
        allx = np.vstack([P, Q])
        allf = ev_multi(allx, g)                  # parents + offspring evaluated under the same noise draw
        rank = _fast_nds(allf); cd = _crowding(allf, rank)
        order = np.lexsort((-cd, rank))[:pop]
        P, Pf = allx[order], allf[order]
        r0 = np.where(rank[order] == 0)[0]
        best_hist.append(P[r0[np.argmin(Pf[r0].sum(1))]].copy())
    rank = _fast_nds(Pf); front = np.where(rank == 0)[0]
    order = front[np.argsort(Pf[front].sum(1))[:5]]
    return P[order], dict(best_x=best_hist, evals_per_gen=2 * pop), dict(front_x=P[front], front_f=Pf[front])
