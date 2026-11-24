"""dqn.py - Double DQN (pure numpy) for rain-aware lane-change steering, trained on the vectorised plant.

State  : 11 normalised features from the *measured* (delayed/noisy) signals + friction estimate (see DQNPolicy.obs)
Action : 7 discrete steering-correction increments added to the kinematic feed-forward (20 Hz decisions)
Reward : -(tracking^2 + heading^2 + friction-limit violation + action size + failure), shaped per decision step
Training episodes randomise rain intensity, speed, sensor noise and friction-estimator error.
"""
import copy, sys, time
import numpy as np
import rainlc as R
from rainlc import DT, G
import plant as P
import policies as PL


class QNet:
    def __init__(self, sizes, rng=None):
        rng = rng or np.random.default_rng(0)
        self.sizes = sizes
        self.W = [rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.W[-1] *= 0.1

    def forward(self, x, cache=False):
        acts = [x]; h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1: h = np.maximum(h, 0)
            acts.append(h)
        return acts if cache else h

    def backward(self, acts, dout):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ d; gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (acts[i] > 0)
        return gW, gb

    def params(self): return self.W + self.b
    def copy(self): return copy.deepcopy(self)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999):
        self.p, self.lr, self.b1, self.b2 = params, lr, b1, b2
        self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]; self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            p -= self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + 1e-8)


def val_scenarios(seed, n=60):
    rng = np.random.default_rng(777 + seed)
    return R.make_scenarios(rng.uniform(0, 1, n), rng.uniform(18, 30, n), rng, mu_err_std=0.10)


def evaluate_q(q, sc):
    m = P.run_policy(PL.DQNPolicy(q), sc)
    J = P.cost_of(m)
    return float(J.mean() + 0.5 * J.max()), float(m["fail"].mean())


def train_dqn(seed=0, n_batches=120, B=128, t_end=10.0, gamma=0.97, lr=1e-3, batch=256, log=True):
    rng = np.random.default_rng(seed)
    q = QNet([PL.OBS_DIM, 128, 128, len(PL.ACTIONS)], rng); tq = q.copy()
    opt = Adam(q.params(), lr)
    cap = 250_000
    Bo = np.zeros((cap, PL.OBS_DIM)); Ba = np.zeros(cap, int); Br = np.zeros(cap); Bn = np.zeros((cap, PL.OBS_DIM))
    ptr, size = 0, 0
    sc_val = val_scenarios(seed)
    best = (np.inf, None, None); curve = []
    n = int(round(t_end / DT)); t_start = time.time()
    for ep in range(n_batches):
        ri = rng.uniform(0, 1, B); v = rng.uniform(18, 32, B)
        sc = R.make_scenarios(ri, v, rng, mu_err_std=0.10, t_end=t_end)
        pol = PL.DQNPolicy(q); pol.setup(sc)
        pl = P.Plant(sc, n, pol.vx)
        eps = max(0.05, 1.0 - ep / (0.4 * n_batches))
        po = pa = None; racc = np.zeros(B); cnt = 0; ep_ret = 0.0
        for k in range(n):
            t = k * DT
            ym, psim, rm = pl.measure(k)
            if k % PL.DECISION_EVERY == 0:
                o = pol.obs(t, ym, psim, rm)
                if po is not None:
                    r = racc / max(cnt, 1)
                    idx = (ptr + np.arange(B)) % cap
                    Bo[idx], Ba[idx], Br[idx], Bn[idx] = po, pa, r, o
                    ptr = (ptr + B) % cap; size = min(size + B, cap); ep_ret += r.mean()
                a = np.argmax(q.forward(o), 1)
                rnd = rng.random(B) < eps
                a = np.where(rnd, rng.integers(0, len(PL.ACTIONS), B), a)
                pol.apply(a)
                po, pa, racc, cnt = o, a, np.zeros(B), 0
                if size >= 4000:
                    j = rng.integers(0, size, batch)
                    s, ac, r_, s2 = Bo[j], Ba[j], Br[j], Bn[j]
                    a2 = np.argmax(q.forward(s2), 1)
                    y = r_ + gamma * tq.forward(s2)[np.arange(batch), a2]
                    acts = q.forward(s, cache=True)
                    qsa = acts[-1][np.arange(batch), ac]
                    dq = np.zeros_like(acts[-1]); dq[np.arange(batch), ac] = np.clip(qsa - y, -1, 1) / batch
                    gW, gb = q.backward(acts, dq)
                    opt.step(gW + gb)
                    for tp, p in zip(tq.params(), q.params()): tp += 0.005 * (p - tp)
            cmd = pol.command(t)
            pol.dh = pol.dh + np.clip((cmd - pol.dh) / R.TAU_ACT, -R.DELTA_RATE, R.DELTA_RATE) * DT
            pol.yref_now = P.ref_at(t, pol.T, pol.vx)[0]
            pl.step(cmd, pol.yref_now)
            _, psi_ref, _ = P.ref_at(t, pol.T, pol.vx)
            e_y = pl.y - pol.yref_now; e_psi = pl.psi - psi_ref
            util = np.abs(pl.AY[k]) / (pl.mu * G)
            rs = ((e_y / 0.2) ** 2 + 0.3 * (e_psi / 0.05) ** 2 + 4 * np.maximum(util - 0.85, 0) / 0.15
                  + 0.1 * (PL.ACTIONS[pa] / 0.012) ** 2 if pa is not None else 0)
            rs = rs + 5.0 * ((np.abs(pl.psi) > 0.5) | (np.abs(pl.BETA[k]) > 0.3))
            racc -= np.minimum(rs, 10.0) * 0.2; cnt += 1
        if (ep + 1) % 10 == 0 or ep == n_batches - 1:
            Jv, fv = evaluate_q(q, sc_val)
            curve.append((ep + 1, ep_ret, Jv, fv))
            if Jv < best[0]: best = (Jv, q.copy(), ep + 1)
            if log: print(f"[DQN s{seed}] batch {ep+1:3d} eps {eps:.2f} train-return {ep_ret:8.1f} val J {Jv:7.2f} fail {fv:.2f} ({time.time()-t_start:.0f}s)", flush=True)
    return best[1], best[0], best[2], curve


if __name__ == "__main__":
    t = time.time()
    q, Jv, ep, curve = train_dqn(0, n_batches=2, log=True)
    print("2 batches:", time.time() - t, "s")
