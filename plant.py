"""
plant.py - generic vectorised plant + policy runner
===================================================
Lets *any* controller (GA-tuned law, MPC, DQN ...) run on the same rain / sensor / vehicle simulation.

Policy interface (all arrays have length B = number of scenarios):
    policy.setup(sc)                -> must set policy.T (manoeuvre duration) and policy.vx (target speed)
    policy.act(k, t, ym, psim, rm)  -> steering command [rad]; must set policy.yref_now (reference at time t)
The policy only sees delayed/noisy lane measurements, IMU yaw rate and the *estimated* friction in `sc`.

Plant variants
    mismatch=None      : identical to rainlc.simulate (tanh tyres, nominal parameters)
    mismatch=dict(...) : "higher-fidelity" plant for sim-to-sim validation - Magic-Formula tyres with
                         post-peak drop, tyre relaxation lag, random mass/inertia/stiffness/CG/steering-gain errors.
"""
import numpy as np
import rainlc as R
from rainlc import G, VEH, MU_DRY, LANE_W, T_START, DT, T_END, DELTA_MAX, DELTA_RATE, TAU_ACT


def make_mismatch(B, rng, level=1.0):
    """Random per-scenario parameter errors (level=1 -> moderate; 0 -> none)."""
    u = lambda lo, hi: 1 + level * rng.uniform(lo, hi, B)
    return dict(m_s=u(-0.05, 0.20), Iz_s=u(-0.10, 0.20), Caf_s=u(-0.20, 0.20), Car_s=u(-0.20, 0.20),
                a_s=u(-0.05, 0.05), steer_gain=u(-0.10, 0.10), mu_s=u(-0.10, 0.05),
                sigma=0.8, C=1.4, E=-0.6)


def ref_at(t, T, vx, preview=0.0):
    """Quintic lane-change reference. Returns (y_ref, psi_ref, kappa_ref) at time t+preview."""
    tau = np.clip((t + preview - T_START) / T, 0, 1)
    s, ds, dds = R._quintic(tau, T)
    return LANE_W * s, np.arctan2(LANE_W * ds, vx), LANE_W * dds / vx ** 2


class Plant:
    def __init__(self, sc, n, vx, mismatch=None, dt=DT):
        self.dt, self.n = dt, n
        B = sc["ri"].size
        self.B = B
        self.sc = sc
        self.vx = vx
        self.mu = sc["mu_nom"] * (1 - 0.006 * np.maximum(vx - 15, 0) * sc["ri"])
        self.nd = sc["nd"]
        zn = sc["z"]
        self.ny = zn[:, :, 0] * sc["sig_y"]
        self.npsi = zn[:, :, 1] * sc["sig_psi"]
        self.nr = zn[:, :, 2] * sc["sig_r"]
        self.mm = mismatch
        one = np.ones(B)
        mm = mismatch or {}
        self.m = VEH["m"] * mm.get("m_s", one)
        self.Iz = VEH["Iz"] * mm.get("Iz_s", one)
        self.a = VEH["a"] * mm.get("a_s", one)
        self.b = VEH["L"] - self.a
        self.Caf = VEH["Caf"] * mm.get("Caf_s", one)
        self.Car = VEH["Car"] * mm.get("Car_s", one)
        self.sgain = mm.get("steer_gain", one)
        if mismatch is not None:
            self.mu = self.mu * mm["mu_s"]
        L = VEH["L"]
        self.Fzf = self.m * G * self.b / L
        self.Fzr = self.m * G * self.a / L
        self.y = np.zeros(B); self.psi = np.zeros(B); self.vy = np.zeros(B); self.r = np.zeros(B)
        self.delta = np.zeros(B); self.Ff = np.zeros(B); self.Fr = np.zeros(B)
        self.k = 0
        self.Yh = np.zeros((n, B)); self.PSIh = np.zeros((n, B))
        self.Yref = np.zeros((n, B)); self.AY = np.zeros((n, B)); self.DEL = np.zeros((n, B))
        self.BETA = np.zeros((n, B)); self.PSI = np.zeros((n, B))
        self.idx = np.arange(B)

    def measure(self, k):
        self.k = k
        self.Yh[k] = self.y; self.PSIh[k] = self.psi
        kk = np.maximum(k - self.nd, 0)
        return (self.Yh[kk, self.idx] + self.ny[k], self.PSIh[kk, self.idx] + self.npsi[k], self.r + self.nr[k])

    def _tyres(self, alf, alr):
        mu = self.mu
        if self.mm is None:
            Fyf = -mu * self.Fzf * np.tanh(self.Caf * alf / (mu * self.Fzf))
            Fyr = -mu * self.Fzr * np.tanh(self.Car * alr / (mu * self.Fzr))
            return Fyf, Fyr
        C, E, sig = self.mm["C"], self.mm["E"], self.mm["sigma"]

        def mf(al, Ca, Fz):
            D = mu * Fz
            Bc = Ca / (C * D)
            x = Bc * al
            return -D * np.sin(C * np.arctan(x - E * (x - np.arctan(x))))
        Fyf_ss, Fyr_ss = mf(alf, self.Caf, self.Fzf), mf(alr, self.Car, self.Fzr)
        lam = np.minimum(self.vx * self.dt / sig, 1.0)          # tyre relaxation
        self.Ff = self.Ff + (Fyf_ss - self.Ff) * lam
        self.Fr = self.Fr + (Fyr_ss - self.Fr) * lam
        return self.Ff, self.Fr

    def step(self, cmd, yref=None):
        k, dt, vx = self.k, self.dt, self.vx
        cmd = np.clip(cmd, -DELTA_MAX, DELTA_MAX)
        ddot = np.clip((cmd - self.delta) / TAU_ACT, -DELTA_RATE, DELTA_RATE)
        self.delta = self.delta + ddot * dt
        d_eff = self.delta * self.sgain
        alf = np.arctan2(self.vy + self.a * self.r, vx) - d_eff
        alr = np.arctan2(self.vy - self.b * self.r, vx)
        Fyf, Fyr = self._tyres(alf, alr)
        ay = (Fyf * np.cos(d_eff) + Fyr) / self.m
        vy_dot = ay - vx * self.r
        r_dot = (self.a * Fyf * np.cos(d_eff) - self.b * Fyr) / self.Iz
        self.vy = np.clip(self.vy + vy_dot * dt, -15, 15)
        self.r = np.clip(self.r + r_dot * dt, -3, 3)
        self.psi = self.psi + self.r * dt
        self.y = self.y + (vx * np.sin(self.psi) + self.vy * np.cos(self.psi)) * dt
        if yref is not None: self.Yref[k] = yref
        self.AY[k] = ay; self.DEL[k] = self.delta
        self.BETA[k] = np.arctan2(self.vy, vx); self.PSI[k] = self.psi
        return ay

    # ------------------------------------------------------------------ metrics (same definitions as rainlc.simulate)
    def metrics(self, T, v0):
        dt, n = self.dt, self.n
        Y, Yref, AY, DEL, BETA, PSI = self.Yh, self.Yref, self.AY, self.DEL, self.BETA, self.PSI
        tt = (np.arange(n) * dt)[:, None]
        mask = (tt >= T_START) & (tt <= T_START + T[None, :] + 1.0)
        cnt = mask.sum(0)
        err = Y - Yref
        rmse = np.sqrt((err ** 2 * mask).sum(0) / cnt)
        maxe = (np.abs(err) * mask).max(0)
        final = np.abs(Y[-int(2 / dt):] - LANE_W).mean(0)
        over = np.maximum(Y.max(0) - LANE_W, 0)
        util = (np.abs(AY) / (self.mu[None, :] * G)).max(0)
        jerk = np.minimum(np.sqrt(((np.diff(AY, axis=0) / dt) ** 2 * mask[1:]).sum(0) / cnt), 50)
        srate = np.sqrt(((np.diff(DEL, axis=0) / dt) ** 2 * mask[1:]).sum(0) / cnt)
        maxbeta = np.abs(BETA).max(0)
        fail = ((np.abs(PSI).max(0) > 0.5) | (maxbeta > 0.3) | (final > 1.0) | ~np.isfinite(rmse)).astype(float)
        rmse, maxe, final, over = np.minimum(rmse, 5), np.minimum(maxe, 10), np.minimum(final, 5), np.minimum(over, 5)
        out = dict(rmse=rmse, maxe=maxe, final=final, over=over, util=util, jerk=jerk, srate=srate,
                   beta=maxbeta, T=T, speed_loss=(v0 - self.vx) / v0, fail=fail, vx=self.vx, mu=self.mu)
        # same layout as rainlc.simulate: (P=1, S)
        return {k: np.nan_to_num(v, nan=1e3, posinf=1e3, neginf=-1e3).reshape(1, -1) for k, v in out.items()}


def run_policy(policy, sc, mismatch=None, traj=False, dt=DT, t_end=T_END, plant_cls=None):
    """Simulate one policy on S scenarios. Returns metrics dict of (1,S) arrays (like rainlc.simulate for P=1)."""
    n = int(round(t_end / dt))
    policy.setup(sc)
    pl = (plant_cls or Plant)(sc, n, policy.vx, mismatch, dt)
    for k in range(n):
        ym, psim, rm = pl.measure(k)
        cmd = policy.act(k, k * dt, ym, psim, rm)
        pl.step(cmd, policy.yref_now)
    met = pl.metrics(policy.T, sc["v0"])
    if traj:
        met["traj"] = dict(t=np.arange(n) * dt, y=pl.Yh, yref=pl.Yref, delta=pl.DEL, ay=pl.AY, mu=pl.mu,
                           vx=pl.vx, T=policy.T, beta=pl.BETA)
        for nm in ("PHI", "VX", "FZ"):
            if hasattr(pl, nm): met["traj"][nm.lower()] = getattr(pl, nm)
    return met


def cost_of(met, w=R.W):
    return R.cost(met, w)[0]
