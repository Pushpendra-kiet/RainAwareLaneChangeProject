"""
rainlc.py - Rain-aware lane-change simulation core
===================================================
Vectorised (batch) simulation of a lane-change manoeuvre on a wet road.

Components
----------
1. Vehicle   : 3-DOF-style dynamic single-track (bicycle) model, saturating tyres (tanh brush-like model).
2. Rain      : rain intensity ri in [0,1] -> (a) peak friction mu, (b) speed-dependent mu loss (water film),
               (c) degraded perception: measurement delay + noise grow with ri.
3. Controller: friction-aware lane-change controller (quintic reference + preview feedback + feed-forward).
               Nine genes (see GENES) are evolved by the GA.
4. Metrics   : tracking, final error, overshoot, friction utilisation, jerk, steering rate, manoeuvre time,
               speed loss, failure flag. Combined into a scalar cost J (lower is better).

All simulation is NumPy-vectorised over (individuals x scenarios) so a whole GA generation is one call.
"""
import numpy as np

G = 9.81
MU_DRY = 0.9
LANE_W = 3.5      # lateral offset of the lane change [m]
T_START = 1.0     # manoeuvre start time [s]
DT = 0.01
T_END = 14.0

# ---------------------------------------------------------------- vehicle (mid-size sedan, illustrative)
VEH = dict(m=1500.0, Iz=2500.0, a=1.1, b=1.6, Caf=90000.0, Car=110000.0)
VEH["L"] = VEH["a"] + VEH["b"]
VEH["Fzf"] = VEH["m"] * G * VEH["b"] / VEH["L"]
VEH["Fzr"] = VEH["m"] * G * VEH["a"] / VEH["L"]
VEH["Kus"] = VEH["m"] / VEH["L"] * (VEH["b"] / VEH["Caf"] - VEH["a"] / VEH["Car"])  # understeer gradient

DELTA_MAX = 0.45      # steering angle limit [rad]
DELTA_RATE = 0.8      # steering rate limit [rad/s]
TAU_ACT = 0.08        # steering actuator lag [s]

# ---------------------------------------------------------------- chromosome definition
GENES = ["T0", "T1", "ky", "kpsi", "kr", "tla", "ffg", "g", "kv"]
LB = np.array([1.8, 0.0, 0.00, 0.0, 0.0, 0.2, 0.0, -0.5, 0.0])
UB = np.array([5.0, 6.0, 0.15, 2.0, 1.0, 1.5, 1.5, 2.0, 0.8])
DESCR = {
    "T0":   "base lane-change duration [s] (dry road)",
    "T1":   "extra duration per unit friction deficit d=1-mu_est/mu_dry [s]",
    "ky":   "lateral preview-error gain [rad/m]",
    "kpsi": "heading-error gain [rad/rad]",
    "kr":   "yaw-rate damping gain [rad/(rad/s)]",
    "tla":  "preview (look-ahead) time [s]",
    "ffg":  "feed-forward scale on nominal-model steering (0..1.5)",
    "g":    "friction gain-scheduling exponent: gains *= (mu_est/mu_dry)^g",
    "kv":   "speed reduction per unit friction deficit: v = v0*(1-kv*d)",
}

# cost weights (see README for the rationale)
W = dict(rmse=10.0, maxe=1.0, final=3.0, over=4.0, util=20.0, util_thr=0.85,
         jerk=0.15, srate=0.5, T=0.8, speed=2.0, beta=3.0, fail=50.0)


# ---------------------------------------------------------------- scenarios
def rain_to_mu(ri):
    """rain intensity [0..1] -> nominal peak friction. 0 -> 0.90 (dry), 1 -> 0.35 (heavy rain/standing water)."""
    return MU_DRY - 0.55 * np.asarray(ri, float) ** 0.7


def make_scenarios(ri, v0, rng, mu_err_std=0.08, dt=DT, t_end=T_END):
    """Build S scenarios. mu_est = mu_nom*(1+err) models an imperfect online friction estimator."""
    ri = np.atleast_1d(np.asarray(ri, float))
    v0 = np.broadcast_to(np.asarray(v0, float), ri.shape).copy()
    S = ri.size
    n = int(round(t_end / dt))
    mu_nom = rain_to_mu(ri)
    err = np.clip(rng.normal(0.0, mu_err_std, S), -0.3, 0.3)
    mu_est = np.clip(mu_nom * (1 + err), 0.12, 1.0)
    return dict(
        ri=ri, v0=v0, mu_nom=mu_nom, mu_est=mu_est,
        nd=np.round((0.05 + 0.15 * ri) / dt).astype(int),        # camera/perception delay 50-200 ms
        sig_y=0.02 + 0.12 * ri, sig_psi=0.002 + 0.012 * ri,      # lane-detection noise grows with rain
        sig_r=np.full(S, 0.003),
        z=rng.standard_normal((n, S, 3)),
    )


# ---------------------------------------------------------------- simulation
def _quintic(tau, T):
    s = tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)
    ds = 30 * tau ** 2 * (1 - tau) ** 2 / T
    dds = 60 * tau * (1 - tau) * (1 - 2 * tau) / T ** 2
    return s, ds, dds


def simulate(pop, sc, dt=DT, t_end=T_END, traj=False):
    """Simulate every individual in `pop` (P,9) on every scenario in `sc` (S). Returns metrics (P,S)."""
    pop = np.atleast_2d(pop)
    P, S = pop.shape[0], sc["ri"].size
    B = P * S
    si = np.tile(np.arange(S), P)
    Pm = np.repeat(pop, S, axis=0)
    T0, T1, ky, kpsi, kr, tla, ffg, gexp, kv = Pm.T
    mu_est, mu_nom, ri, v0 = sc["mu_est"][si], sc["mu_nom"][si], sc["ri"][si], sc["v0"][si]
    nd = sc["nd"][si]
    n = int(round(t_end / dt))

    # ---- controller-side quantities (use the *estimated* friction only)
    d = np.clip(1 - mu_est / MU_DRY, 0, 1)
    vx = v0 * (1 - kv * d)
    T = T0 + T1 * d
    s_g = np.clip(mu_est / MU_DRY, 0.1, 1.1) ** gexp
    ky_e, kp_e, kr_e = ky * s_g, kpsi * s_g, kr * s_g
    # ---- plant-side truth
    mu = mu_nom * (1 - 0.006 * np.maximum(vx - 15, 0) * ri)          # water film: mu falls with speed in rain

    m, Iz, a, b, L = VEH["m"], VEH["Iz"], VEH["a"], VEH["b"], VEH["L"]
    Caf, Car, Fzf, Fzr, Kus = VEH["Caf"], VEH["Car"], VEH["Fzf"], VEH["Fzr"], VEH["Kus"]

    zn = sc["z"][:, si, :]
    ny = zn[:, :, 0] * sc["sig_y"][si]
    npsi = zn[:, :, 1] * sc["sig_psi"][si]
    nr = zn[:, :, 2] * sc["sig_r"][si]

    y = np.zeros(B); psi = np.zeros(B); vy = np.zeros(B); r = np.zeros(B); delta = np.zeros(B)
    Yh = np.zeros((n, B)); PSIh = np.zeros((n, B))
    Yref = np.zeros((n, B)); AY = np.zeros((n, B)); DEL = np.zeros((n, B))
    BETA = np.zeros((n, B)); PSI = np.zeros((n, B))
    idx = np.arange(B)

    for k in range(n):
        t = k * dt
        Yh[k] = y; PSIh[k] = psi
        kk = np.maximum(k - nd, 0)
        ym = Yh[kk, idx] + ny[k]               # delayed + noisy lane-position measurement
        psim = PSIh[kk, idx] + npsi[k]         # delayed + noisy heading measurement
        rm = r + nr[k]                          # IMU yaw rate (no delay)

        tau = np.clip((t - T_START) / T, 0, 1)
        s0, ds0, dds0 = _quintic(tau, T)
        tau_p = np.clip((t + tla - T_START) / T, 0, 1)
        sp, _, _ = _quintic(tau_p, T)
        yref = LANE_W * s0
        psi_ref = np.arctan2(LANE_W * ds0, vx)
        kappa = LANE_W * dds0 / vx ** 2
        e_y = LANE_W * sp - (ym + vx * tla * np.sin(psim))
        e_psi = psi_ref - psim
        d_ff = ffg * kappa * (L + Kus * vx ** 2)
        cmd = np.clip(d_ff + ky_e * e_y + kp_e * e_psi - kr_e * rm, -DELTA_MAX, DELTA_MAX)

        ddot = np.clip((cmd - delta) / TAU_ACT, -DELTA_RATE, DELTA_RATE)
        delta = delta + ddot * dt

        alf = np.arctan2(vy + a * r, vx) - delta
        alr = np.arctan2(vy - b * r, vx)
        Fyf = -mu * Fzf * np.tanh(Caf * alf / (mu * Fzf))
        Fyr = -mu * Fzr * np.tanh(Car * alr / (mu * Fzr))
        ay = (Fyf * np.cos(delta) + Fyr) / m
        vy_dot = ay - vx * r
        r_dot = (a * Fyf * np.cos(delta) - b * Fyr) / Iz
        vy = np.clip(vy + vy_dot * dt, -15, 15)
        r = np.clip(r + r_dot * dt, -3, 3)
        psi = psi + r * dt
        y = y + (vx * np.sin(psi) + vy * np.cos(psi)) * dt

        Yref[k] = yref; AY[k] = ay; DEL[k] = delta
        BETA[k] = np.arctan2(vy, vx); PSI[k] = psi

    # Yh holds y *before* each update; rebuild exact y history for metrics
    Y = Yh
    tt = (np.arange(n) * dt)[:, None]
    mask = (tt >= T_START) & (tt <= T_START + T[None, :] + 1.0)
    cnt = mask.sum(0)
    err = Y - Yref
    rmse = np.sqrt((err ** 2 * mask).sum(0) / cnt)
    maxe = (np.abs(err) * mask).max(0)
    final = np.abs(Y[-int(2 / dt):] - LANE_W).mean(0)
    over = np.maximum(Y.max(0) - LANE_W, 0)
    util = (np.abs(AY) / (mu[None, :] * G)).max(0)
    jerk = np.minimum(np.sqrt(((np.diff(AY, axis=0) / dt) ** 2 * mask[1:]).sum(0) / cnt), 50)
    srate = np.sqrt(((np.diff(DEL, axis=0) / dt) ** 2 * mask[1:]).sum(0) / cnt)
    maxbeta = np.abs(BETA).max(0)
    fail = ((np.abs(PSI).max(0) > 0.5) | (maxbeta > 0.3) | (final > 1.0) | ~np.isfinite(rmse)).astype(float)
    rmse, maxe, final, over = np.minimum(rmse, 5), np.minimum(maxe, 10), np.minimum(final, 5), np.minimum(over, 5)
    out = dict(rmse=rmse, maxe=maxe, final=final, over=over, util=util, jerk=jerk, srate=srate,
               beta=maxbeta, T=T, speed_loss=(v0 - vx) / v0, fail=fail, vx=vx, mu=mu)
    out = {kname: np.nan_to_num(v, nan=1e3, posinf=1e3, neginf=-1e3).reshape(P, S) for kname, v in out.items()}
    if traj:
        out["traj"] = dict(t=np.arange(n) * dt, y=Y, yref=Yref, delta=DEL, ay=AY, mu=mu, vx=vx, T=T, beta=BETA)
    return out


def cost(met, w=W):
    """Scalar per-(individual,scenario) cost from metrics. Lower is better."""
    return (w["rmse"] * met["rmse"] + w["maxe"] * met["maxe"] + w["final"] * met["final"]
            + w["over"] * met["over"] + w["util"] * np.maximum(met["util"] - w["util_thr"], 0)
            + w["jerk"] * met["jerk"] + w["srate"] * met["srate"] + w["T"] * met["T"]
            + w["speed"] * met["speed_loss"] + w["beta"] * met["beta"] + w["fail"] * met["fail"])


def fitness(pop, sc, w=W, worst_weight=0.5):
    """GA fitness: mean cost across scenarios + worst_weight * worst-case scenario (robustness)."""
    met = simulate(pop, sc)
    J = cost(met, w)
    return J.mean(1) + worst_weight * J.max(1), met, J


def unpack(x):
    return {g: float(v) for g, v in zip(GENES, x)}


# ---------------------------------------------------------------- reference controllers
# Hand-tuned, fixed (non-adaptive) controller: gains chosen by coarse manual search on a DRY road.
HAND_TUNED = np.array([3.0, 0.0, 0.04, 0.8, 0.15, 0.8, 1.0, 0.0, 0.0])
