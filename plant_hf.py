"""
plant_hf.py - higher-fidelity vehicle plant for sim-to-sim validation (CPU-only stand-in for CARLA / CarMaker)
=============================================================================================================
Structurally different from the single-track model the controllers were designed / trained on:

  * four wheels with individual vertical loads
  * roll DOF (sprung mass, roll stiffness/damping, roll-centre heights) -> dynamic lateral load transfer
  * longitudinal load transfer from the speed-hold controller / drag
  * load-sensitive tyres: Magic-Formula lateral force with load-dependent cornering stiffness and peak friction,
    post-peak force drop, friction-ellipse combined slip with the longitudinal force, tyre relaxation (lag)
  * steering: ratio error, actuator lag, roll steer, lateral-force compliance steer
  * longitudinal speed controller (the earlier models held speed constant), drag and rolling resistance
  * wet road: water-film depth h(ri); friction falls with speed (water film) and with a hydroplaning onset
    speed that drops with water depth (Horne-style sigmoid, up to -25 % grip)
  * unmodelled disturbances: road-crown / crosswind lateral acceleration that grows with rain
  * all parameters drawn from distributions that are NOT the ones used for training (plant.make_mismatch)

It is NOT a validated model of any specific car and the wet-road laws are assumed, not calibrated to data. It is an
independent, more complex plant - a stronger test than parameter perturbations of the same model - but it is
not a substitute for CARLA / CarMaker.

Interface: drop-in replacement for plant.Plant   ->   plant.run_policy(policy, sc, mismatch=make_hf_params(...), plant_cls=Plant14)
"""
import numpy as np
import rainlc as R
import plant as P
from rainlc import G, VEH, DELTA_MAX, DELTA_RATE

SGN_Y = np.array([1.0, -1.0, 1.0, -1.0])      # wheel order FL, FR, RL, RR ; y > 0 is LEFT
FRONT = np.array([1.0, 1.0, 0.0, 0.0])
L_WB = VEH["L"]
TRACK = 1.55
RHO_AIR, CD_A = 1.2, 0.66                     # 0.5*rho*Cd*A = 0.40 N/(m/s)^2
CRR = 0.012


def make_hf_params(B, rng, level=1.0):
    """Random plant parameters. level scales the deviation from nominal (0 = nominal vehicle, still the HF structure)."""
    lv = level
    U = lambda lo, hi: rng.uniform(lo, hi, B)
    m = 1500.0 * (1 + lv * U(-0.05, 0.25))
    a = 1.1 * (1 + lv * U(-0.06, 0.08))
    h_cg = 0.55 * (1 + lv * U(-0.10, 0.15))
    hrf, hrr = 0.08, 0.14
    hs = h_cg - ((L_WB - a) * hrf + a * hrr) / L_WB
    return dict(
        m=m, Iz=2500.0 * (m / 1500.0) * (1 + lv * U(-0.10, 0.10)), a=a, h_cg=h_cg, hs=hs,
        Ix=500.0 * (m / 1500.0), Kphi=60000.0 * (1 + lv * U(-0.2, 0.2)), lam_f=0.55 + lv * U(-0.08, 0.08),
        Cphi=6000.0 * (1 + lv * U(-0.3, 0.3)),
        Caf=VEH["Caf"] * (1 + lv * U(-0.2, 0.2)), Car=VEH["Car"] * (1 + lv * U(-0.2, 0.2)),
        mu_scale=1 + lv * U(-0.10, 0.05), Cmf=1.4 + lv * U(-0.1, 0.2), Emf=-0.6 + lv * U(-0.4, 0.4),
        sigma=0.8 * (1 + lv * U(-0.4, 0.5)), sgain=1 + lv * U(-0.06, 0.06),
        kc=0.6e-6 * (1 + lv * U(-0.5, 1.0)), eps=0.04 * (1 + lv * U(-0.5, 0.5)),
        tau=0.08 * (1 + lv * U(-0.3, 0.6)), water=1 + lv * U(-0.3, 0.3),
        dist_z=rng.standard_normal(B) * lv,
    )


def wet_friction(ri, vx, mu_scale, water):
    """Peak friction of the wet road at speed vx. (all arrays, shape (B,))"""
    h = np.where(ri > 0, 0.4 + 2.2 * ri, 0.0) * water                     # water-film depth [mm]
    base = 0.9 * (1 - 0.58 * np.power(ri, 0.75)) * mu_scale
    fv = 1 - 0.07 * h * np.clip(vx - 10, 0, None) / 30.0                  # water-film speed effect
    vhp = 31.0 - 2.0 * h                                                  # hydroplaning onset speed [m/s]
    hf = 1 - 0.25 * (h > 0.3) / (1 + np.exp(-(vx - vhp) / 2.5))
    return base * fv * hf


class Plant14(P.Plant):
    """Subclass of plant.Plant: re-uses sensing (delay/noise) and the metric definitions; replaces the dynamics."""

    def __init__(self, sc, n, vx_target, hf, dt=R.DT):
        if hf is None: raise ValueError("Plant14 needs hf parameters (plant_hf.make_hf_params)")
        super().__init__(sc, n, vx_target, None, dt)
        p = self.p = hf
        B = self.B
        self.vt = np.asarray(vx_target, float)
        self.ri = sc["ri"]
        self.mu = wet_friction(self.ri, self.vt, p["mu_scale"], p["water"])      # peak friction at target speed (for metrics)
        self.b = L_WB - p["a"]
        self.Fzn_f = p["m"] * G * self.b / (2 * L_WB)
        self.Fzn_r = p["m"] * G * p["a"] / (2 * L_WB)
        self.xw = np.stack([p["a"], p["a"], -self.b, -self.b], 1)               # (B,4)
        self.yw = np.broadcast_to(SGN_Y * TRACK / 2, (B, 4))
        self.vxa = self.vt.copy()                                               # actual forward speed
        self.phi = np.zeros(B); self.phid = np.zeros(B)
        self.Fy = np.zeros((B, 4)); self.Fyf_tot = np.zeros(B)
        self.ax_prev = np.zeros(B); self.ay_prev = np.zeros(B)
        self.a_dist = p["dist_z"] * (0.08 + 0.20 * self.ri)                     # [m/s^2] left-positive
        self.PHI = np.zeros((n, B)); self.VX = np.zeros((n, B)); self.FZ = np.zeros((n, B, 4))

    # ------------------------------------------------------------------------------------------
    def step(self, cmd, yref=None):
        k, dt, p = self.k, self.dt, self.p
        m = p["m"]
        cmd = np.clip(cmd, -DELTA_MAX, DELTA_MAX)
        ddot = np.clip((cmd - self.delta) / p["tau"], -DELTA_RATE, DELTA_RATE)
        self.delta = self.delta + ddot * dt
        d_eff = self.delta * p["sgain"] - p["eps"] * self.phi - p["kc"] * self.Fyf_tot
        dw = d_eff[:, None] * FRONT                                                # (B,4) wheel steer angles
        cd, sd = np.cos(dw), np.sin(dw)

        # ---- vertical loads: static + lateral (roll DOF + roll-centre) + longitudinal transfer
        Fz_stat = np.stack([self.Fzn_f, self.Fzn_f, self.Fzn_r, self.Fzn_r], 1)
        dFz_f = ((p["Kphi"] * self.phi + p["Cphi"] * self.phid) * p["lam_f"] + m * self.ay_prev * (self.b / L_WB) * 0.08) / TRACK
        dFz_r = ((p["Kphi"] * self.phi + p["Cphi"] * self.phid) * (1 - p["lam_f"]) + m * self.ay_prev * (p["a"] / L_WB) * 0.14) / TRACK
        dFz_lat = np.stack([-dFz_f, dFz_f, -dFz_r, dFz_r], 1)                       # right wheels (y<0) gain when phi>0
        dFz_lon = m * self.ax_prev * p["h_cg"] / L_WB
        dFz_lon = np.stack([-dFz_lon, -dFz_lon, dFz_lon, dFz_lon], 1) / 2.0
        Fz = np.maximum(Fz_stat + dFz_lat + dFz_lon, 100.0)

        # ---- wheel kinematics
        vxw = self.vxa[:, None] - self.r[:, None] * self.yw
        vyw = self.vy[:, None] + self.r[:, None] * self.xw
        u = vxw * cd + vyw * sd
        v = -vxw * sd + vyw * cd
        alpha = np.arctan2(v, np.maximum(u, 1.0))

        # ---- tyres
        mu_t = wet_friction(self.ri, self.vxa, p["mu_scale"], p["water"])
        xr = Fz / np.stack([self.Fzn_f, self.Fzn_f, self.Fzn_r, self.Fzn_r], 1)
        Ca_ax = np.stack([p["Caf"], p["Caf"], p["Car"], p["Car"]], 1) / 2.0
        Ca = Ca_ax * np.maximum(xr * (1 - 0.3 * (xr - 1)), 0.1)                     # load-sensitive stiffness
        mu_w = mu_t[:, None] * np.maximum(1 - 0.10 * (xr - 1), 0.5)
        D = mu_w * Fz
        Cm, Em = p["Cmf"][:, None], p["Emf"][:, None]
        xb = (Ca / (Cm * D)) * alpha
        Fy0 = -D * np.sin(Cm * np.arctan(xb - Em * (xb - np.arctan(xb))))
        # longitudinal: speed controller demand shared by vertical load, friction-limited, friction-ellipse coupling
        drag = 0.5 * RHO_AIR * CD_A * self.vxa ** 2 + CRR * m * G
        Fdem = np.clip(m * 1.0 * (self.vt - self.vxa) + drag, -6000.0, 3500.0)
        Fx = Fdem[:, None] * Fz / Fz.sum(1, keepdims=True)
        lim = 0.95 * D
        Fx = np.clip(Fx, -lim, lim)
        Fy_ss = Fy0 * np.sqrt(np.maximum(1 - (Fx / D) ** 2, 0.04))
        lam = np.minimum(np.maximum(u, 3.0) * dt / p["sigma"][:, None], 1.0)       # tyre relaxation
        self.Fy = self.Fy + (Fy_ss - self.Fy) * lam
        Fy = self.Fy

        # ---- body forces / moments
        Fxb = Fx * cd - Fy * sd
        Fyb = Fx * sd + Fy * cd
        SFx = Fxb.sum(1) - drag
        SFy_tyre = Fyb.sum(1)
        SFy = SFy_tyre + m * self.a_dist
        Mz = (self.xw * Fyb - self.yw * Fxb).sum(1)
        ax_b = SFx / m
        ay_tot = SFy / m
        # roll (semi-implicit Euler)
        ms = 0.9 * m
        phidd = (ms * ay_tot * p["hs"] + ms * G * p["hs"] * self.phi - p["Kphi"] * self.phi - p["Cphi"] * self.phid) / p["Ix"]
        self.phid = self.phid + phidd * dt
        self.phi = np.clip(self.phi + self.phid * dt, -0.3, 0.3)
        # rigid-body states
        vxd = ax_b + self.vy * self.r
        vyd = ay_tot - self.vxa * self.r
        rd = Mz / p["Iz"]
        self.vxa = np.maximum(self.vxa + vxd * dt, 5.0)
        self.vy = np.clip(self.vy + vyd * dt, -15, 15)
        self.r = np.clip(self.r + rd * dt, -3, 3)
        self.psi = self.psi + self.r * dt
        self.y = self.y + (self.vxa * np.sin(self.psi) + self.vy * np.cos(self.psi)) * dt
        self.ax_prev, self.ay_prev = ax_b, ay_tot
        self.Fyf_tot = Fy[:, 0] + Fy[:, 1]

        ay_log = SFy_tyre / m
        if yref is not None: self.Yref[k] = yref
        self.AY[k] = ay_log; self.DEL[k] = self.delta
        self.BETA[k] = np.arctan2(self.vy, self.vxa); self.PSI[k] = self.psi
        self.PHI[k] = self.phi; self.VX[k] = self.vxa; self.FZ[k] = Fz
        return ay_log
