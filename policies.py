"""
policies.py - controllers that plug into plant.run_policy
=========================================================
ParamPolicy : the 9-gene friction-scheduled law evolved by GA/PSO/CMA-ES/NSGA-II (identical to rainlc.simulate)
MPCPolicy   : linear MPC (condensed QP) with Kalman observer; friction-scaled tyre stiffness ("adhesion-aware")
DQNPolicy   : Deep-Q agent (numpy MLP) choosing small steering corrections on top of feed-forward
"""
import numpy as np
from scipy.linalg import expm, solve_discrete_are
import rainlc as R
from rainlc import VEH, MU_DRY, DELTA_MAX, DELTA_RATE, TAU_ACT, DT
from plant import ref_at


def _designer_duration(mu_est, T0, T1):
    d = np.clip(1 - mu_est / MU_DRY, 0, 1)
    return T0 + T1 * d, d


# ======================================================================= GA-type parametric law
class ParamPolicy:
    def __init__(self, x):
        self.x = np.asarray(x, float)

    def setup(self, sc):
        B = sc["ri"].size
        T0, T1, ky, kpsi, kr, tla, ffg, gexp, kv = self.x if self.x.ndim == 1 else self.x.T
        d = np.clip(1 - sc["mu_est"] / MU_DRY, 0, 1)
        self.vx = sc["v0"] * (1 - kv * d)
        self.T = T0 + T1 * d          # scalar genes or per-scenario gene vectors (population batch)
        s = np.clip(sc["mu_est"] / MU_DRY, 0.1, 1.1) ** gexp
        self.ky, self.kp, self.kr = ky * s, kpsi * s, kr * s
        self.tla, self.ffg = tla, ffg

    def act(self, k, t, ym, psim, rm):
        vx, T, tla = self.vx, self.T, self.tla
        yref, psi_ref, kappa = ref_at(t, T, vx)
        yp, _, _ = ref_at(t, T, vx, tla)
        self.yref_now = yref
        e_y = yp - (ym + vx * tla * np.sin(psim))
        e_psi = psi_ref - psim
        d_ff = self.ffg * kappa * (VEH["L"] + VEH["Kus"] * vx ** 2)
        return np.clip(d_ff + self.ky * e_y + self.kp * e_psi - self.kr * rm, -DELTA_MAX, DELTA_MAX)


# ======================================================================= MPC
class MPCPolicy:
    """
    Linear time-invariant MPC on the single-track model (condensed form, N steps of Ts).
      - tyre stiffness scaled with estimated friction:  Ca_eff = Ca * clip(mu_est/mu_dry, .35, 1)   if mu_aware
      - reference duration: designer rule T = T0 + T1*d  (T1=0 and no stiffness scaling for the 'dry' variant)
      - states [y, psi, vy, r] from a steady-state Kalman filter (delayed/noisy y, psi + IMU r); input = steering increment
      - QP is solved without inequality constraints; the steering command is then saturated (angle/rate limits are
        enforced by the actuator model). This is the cheapest MPC variant; see README for the caveat.
    """

    def __init__(self, mu_aware=True, T0=2.8, T1=1.5, N=25, Ts=0.1, q=(1.0, 3.0, 0.3), r_rate=60.0,
                 stiff_exp=1.0, r_rain=0.0):
        self.mu_aware, self.T0, self.T1 = mu_aware, T0, (T1 if mu_aware else 0.0)
        self.N, self.Ts, self.q, self.r_rate = N, Ts, q, r_rate
        self.stiff_exp, self.r_rain = stiff_exp, r_rain      # stiffness scaling exponent; rate-penalty growth with rain

    @staticmethod
    def _model(Ca_f, Ca_r, vx):
        m, Iz, a, b = VEH["m"], VEH["Iz"], VEH["a"], VEH["b"]
        A = np.array([[0, vx, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, -(Ca_f + Ca_r) / (m * vx), (b * Ca_r - a * Ca_f) / (m * vx) - vx],
                      [0, 0, (b * Ca_r - a * Ca_f) / (Iz * vx), -(a * a * Ca_f + b * b * Ca_r) / (Iz * vx)]])
        Bm = np.array([[0], [0], [Ca_f / m], [a * Ca_f / Iz]])
        return A, Bm

    @staticmethod
    def _zoh(A, Bm, Ts):
        n = A.shape[0]
        M = np.zeros((n + 1, n + 1)); M[:n, :n] = A; M[:n, n:] = Bm
        E = expm(M * Ts)
        return E[:n, :n], E[:n, n:]

    def setup(self, sc):
        B = sc["ri"].size
        mu_est = sc["mu_est"]
        self.T, d_fric = _designer_duration(mu_est, self.T0, self.T1)
        d_fric = np.clip(1 - mu_est / MU_DRY, 0, 1)
        self.vx = sc["v0"].copy()
        N, Ts = self.N, self.Ts
        qy, qp, qr = self.q
        Qd = np.diag(np.tile([qy, qp, qr], N))
        Ca = np.array([1, 0, 0, 0.]); Cp = Ca
        Cout = np.zeros((3, 5)); Cout[0, 0] = 1; Cout[1, 1] = 1; Cout[2, 3] = 1
        Cobs = np.zeros((3, 4)); Cobs[0, 0] = 1; Cobs[1, 1] = 1; Cobs[2, 3] = 1
        self.M0 = np.zeros((B, 3 * N)); self.Kx = np.zeros((B, 5))
        self.Ad = np.zeros((B, 4, 4)); self.Bd = np.zeros((B, 4)); self.Kk = np.zeros((B, 4, 3))
        for i in range(B):
            vx = float(self.vx[i])
            s = float(np.clip(mu_est[i] / MU_DRY, 0.35, 1.0)) ** self.stiff_exp if self.mu_aware else 1.0
            A, Bm = self._model(VEH["Caf"] * s, VEH["Car"] * s, vx)
            # --- prediction model (augmented with previous steering)
            Ad_p, Bd_p = self._zoh(A, Bm, Ts)
            Aa = np.zeros((5, 5)); Aa[:4, :4] = Ad_p; Aa[:4, 4] = Bd_p[:, 0]; Aa[4, 4] = 1
            Ba = np.zeros((5, 1)); Ba[:4, 0] = Bd_p[:, 0]; Ba[4, 0] = 1
            Phi = np.zeros((3 * N, 5)); Gam = np.zeros((3 * N, N))
            Apow = [np.eye(5)]
            for j in range(N): Apow.append(Apow[-1] @ Aa)
            CAB = [Cout @ Apow[mm] @ Ba for mm in range(N)]
            for j in range(N):
                Phi[3 * j:3 * j + 3] = Cout @ Apow[j + 1]
                for ii in range(j + 1):
                    Gam[3 * j:3 * j + 3, ii] = CAB[j - ii][:, 0]
            H = Gam.T @ Qd @ Gam + self.r_rate * (1 + self.r_rain * d_fric[i]) * np.eye(N)
            M = np.linalg.solve(H, Gam.T @ Qd)[0]
            self.M0[i] = M; self.Kx[i] = M @ Phi
            # --- Kalman observer at the simulation rate
            Ad_o, Bd_o = self._zoh(A, Bm, DT)
            sy, sp_, sr = sc["sig_y"][i], sc["sig_psi"][i], sc["sig_r"][i]
            Rn = np.diag([sy ** 2, sp_ ** 2, sr ** 2])
            Qn = np.diag([1e-6, 1e-7, 2e-4, 2e-5])
            P = solve_discrete_are(Ad_o.T, Cobs.T, Qn, Rn)
            K = P @ Cobs.T @ np.linalg.inv(Cobs @ P @ Cobs.T + Rn)
            self.Ad[i], self.Bd[i], self.Kk[i] = Ad_o, Bd_o[:, 0], K
        self.Cobs = Cobs
        self.xh = np.zeros((B, 4)); self.dh = np.zeros(B); self.B = B

    def act(self, k, t, ym, psim, rm):
        vx, T, Ts, N = self.vx, self.T, self.Ts, self.N
        # observer
        xp = np.einsum("bij,bj->bi", self.Ad, self.xh) + self.Bd * self.dh[:, None]
        z = np.stack([ym, psim, rm], 1)
        xh = xp + np.einsum("bij,bj->bi", self.Kk, z - xp[:, [0, 1, 3]])
        self.xh = xh
        # reference stack [y, psi, r] over the horizon
        refs = np.zeros((self.B, 3 * N))
        for j in range(N):
            yr, pr, kap = ref_at(t, T, vx, (j + 1) * Ts)
            refs[:, 3 * j] = yr; refs[:, 3 * j + 1] = pr; refs[:, 3 * j + 2] = kap * vx
        self.yref_now = ref_at(t, T, vx)[0]
        z0 = np.concatenate([xh, self.dh[:, None]], 1)
        d_inc = np.einsum("bj,bj->b", self.M0, refs) - np.einsum("bj,bj->b", self.Kx, z0)
        cmd = np.clip(self.dh + d_inc, -DELTA_MAX, DELTA_MAX)
        # own actuator model for the observer input
        self.dh = self.dh + np.clip((cmd - self.dh) / TAU_ACT, -DELTA_RATE, DELTA_RATE) * DT
        return cmd


# ======================================================================= DQN
ACTIONS = np.array([-0.012, -0.005, -0.002, 0.0, 0.002, 0.005, 0.012])   # steering-correction increments per decision [rad]
DECISION_EVERY = 5                                                         # 20 Hz decisions on a 100 Hz plant
U_MAX = 0.12
OBS_DIM = 11


class DQNPolicy:
    """Policy = argmax_a Q(obs, a). Steering = feed-forward + integrated correction u (u += ACTIONS[a])."""

    def __init__(self, qnet=None, T0=2.8, T1=1.5, eps=0.0, rng=None):
        self.q, self.T0, self.T1, self.eps = qnet, T0, T1, eps
        self.rng = rng or np.random.default_rng(0)

    def setup(self, sc):
        B = sc["ri"].size
        self.B = B
        self.mu_est = sc["mu_est"]
        self.T, _ = _designer_duration(sc["mu_est"], self.T0, self.T1)
        self.vx = sc["v0"].copy()
        self.u = np.zeros(B); self.dh = np.zeros(B); self.cmd = np.zeros(B)
        self.last_a = np.full(B, 3)

    def ff(self, t):
        _, _, kap = ref_at(t, self.T, self.vx)
        return kap * (VEH["L"] + VEH["Kus"] * self.vx ** 2)

    def obs(self, t, ym, psim, rm):
        vx, T = self.vx, self.T
        yref, psi_ref, _ = ref_at(t, T, vx)
        yp, _, _ = ref_at(t, T, vx, 0.8)
        ff_f = self.ff(t + 0.5) if False else (ref_at(t, T, vx, 0.5)[2] * (VEH["L"] + VEH["Kus"] * vx ** 2))
        ph = np.clip((t - R.T_START) / T, 0, 1)
        return np.stack([(yp - (ym + vx * 0.8 * np.sin(psim))) / 1.0, (yref - ym) / 0.5, (psi_ref - psim) / 0.1,
                         rm / 0.3, self.dh / 0.2, self.u / 0.1, self.ff(t) / 0.1, self.mu_est / 0.9, vx / 30.0,
                         ff_f / 0.1, ph], 1)

    def apply(self, a):
        self.u = np.clip(self.u + ACTIONS[a], -U_MAX, U_MAX)

    def command(self, t):
        return np.clip(self.ff(t) + self.u, -DELTA_MAX, DELTA_MAX)

    def act(self, k, t, ym, psim, rm):
        self.yref_now = ref_at(t, self.T, self.vx)[0]
        if k % DECISION_EVERY == 0:
            o = self.obs(t, ym, psim, rm)
            a = np.argmax(self.q.forward(o), 1)
            if self.eps > 0:
                rnd = self.rng.random(self.B) < self.eps
                a = np.where(rnd, self.rng.integers(0, len(ACTIONS), self.B), a)
            self.apply(a)
        cmd = self.command(t)
        self.dh = self.dh + np.clip((cmd - self.dh) / TAU_ACT, -DELTA_RATE, DELTA_RATE) * DT
        return cmd
