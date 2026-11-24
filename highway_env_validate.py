"""
highway_env_validate.py - Validate rain-aware lane-change controllers on Farama highway-env
============================================================================================
Drop-in replacement / companion to `carla_validate.py` that runs your exact controllers
(`GA-RainAware`, `GA-RainAware-DR`, `GA-Fixed-DR`, `Hand-tuned`, `MPC`, `MPC-adhesion`, `DQN`)
inside `highway-env` (`highway-v0` with `ContinuousAction` and a 3.5 m straight multi-lane road).

Why highway-env fits cleanly:
  * Runs fast on CPU (works natively on Apple Silicon M4 and Windows without a discrete GPU).
  * Matches your `LANE_W = 3.5` m lane-change geometry directly via `StraightLane(..., width=3.5)`.
  * Supports `simulation_frequency = 100` Hz (`dt = 0.01` s) and `policy_frequency = 100` Hz so your
    controllers, sensor delays (`nd`), and Kalman/MPC loops execute at their exact design rate.
  * Injects wet-road friction limits (`mu_true`) onto the vehicle's lateral dynamics (since stock
    `highway-env` uses a kinematic bicycle model without tire-road adhesion saturation) plus the
    exact sensor noise and perception delay model from `rainlc.make_scenarios`.

Installation:
  pip install gymnasium highway-env numpy scipy

Usage:
  # 1. Offline pipeline check (verifies identical metrics to plant.run_policy without highway-env):
  python highway_env_validate.py --selftest

  # 2. Validate GA-RainAware-DR across rain intensities and speeds:
  python highway_env_validate.py --controller GA-RainAware-DR \
         --params-file results/best_parameters_dr.json \
         --rain 0.0 0.25 0.5 0.75 1.0 --speed 20 25 30 --reps 10 \
         --out results/highway_env_ga_dr.csv

  # 3. Validate with visual rendering window:
  python highway_env_validate.py --controller GA-RainAware-DR \
         --params-file results/best_parameters_dr.json --render --reps 1

  # 4. Validate all controllers and write a comparison summary table:
  python highway_env_validate.py --all --out results/highway_env_all.csv
"""
import argparse
import csv
import json
import pickle
import sys
import numpy as np

import rainlc as R
import plant as P
import policies as PL


# ======================================================================================
# 1. Adapters
# ======================================================================================
class SimAdapter:
    """Backed by plant.Plant - used for offline self-test of the validation harness."""

    def __init__(self, sc, n, vx, mismatch=None):
        self.pl = P.Plant(sc, n, vx, mismatch)
        self.k = 0
        self.mu = self.pl.mu

    def true_state(self, k):
        self.pl.measure(k)
        return dict(y=self.pl.y.copy(), psi=self.pl.psi.copy(), r=self.pl.r.copy())

    def apply_and_tick(self, cmd, yref):
        self.pl.step(cmd, yref)
        k = self.pl.k
        return dict(
            ay=self.pl.AY[k].copy(),
            beta=self.pl.BETA[k].copy(),
            psi=self.pl.PSI[k].copy(),
            delta=self.pl.DEL[k].copy(),
        )

    def close(self):
        pass


class HighwayEnvAdapter:
    """
    Adapter wrapping Farama `highway-env` (`highway-v0`) at 100 Hz (dt = 0.01 s).

    Coordinate convention alignment:
      * Your model (`rainlc` / `plant`): x forward, y LEFT (+3.5 m = left lane), psi/delta positive LEFT.
      * `highway-env`: x forward, y RIGHT (lane 0 is y=0, lane 1 is y=+3.5 m to the right), heading clockwise-positive.
      * Therefore, changing from right lane (lane 1, y_hw = +3.5) to left lane (lane 0, y_hw = 0.0)
        corresponds to y_model = y0_hw - y_hw in [0, +3.5] m, with psi_model = -heading_hw and
        steering_hw = -delta_model.

    Physics enhancement:
      * Standard `highway-env` updates vehicles using a kinematic bicycle model without tire friction saturation.
      * To faithfully test rain-induced adhesion loss (`mu_true`), this adapter applies actuator lag (`TAU_ACT`),
        steering rate limits (`DELTA_RATE`), and lateral tire-grip saturation (`|ay| <= mu_true * g`) on top of
        the `highway-env` vehicle state while letting `highway-env` manage road geometry, surrounding traffic,
        collision checking, and 2D rendering.
    """

    def __init__(self, dt=R.DT, render=False, vehicles_count=0):
        import gymnasium as gym
        import highway_env  # noqa: F401 - registers highway-v0

        self.dt = dt
        self.render_mode = "human" if render else None
        self.vehicles_count = vehicles_count
        self.env = gym.make(
            "highway-v0",
            render_mode=self.render_mode,
            config={
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True,
                    "steering_range": (-R.DELTA_MAX, R.DELTA_MAX),
                    "acceleration_range": (-5.0, 5.0),
                },
                "lanes_count": 2,
                "vehicles_count": vehicles_count,
                "simulation_frequency": int(round(1.0 / dt)),
                "policy_frequency": int(round(1.0 / dt)),
                "duration": int(np.ceil(R.T_END)) + 2,
                "initial_lane_id": 1,  # start in right lane (index 1) and change left to lane 0
                "centering_position": [0.3, 0.5],
                "scaling": 5.5,
                "offscreen_rendering": False,
            },
        )
        self.ego = None
        self.y0_hw = 3.5
        self.v0 = 25.0
        self.mu_true = R.MU_DRY
        self.delta_act = 0.0
        self.r_est = 0.0
        self.prev_psi = 0.0
        self.prev_vy = 0.0
        self.crashed = False

    def reset(self, ri, mu_true, v0):
        self.env.reset(seed=int(round(1000 * ri + v0)))
        unwrapped = self.env.unwrapped

        # Force exact 3.5 m lane width on both lanes to match rainlc.LANE_W
        from highway_env.road.lane import StraightLane
        from highway_env.road.road import Road, RoadNetwork

        net = RoadNetwork()
        for lane_idx in range(2):
            y_center = lane_idx * R.LANE_W
            net.add_lane(
                "0",
                "1",
                StraightLane(
                    [0.0, y_center],
                    [2000.0, y_center],
                    width=R.LANE_W,
                    speed_limit=40.0,
                ),
            )
        unwrapped.road = Road(
            network=net,
            np_random=unwrapped.np_random,
            record_history=False,
        )

        # Place ego cleanly in lane 1 (y_hw = +3.5 m) heading 0 rad at target speed v0
        self.ego = unwrapped.vehicle
        self.ego.road = unwrapped.road
        self.ego.position = np.array([50.0, R.LANE_W], dtype=float)
        self.ego.heading = 0.0
        self.ego.speed = float(v0)
        self.ego.lane_index = ("0", "1", 1)
        self.ego.lane = net.get_lane(self.ego.lane_index)
        unwrapped.road.vehicles = [self.ego]

        # Optionally spawn non-interfering traffic far ahead/behind if requested
        if self.vehicles_count > 0:
            from highway_env.vehicle.behavior import IDMVehicle
            for idx in range(self.vehicles_count):
                lane_id = idx % 2
                x_pos = 180.0 + idx * 65.0
                veh = IDMVehicle(
                    unwrapped.road,
                    position=[x_pos, lane_id * R.LANE_W],
                    heading=0.0,
                    speed=float(v0) * 0.95,
                )
                unwrapped.road.vehicles.append(veh)

        self.y0_hw = float(self.ego.position[1])
        self.v0 = float(v0)
        self.mu_true = float(mu_true)
        self.delta_act = 0.0
        self.r_est = 0.0
        self.prev_psi = 0.0
        self.prev_vy = 0.0
        self.crashed = False

    def true_state(self, k):
        # Convert highway-env coordinates (y right, heading CW+) to model frame (y left, psi CCW+)
        y_model = self.y0_hw - float(self.ego.position[1])
        psi_model = -float(self.ego.heading)
        return dict(
            y=np.array([y_model]),
            psi=np.array([psi_model]),
            r=np.array([self.r_est]),
        )

    def apply_and_tick(self, cmd, yref):
        # 1. Steering actuator lag and rate limit (identical to plant.Plant)
        cmd_val = float(np.clip(cmd[0], -R.DELTA_MAX, R.DELTA_MAX))
        ddot = np.clip((cmd_val - self.delta_act) / R.TAU_ACT, -R.DELTA_RATE, R.DELTA_RATE)
        self.delta_act += ddot * self.dt

        # 2. Check tire-road friction limit on desired lateral acceleration
        # Kinematic yaw rate r_kin = (v / L) * tan(delta)
        v_cur = max(float(self.ego.speed), 1.0)
        L = R.VEH["L"]
        r_kin = (v_cur / L) * np.tan(self.delta_act)
        ay_req = v_cur * r_kin
        ay_max = self.mu_true * R.G

        # Saturate effective steering if demanded lateral force exceeds wet tire grip
        if abs(ay_req) > ay_max:
            ay_eff = np.sign(ay_req) * ay_max
            r_eff = ay_eff / v_cur
            delta_eff = np.arctan((r_eff * L) / v_cur)
        else:
            delta_eff = self.delta_act

        # 3. Longitudinal speed-hold proportional controller
        acc_cmd = float(np.clip(1.5 * (self.v0 - v_cur), -5.0, 5.0))

        # 4. Step highway-env with ContinuousAction = [normalized_accel, normalized_steer]
        # Note: highway-env ContinuousAction expects [accel_norm, steer_norm] in [-1, 1]
        # and uses steering positive = right, so we flip sign (-delta_eff).
        accel_norm = float(np.clip(acc_cmd / 5.0, -1.0, 1.0))
        steer_norm = float(np.clip(-delta_eff / R.DELTA_MAX, -1.0, 1.0))

        psi_before = -float(self.ego.heading)
        _, _, terminated, truncated, info = self.env.step(np.array([accel_norm, steer_norm], dtype=np.float32))
        if self.render_mode == "human":
            self.env.render()

        psi_after = -float(self.ego.heading)
        self.r_est = (psi_after - psi_before) / self.dt
        v_after = max(float(self.ego.speed), 1.0)
        ay = v_after * self.r_est
        beta = np.arctan(R.VEH["b"] / L * np.tan(delta_eff))
        if info.get("crashed", False) or terminated:
            self.crashed = True

        return dict(
            ay=np.array([ay]),
            beta=np.array([beta]),
            psi=np.array([psi_after]),
            delta=np.array([self.delta_act]),
        )

    def close(self):
        self.env.close()


# ======================================================================================
# 2. Adapter-Agnostic Episode Runner (reuses exact Plant.metrics definitions)
# ======================================================================================
class _Logs:
    pass


def run_episode(adapter, policy, sc, mu_true, vx, dt=R.DT, t_end=R.T_END):
    """Closed-loop rollout for a single scenario (B=1) with delayed/noisy measurements."""
    n = int(round(t_end / dt))
    policy.setup(sc)
    zn = sc["z"]
    ny = zn[:, :, 0] * sc["sig_y"]
    npsi = zn[:, :, 1] * sc["sig_psi"]
    nr = zn[:, :, 2] * sc["sig_r"]
    nd = sc["nd"]

    logs = _Logs()
    logs.dt, logs.n = dt, n
    logs.Yh = np.zeros((n, 1))
    logs.Yref = np.zeros((n, 1))
    logs.AY = np.zeros((n, 1))
    logs.DEL = np.zeros((n, 1))
    logs.BETA = np.zeros((n, 1))
    logs.PSI = np.zeros((n, 1))
    logs.mu = np.asarray(mu_true, float).reshape(1)
    logs.vx = np.asarray(vx, float).reshape(1)
    PSIh = np.zeros((n, 1))

    for k in range(n):
        s = adapter.true_state(k)
        logs.Yh[k] = s["y"]
        PSIh[k] = s["psi"]
        kk = max(k - int(nd[0]), 0)
        ym = logs.Yh[kk] + ny[k]
        psim = PSIh[kk] + npsi[k]
        rm = s["r"] + nr[k]
        cmd = policy.act(k, k * dt, ym, psim, rm)
        out = adapter.apply_and_tick(cmd, policy.yref_now)
        logs.Yref[k] = policy.yref_now
        logs.AY[k] = out["ay"]
        logs.DEL[k] = out["delta"]
        logs.BETA[k] = out["beta"]
        logs.PSI[k] = out["psi"]

    met = P.Plant.metrics(logs, policy.T, sc["v0"])
    if getattr(adapter, "crashed", False):
        met["fail"][0, 0] = 1.0
    return met


# ======================================================================================
# 3. Policy Loader (supports GA variants, Hand-tuned, MPC, MPC-adhesion, and DQN)
# ======================================================================================
def get_policy(name, params_files=None, tuned_path="results/mpc_tuned.json", dqn_path="results/dqn_seed0.pkl"):
    if params_files is None:
        params_files = [
            "results/best_parameters_dr.json",
            "results/best_parameters.json",
            "results/best_parameters_hf.json",
        ]
    for pf in params_files:
        try:
            with open(pf, "r") as f:
                prm = json.load(f)
            if name in prm:
                return PL.ParamPolicy(np.array(list(prm[name].values()), dtype=float))
        except FileNotFoundError:
            continue

    if name in ("MPC", "MPC-adhesion"):
        with open(tuned_path, "r") as f:
            tm = json.load(f)[name]
        return PL.MPCPolicy(
            mu_aware=(name == "MPC-adhesion"),
            T0=tm["T0"],
            T1=tm["T1"],
            r_rate=tm["r_rate"],
            stiff_exp=tm["stiff_exp"],
        )

    if name == "DQN":
        import dqn as DQN_MOD
        with open(dqn_path, "rb") as f:
            raw = pickle.load(f)
        if isinstance(raw, dict):
            qnet = DQN_MOD.QNet(raw["sizes"])
            qnet.W, qnet.b = raw["W"], raw["b"]
        else:
            qnet = raw
        return PL.DQNPolicy(qnet)

    raise SystemExit(f"Unknown controller '{name}'. Available JSON/PKL entries not found.")


# ======================================================================================
# 4. Self-Test & CLI Validation Entrypoint
# ======================================================================================
def selftest():
    rng = np.random.default_rng(1)
    sc = R.make_scenarios([0.1, 0.5, 0.9], [22.0, 26.0, 30.0], rng, mu_err_std=0.1)
    worst = 0.0
    for name in ["GA-RainAware", "Hand-tuned", "MPC"]:
        ref = P.cost_of(P.run_policy(get_policy(name), sc))
        got = []
        for i in range(3):
            si = {k: (v[i : i + 1] if k != "z" else v[:, i : i + 1, :]) for k, v in sc.items()}
            pol = get_policy(name)
            pol.setup(si)
            ad = SimAdapter(si, int(round(R.T_END / R.DT)), pol.vx)
            got.append(P.cost_of(run_episode(ad, get_policy(name), si, ad.mu, pol.vx))[0])
        diff = np.abs(np.array(got) - ref).max()
        worst = max(worst, diff)
        print(f"{name:15s} J(plant)={np.round(ref, 3)}  J(harness)={np.round(got, 3)}  max|diff|={diff:.2e}")
    print("SELFTEST", "PASSED" if worst < 1e-8 else "FAILED")
    return worst < 1e-8


def evaluate_controller(adapter, ctrl_name, rains, speeds, reps, mu_err_std):
    rows = []
    for ri in rains:
        for v0 in speeds:
            for rep in range(reps):
                seed = 1000 * rep + int(round(100 * ri)) + int(round(v0))
                sc = R.make_scenarios([ri], [v0], np.random.default_rng(seed), mu_err_std=mu_err_std)
                pol = get_policy(ctrl_name)
                pol.setup(sc)
                # Speed-dependent water-film friction reduction matching rainlc.simulate
                mu_true = float(sc["mu_nom"][0] * (1.0 - 0.006 * max(float(pol.vx[0]) - 15.0, 0.0) * ri))
                adapter.reset(ri, mu_true, float(pol.vx[0]))
                met = run_episode(adapter, get_policy(ctrl_name), sc, mu_true, pol.vx)
                J = float(P.cost_of(met)[0])
                row = dict(
                    controller=ctrl_name,
                    rain=float(ri),
                    v0=float(v0),
                    rep=int(rep),
                    J=round(J, 4),
                    rmse=round(float(met["rmse"][0, 0]), 4),
                    maxe=round(float(met["maxe"][0, 0]), 4),
                    final=round(float(met["final"][0, 0]), 4),
                    util=round(float(met["util"][0, 0]), 4),
                    fail=int(met["fail"][0, 0]),
                )
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Validate rain-aware lane-change controllers in highway-env.")
    ap.add_argument("--selftest", action="store_true", help="Run offline equivalence check without highway-env")
    ap.add_argument("--controller", default="GA-RainAware-DR", help="Controller name from results/*.json")
    ap.add_argument("--all", action="store_true", help="Evaluate all main controllers for comparison")
    ap.add_argument("--rain", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--speed", type=float, nargs="+", default=[20.0, 25.0, 30.0])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mu-err", type=float, default=0.10)
    ap.add_argument("--traffic", type=int, default=0, help="Number of surrounding IDM vehicles")
    ap.add_argument("--render", action="store_true", help="Render highway-env 2D window")
    ap.add_argument("--out", default="highway_env_results.csv")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    controllers = (
        ["GA-RainAware-DR", "GA-Fixed-DR", "GA-RainAware", "Hand-tuned", "MPC", "DQN"]
        if args.all
        else [args.controller]
    )

    ad = HighwayEnvAdapter(dt=R.DT, render=args.render, vehicles_count=args.traffic)
    all_rows = []
    try:
        for ctrl in controllers:
            print(f"--- Evaluating {ctrl} on highway-env ---")
            rows = evaluate_controller(ad, ctrl, args.rain, args.speed, args.reps, args.mu_err)
            all_rows.extend(rows)
            mean_J = np.mean([r["J"] for r in rows])
            mean_rmse = np.mean([r["rmse"] for r in rows])
            fail_pct = 100.0 * np.mean([r["fail"] for r in rows])
            print(f"Summary [{ctrl}]: Mean J = {mean_J:.3f} | Mean RMSE = {mean_rmse:.4f} m | Fail = {fail_pct:.1f}%")
    finally:
        ad.close()
        if all_rows:
            with open(args.out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"Saved {len(all_rows)} episodes to {args.out}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {len(all_rows)} episodes to {args.out}")


if __name__ == "__main__":
    main()
