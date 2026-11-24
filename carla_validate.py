"""
carla_validate.py - run the lane-change controllers in CARLA (0.9.x, synchronous mode) under simulated rain
============================================================================================================
STATUS: the controller/measurement/metrics pipeline is verified offline (`--selftest` reproduces plant.run_policy
exactly). The `CarlaAdapter` class itself is NOT tested - CARLA could not be installed where this was written.
Expect to fix small API/geometry details on first contact with a real server (spawn point, steering sign, friction scale).

Usage
  python carla_validate.py --selftest                                  # no CARLA needed
  python carla_validate.py --host localhost --port 2000 \
         --controller GA-RainAware --rain 0 0.25 0.5 0.75 1.0 --speed 20 25 --reps 5 --out carla_results.csv

Conventions: model frame = x forward, y LEFT, psi/steering positive to the LEFT. CARLA is left-handed (y right, yaw
clockwise-positive), so lateral position, yaw, yaw rate, lateral acceleration and steering are sign-flipped below.

Rain -> CARLA mapping (ASSUMPTIONS, calibrate before publishing):
  * visual weather:   precipitation / deposits / wetness = 100*ri
  * tyre friction:    CARLA's wheel.tire_friction (default ~3.5) is scaled by mu(ri)/mu_dry. CARLA's value is a friction
                      *scale*, not a physical mu - run a calibration sweep (steady-state cornering) to map it.
  * perception:       CARLA's camera is not used; measurement delay/noise growing with ri is injected exactly as in the
                      simulator (same random numbers), so only the vehicle dynamics differ between the two environments.
"""
import argparse, csv, json, sys, time
import numpy as np
import rainlc as R
import plant as P
import policies as PL


# ====================================================================================== adapters
class SimAdapter:
    """Backed by plant.Plant - used for the offline self-test of the pipeline."""
    def __init__(self, sc, n, vx, mismatch=None):
        self.pl = P.Plant(sc, n, vx, mismatch)
        self.k = 0
        self.mu = self.pl.mu

    def true_state(self, k):
        self.pl.measure(k)                          # logs history exactly as run_policy does
        return dict(y=self.pl.y.copy(), psi=self.pl.psi.copy(), r=self.pl.r.copy())

    def apply_and_tick(self, cmd, yref):
        self.pl.step(cmd, yref)
        k = self.pl.k
        return dict(ay=self.pl.AY[k].copy(), beta=self.pl.BETA[k].copy(), psi=self.pl.PSI[k].copy(), delta=self.pl.DEL[k].copy())


class CarlaAdapter:
    """CARLA 0.9.x adapter (synchronous mode, fixed step 0.01 s). UNTESTED - see module docstring."""
    def __init__(self, host="localhost", port=2000, town="Town04", spawn=None, dt=R.DT):
        import carla
        self.carla = carla
        self.client = carla.Client(host, port); self.client.set_timeout(30.0)
        self.world = self.client.load_world(town)
        self.dt = dt
        s = self.world.get_settings(); self._old = (s.synchronous_mode, s.fixed_delta_seconds)
        s.synchronous_mode = True; s.fixed_delta_seconds = dt; self.world.apply_settings(s)
        self.map = self.world.get_map()
        self.wp0 = self._find_segment() if spawn is None else self.map.get_waypoint(carla.Location(*spawn))
        self.lane_w = self.wp0.lane_width
        self.vehicle = None

    def _find_segment(self, length=300.0, step=15.0):
        """first straight driving waypoint with a same-direction left lane for `length` metres"""
        for wp in self.map.generate_waypoints(25.0):
            if wp.lane_type != self.carla.LaneType.Driving: continue
            left = wp.get_left_lane()
            if left is None or left.lane_type != self.carla.LaneType.Driving or left.lane_id * wp.lane_id < 0: continue
            ok, cur = True, wp
            for _ in range(int(length / step)):
                nxt = cur.next(step)
                if not nxt: ok = False; break
                nxt = nxt[0]
                dyaw = abs((nxt.transform.rotation.yaw - wp.transform.rotation.yaw + 180) % 360 - 180)
                l2 = nxt.get_left_lane()
                if dyaw > 1.0 or l2 is None or l2.lane_id * nxt.lane_id < 0: ok = False; break
                cur = nxt
            if ok: return wp
        raise RuntimeError("no straight two-lane segment found; pass --spawn x y z")

    def reset(self, ri, mu_true, v0, mu_dry=R.MU_DRY):
        c = self.carla
        if self.vehicle is not None: self.vehicle.destroy()
        self.world.set_weather(c.WeatherParameters(cloudiness=30 + 60 * ri, precipitation=100 * ri,
                                                   precipitation_deposits=100 * ri, wetness=100 * ri, fog_density=15 * ri))
        bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        tf = self.wp0.transform; tf.location.z += 0.5
        self.vehicle = self.world.spawn_actor(bp, tf)
        ph = self.vehicle.get_physics_control()
        base = 3.5
        for w in ph.wheels: w.tire_friction = base * mu_true / mu_dry
        self.vehicle.apply_physics_control(ph)
        self.max_steer = np.radians(ph.wheels[0].max_steer_angle)
        self.loc0, self.yaw0 = tf.location, tf.rotation.yaw
        self.right0 = self.wp0.transform.get_right_vector()
        fwd = self.wp0.transform.get_forward_vector()
        self.vehicle.set_target_velocity(c.Vector3D(fwd.x * v0, fwd.y * v0, 0))
        self.v0 = v0; self.int_v = 0.0
        for _ in range(20): self._drive(0.0); self.world.tick()      # settle
        self.loc0 = self.vehicle.get_location(); self.yaw0 = self.vehicle.get_transform().rotation.yaw

    def _drive(self, delta):
        c = self.carla
        v = self.vehicle.get_velocity(); sp = np.hypot(v.x, v.y)
        self.int_v += (self.v0 - sp) * self.dt
        thr = np.clip(0.4 * (self.v0 - sp) + 0.05 * self.int_v + 0.15, 0, 1); brk = float(np.clip(0.3 * (sp - self.v0 - 0.5), 0, 1))
        self.vehicle.apply_control(c.VehicleControl(throttle=float(thr), brake=brk, steer=float(np.clip(-delta / self.max_steer, -1, 1))))

    def true_state(self, k):
        t = self.vehicle.get_transform(); loc = t.location
        d = np.array([loc.x - self.loc0.x, loc.y - self.loc0.y, loc.z - self.loc0.z])
        y = -(d[0] * self.right0.x + d[1] * self.right0.y)
        yaw = (t.rotation.yaw - self.yaw0 + 180) % 360 - 180
        r = -np.radians(self.vehicle.get_angular_velocity().z)
        return dict(y=np.array([y]), psi=np.array([-np.radians(yaw)]), r=np.array([r]))

    def apply_and_tick(self, cmd, yref):
        self._drive(float(cmd[0])); self.world.tick()
        t = self.vehicle.get_transform(); rv = t.get_right_vector(); fv = t.get_forward_vector()
        a = self.vehicle.get_acceleration(); v = self.vehicle.get_velocity()
        ay = -(a.x * rv.x + a.y * rv.y)
        vxb = v.x * fv.x + v.y * fv.y; vyb = -(v.x * rv.x + v.y * rv.y)
        yaw = (t.rotation.yaw - self.yaw0 + 180) % 360 - 180
        ctrl = self.vehicle.get_control()
        return dict(ay=np.array([ay]), beta=np.array([np.arctan2(vyb, max(vxb, 1.0))]), psi=np.array([-np.radians(yaw)]),
                    delta=np.array([-ctrl.steer * self.max_steer]))

    def close(self):
        if self.vehicle is not None: self.vehicle.destroy()
        s = self.world.get_settings(); s.synchronous_mode, s.fixed_delta_seconds = self._old; self.world.apply_settings(s)


# ====================================================================================== episode runner (adapter-agnostic)
class _Logs:     # duck-typed stand-in so Plant.metrics can be reused
    pass


def run_episode(adapter, policy, sc, mu_true, vx, dt=R.DT, t_end=R.T_END):
    """Closed loop for ONE scenario (B=1). Measurement delay/noise/IMU exactly as plant.Plant.measure."""
    n = int(round(t_end / dt))
    policy.setup(sc)
    zn = sc["z"]; ny = zn[:, :, 0] * sc["sig_y"]; npsi = zn[:, :, 1] * sc["sig_psi"]; nr = zn[:, :, 2] * sc["sig_r"]
    nd = sc["nd"]
    L = _Logs(); L.dt, L.n = dt, n
    L.Yh = np.zeros((n, 1)); L.Yref = np.zeros((n, 1)); L.AY = np.zeros((n, 1)); L.DEL = np.zeros((n, 1))
    L.BETA = np.zeros((n, 1)); L.PSI = np.zeros((n, 1)); L.mu = np.asarray(mu_true, float).reshape(1); L.vx = np.asarray(vx, float).reshape(1)
    PSIh = np.zeros((n, 1))
    for k in range(n):
        s = adapter.true_state(k)
        L.Yh[k] = s["y"]; PSIh[k] = s["psi"]
        kk = max(k - int(nd[0]), 0)
        ym, psim, rm = L.Yh[kk] + ny[k], PSIh[kk] + npsi[k], s["r"] + nr[k]
        cmd = policy.act(k, k * dt, ym, psim, rm)
        o = adapter.apply_and_tick(cmd, policy.yref_now)
        L.Yref[k] = policy.yref_now; L.AY[k] = o["ay"]; L.DEL[k] = o["delta"]; L.BETA[k] = o["beta"]; L.PSI[k] = o["psi"]
    return P.Plant.metrics(L, policy.T, sc["v0"])


def get_policy(name, params_path="results/best_parameters.json", tuned_path="results/mpc_tuned.json"):
    prm = json.load(open(params_path))
    if name in prm: return PL.ParamPolicy(np.array(list(prm[name].values())))
    tm = json.load(open(tuned_path))
    if name == "MPC": return PL.MPCPolicy(True, T0=tm["MPC"]["T0"], T1=tm["MPC"]["T1"], r_rate=tm["MPC"]["r_rate"], stiff_exp=0.0)
    raise SystemExit(f"unknown controller {name}")


def selftest():
    rng = np.random.default_rng(1)
    sc = R.make_scenarios([0.1, 0.5, 0.9], [22., 26., 30.], rng, mu_err_std=0.1)
    worst = 0.0
    for name in ["GA-RainAware", "Hand-tuned", "MPC"]:
        ref = P.cost_of(P.run_policy(get_policy(name), sc))
        got = []
        for i in range(3):
            si = {k: (v[i:i + 1] if k != "z" else v[:, i:i + 1, :]) for k, v in sc.items()}
            pol = get_policy(name); pol.setup(si)
            ad = SimAdapter(si, int(round(R.T_END / R.DT)), pol.vx)
            got.append(P.cost_of(run_episode(ad, get_policy(name), si, ad.mu, pol.vx))[0])
        diff = np.abs(np.array(got) - ref).max(); worst = max(worst, diff)
        print(f"{name:13s} J(run_policy) {np.round(ref,3)}  J(harness) {np.round(got,3)}  max|diff| {diff:.2e}")
    print("SELFTEST", "PASSED" if worst < 1e-8 else "FAILED")
    return worst < 1e-8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town04"); ap.add_argument("--spawn", type=float, nargs=3)
    ap.add_argument("--controller", default="GA-RainAware"); ap.add_argument("--rain", type=float, nargs="+", default=[0, .25, .5, .75, 1.0])
    ap.add_argument("--speed", type=float, nargs="+", default=[20., 25.]); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mu-err", type=float, default=0.10); ap.add_argument("--out", default="carla_results.csv")
    a = ap.parse_args()
    if a.selftest: sys.exit(0 if selftest() else 1)
    ad = CarlaAdapter(a.host, a.port, a.town, a.spawn)
    rows = []
    try:
        for ri in a.rain:
            for v0 in a.speed:
                for rep in range(a.reps):
                    sc = R.make_scenarios([ri], [v0], np.random.default_rng(1000 * rep + int(100 * ri)), mu_err_std=a.mu_err)
                    pol = get_policy(a.controller); pol.setup(sc)
                    ad.reset(ri, float(sc["mu_nom"][0]), float(pol.vx[0]))
                    # NOTE: CARLA's lane width replaces LANE_W only if you also change rainlc.LANE_W (default 3.5 m)
                    m = run_episode(ad, get_policy(a.controller), sc, sc["mu_nom"], pol.vx)
                    J = P.cost_of(m)[0]
                    rows.append(dict(controller=a.controller, rain=ri, v0=v0, rep=rep, J=float(J), rmse=float(m["rmse"][0, 0]),
                                     final=float(m["final"][0, 0]), util=float(m["util"][0, 0]), fail=float(m["fail"][0, 0])))
                    print(rows[-1], flush=True)
    finally:
        ad.close()
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
