# Extension: MPC / DQN baselines, optimiser comparison, validation

Reproduce: `python3 run_dqn.py; python3 run_optimizers.py; python3 run_dr.py; python3 run_extension.py`
(~25 min on 2 cores). CARLA harness: `python3 carla_validate.py --selftest`.

## What was added
| File | Content |
|---|---|
| `plant.py` | Generic plant + policy interface. `ParamPolicy` reproduces `rainlc.simulate` exactly (max diff 0.0). Optional **mismatched plant**: Magic-Formula tyres with post-peak drop, tyre relaxation lag, random mass/inertia/stiffness/CG/steering-gain/friction errors |
| `policies.py` | `ParamPolicy` (GA-type law), `MPCPolicy`, `DQNPolicy` |
| `dqn.py`, `run_dqn.py` | Double DQN in numpy (128x128 MLP, 7 steering-correction actions at 20 Hz, 3 seeds) |
| `optimizers.py`, `run_optimizers.py` | PSO, CMA-ES, NSGA-II (3 objectives) vs GA: 5 seeds, 6,400 evaluations each |
| `dr.py`, `run_dr.py` | Domain-randomised GA (trained on mismatched plants) |
| `carla_validate.py` | CARLA harness (**untested against CARLA**) + offline self-test (passes) |
| `ga.py` | Crossover bug fixed (see below) |

## 1. Baselines (nominal plant, held-out N=300)
| Controller | J | Fail % | RMSE m | Friction util | Duration s |
|---|---|---|---|---|---|
| GA-RainAware | **4.36** | 0 | **0.056** | 0.51 | 2.92 |
| GA-Fixed-DR | 4.71 | 0 | 0.060 | 0.49 | 2.91 |
| GA-RainAware-DR | 4.96 | 0 | 0.072 | 0.48 | 2.90 |
| MPC | 5.57 | 0 | 0.155 | 0.39 | 3.70 |
| DQN | 5.63 | 0 | 0.123 | 0.41 | 3.35 |
| MPC-dry (no friction info) | 17.45 | 13.0 | 0.206 | 0.63 | 3.00 |
| MPC-adhesion | 28.85 | 28.0 | 0.164 | 0.58 | 3.70 |

- MPC and DQN never fail and use much more grip margin (util 0.39-0.41 vs 0.51), but they are slower (3.4-3.7 s) and track about 2x worse. The J gap to GA is mostly their longer duration and larger tracking error. The cost function weights duration, so this ranking depends on those weights.
- Friction information is what matters: MPC without it fails in 13% of cases, with it 0%.
- **MPC-adhesion** (stiffness scaled with estimated friction) performed *worse*. My explanation (stiffness scaling makes the model too soft, so the MPC over-steers) is a hypothesis; I did not test it.
- **MPC tuning** was a coarse grid on the GA training scenarios (rate weight, duration rule), so MPC is reasonable but not exhaustively optimised. It is unconstrained-QP MPC with saturation, not a constrained MPC.
- **DQN is unstable.** Validation cost was best early (batch 30-40 of 300, J ~ 9-12) and then degraded (J ~ 50-150), so I used early-stopped checkpoints chosen on validation. Seeds differ a lot (J 5.6-7.3 on test). I did not tune the reward, learning rate or target update. A better-tuned DQN or PPO may do better; treat this as a weak baseline.
- DQN has a steady-state offset (final error 0.205 m vs 0.03 for others), because it only applies discrete corrections on top of the feedforward and has no integral action.

## 2. Optimisers (5 seeds, equal budget, same selection protocol)
| Optimiser | Test J (mean ± sd) | Best / worst seed | Seeds with failures |
|---|---|---|---|
| GA | 4.58 ± 0.16 | 4.36 / 4.80 | 0/5 |
| PSO | 5.02 ± 1.03 | 4.38 / 6.80 | 2/5 |
| CMA-ES | 4.74 ± 0.52 | 4.33 / 5.31 | 2/5 |
| NSGA-II | 5.91 ± 0.62 | 5.54 / 7.02 | 0/5 |

- GA, PSO and CMA-ES are **not significantly different** (Mann-Whitney p = 0.84 for each vs GA, n = 5). Best seeds are equal (4.33-4.38). GA was most *consistent* (lowest spread); PSO and CMA-ES occasionally converge to worse optima. CMA-ES takes ~1.7x the wall-clock.
- NSGA-II is worse on the scalar test cost (p = 0.01). That is expected: it spreads effort over a Pareto front rather than minimising the weighted sum. Its value is the front (`ext_fig5_optimisers.png`, `ext_fig6_pareto_physical.png`), which exposes the tracking/safety/comfort trade-off without fixing weights.
- With 5 seeds only large differences are detectable. Do not claim GA beats PSO/CMA-ES; claim comparable performance with lower variance in this setting.

## 3. Validation on a different plant (stand-in for CARLA)
Same controllers, same 300 scenarios, mismatched higher-fidelity plant:

| Controller | J | Fail % |
|---|---|---|
| GA-Fixed-DR | **4.26** | 0 |
| GA-RainAware-DR | 4.41 | 0 |
| GA-Fixed | 4.81 | 0 |
| DQN | 5.88 | 0 |
| MPC | 6.06 | 0.3 |
| **GA-RainAware (nominal-trained)** | 8.16 | **4.3** |
| MPC-dry / Hand-tuned / GA-Dry / MPC-adhesion | 21.1 / 15.9 / 14.7 / 31.4 | 16 / 12 / 11 / 30 |

**Important findings:**
1. The original GA-RainAware **overfitted to its training plant**: failures rise 0 → 4.3% → 16% as mismatch grows (`ext_fig2_mismatch_robustness.png`). Failures happened at high speed with the rear-stiffness error, across all rain levels.
2. **Domain randomisation fixes it**: GA-RainAware-DR has 0% failures at mismatch level 1 (1% at level 2), at a small nominal-plant cost (J 4.36 → 4.96).
3. **This weakens the rain-aware claim.** Under mismatch, the non-adaptive GA-Fixed-DR is as good or better than the rain-aware version (J 4.26 vs 4.41; paired Wilcoxon p < 0.001, Fixed-DR better in 251/300 scenarios, but the mean gap is small, 0.15 J; RMSE 0.070 vs 0.080). Rain-aware scheduling helps on the nominal plant (RMSE 0.056 vs 0.097) but not clearly once the plant is uncertain. Report this honestly: the main contribution is *robust GA tuning across rain conditions with domain randomisation*; rain-adaptive scheduling gives a modest, plant-dependent gain.
4. Note the mismatch plant is something I built, and the GA-DR controllers were trained on the *same family* of mismatches (different random draws). It is therefore not an independent test of generalisation to a new plant. CARLA would be.

## 4. CARLA / IPG CarMaker: **not done**
CARLA and CarMaker are not installed here and there is no network, so **no CARLA results exist**. `carla_validate.py` contains a CARLA 0.9.x adapter plus an episode runner. Verified: the runner reproduces `run_policy` exactly through a simulated adapter (`--selftest`, max diff < 1e-15). **Not verified:** the `CarlaAdapter` class itself. Expect to debug spawn location, sign conventions (CARLA is left-handed), steering normalisation and the friction scale on first run. Rain→`tire_friction` scaling is an assumption (CARLA's value is not a physical μ), so calibrate it first. Only vehicle dynamics differ from this simulator; perception delay/noise are injected identically. CarMaker would need its own interface (Python via its API / Simulink) and a licence.

## 5. Corrections to the first report
- **GA bug:** the original crossover reused only half the selected parents' children. Fixed in `ga.py` (`ga_v1_original.py` keeps the old version; `results_v1_old_ga/` keeps old results). Rerunning changed the headline numbers negligibly (GA-RainAware J 4.37 → 4.36; GA-Dry 14.88 → 11.30, 11.3% → 7.0% failures) so the earlier conclusions hold; use the new numbers.
- The earlier README claim "insensitive to friction-estimator error up to 30%" was only tested on the nominal plant.

## 6. Suggested next steps
1. Run `carla_validate.py` on a machine with CARLA (calibrate friction first); this is the main missing piece for a venue that expects high-fidelity validation.
2. Test on a plant family *not* used in training (e.g. train on level-1 mismatch, test with different tyre shapes).
3. Tune DQN properly or use PPO/SAC with continuous steering; add integral action.
4. Implement constrained MPC (steering and slip-angle constraints) with a QP solver.

   **##Extra**
1. python highway_env_validate.py --selftest
2. python highway_env_validate.py --controller GA-RainAware-DR --rain 0.0 0.25 0.5 0.75 1.0 --speed 20 25 30 --reps 10 --out results/highway_env_ga_dr.csv
3. python highway_env_validate.py --all --reps 10 --out results/highway_env_all.csv
4. python highway_env_validate.py --controller GA-RainAware-DR --rain 0.75 --speed 25 --reps 1 --render
6. More optimiser seeds (≥20) and the hyperparameter sensitivity of each optimiser before making any ranking claim.
7. Re-examine the cost weights (duration vs tracking) since they drive the ranking against MPC/DQN.
