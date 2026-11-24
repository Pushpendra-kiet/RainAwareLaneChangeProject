"""dr.py - domain-randomised GA fitness: every individual is evaluated on randomly mismatched plants."""
import numpy as np
import rainlc as R, plant as P, policies as PL


def fitness_dr(pop, sc, mm, w=R.W, worst_weight=0.5, plant_cls=None):
    Pn, S = pop.shape[0], sc["ri"].size
    sce = {k: (np.tile(v, Pn) if k != "z" else np.tile(v, (1, Pn, 1))) for k, v in sc.items()}
    mme = {k: (np.tile(v, Pn) if isinstance(v, np.ndarray) else v) for k, v in mm.items()}
    pol = PL.ParamPolicy(np.repeat(pop, S, axis=0))
    met = P.run_policy(pol, sce, mismatch=mme, plant_cls=plant_cls)
    met = {k: v.reshape(Pn, S) for k, v in met.items()}
    J = R.cost(met, w)
    return J.mean(1) + worst_weight * J.max(1), met, J
