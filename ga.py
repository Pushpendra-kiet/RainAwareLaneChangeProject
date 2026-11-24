"""ga.py - real-coded genetic algorithm (tournament selection, SBX crossover, polynomial mutation, elitism)."""
import numpy as np


class GeneticAlgorithm:
    def __init__(self, lb, ub, pop_size=60, generations=50, seed=0, pc=0.9, pm=None,
                 eta_c=15.0, eta_m=20.0, elite=2, tournament=3):
        self.lb, self.ub = np.asarray(lb, float), np.asarray(ub, float)
        self.n = self.lb.size
        self.pop_size, self.generations = pop_size, generations
        self.rng = np.random.default_rng(seed)
        self.pc = pc
        self.pm = pm if pm is not None else 1.0 / self.n
        self.eta_c, self.eta_m, self.elite, self.k = eta_c, eta_m, elite, tournament
        self.free = self.ub > self.lb          # locked genes have lb == ub

    def _init(self):
        return self.lb + self.rng.random((self.pop_size, self.n)) * (self.ub - self.lb)

    def select(self, fit):
        idx = self.rng.integers(0, self.pop_size, (self.pop_size, self.k))
        return idx[np.arange(self.pop_size), np.argmin(fit[idx], axis=1)]

    def crossover(self, sel):
        """SBX on consecutive parent pairs (sel[0],sel[1]), (sel[2],sel[3]), ...; both children are kept."""
        kids = sel.copy()
        for i in range(0, len(sel) - 1, 2):
            if self.rng.random() < self.pc:
                a, b = sel[i], sel[i + 1]
                u = self.rng.random(self.n)
                beta = np.where(u <= 0.5, (2 * u) ** (1 / (self.eta_c + 1)),
                                (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1)))
                swap = self.rng.random(self.n) < 0.5
                x1 = 0.5 * ((1 + beta) * a + (1 - beta) * b)
                x2 = 0.5 * ((1 - beta) * a + (1 + beta) * b)
                kids[i] = np.where(swap, x1, a); kids[i + 1] = np.where(swap, x2, b)
        return kids

    def mutate(self, x):
        span = np.where(self.free, self.ub - self.lb, 1.0)
        mask = (self.rng.random(x.shape) < self.pm) & self.free
        u = self.rng.random(x.shape)
        d = np.where(u < 0.5, (2 * u) ** (1 / (self.eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1)))
        return np.where(mask, x + d * span, x)

    def run(self, eval_fn, verbose=False):
        """eval_fn(pop, gen) -> fitness (lower is better). Returns best, best_fit, history dict."""
        pop = self._init()
        hist = dict(best=[], mean=[], gen_best_x=[])
        best_x, best_f = None, np.inf
        for g in range(self.generations):
            fit = eval_fn(pop, g)
            order = np.argsort(fit)
            hist["best"].append(float(fit[order[0]])); hist["mean"].append(float(fit.mean()))
            hist["gen_best_x"].append(pop[order[0]].copy())
            if fit[order[0]] < best_f:
                best_f, best_x = float(fit[order[0]]), pop[order[0]].copy()
            if verbose and g % 10 == 0:
                print(f"gen {g:3d} best {fit[order[0]]:.3f} mean {fit.mean():.3f}", flush=True)
            if g == self.generations - 1:
                break
            elites = pop[order[: self.elite]].copy()
            sel = pop[self.select(fit)]
            kids = self.crossover(sel)
            kids = np.clip(self.mutate(kids), self.lb, self.ub)
            kids[: self.elite] = elites
            pop = kids
        hist["final_pop"] = pop
        hist["final_fit"] = fit
        return best_x, best_f, hist
