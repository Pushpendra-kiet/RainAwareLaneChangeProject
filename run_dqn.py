"""Train DQN agents (3 seeds in parallel) and save the best-validated weights per seed."""
import json, pickle, time
from multiprocessing import Pool
import numpy as np
import dqn

def job(seed):
    q, Jv, ep, curve = dqn.train_dqn(seed, n_batches=300, log=True)
    pickle.dump(dict(W=q.W, b=q.b, sizes=q.sizes), open(f"results/dqn_seed{seed}.pkl", "wb"))
    return dict(seed=seed, val_J=Jv, best_batch=ep, curve=curve)

if __name__ == "__main__":
    t = time.time()
    with Pool(2) as p: out = p.map(job, [0, 1, 2], chunksize=1)
    json.dump(out, open("results/dqn_training.json", "w"), indent=1)
    print("done", time.time() - t)
