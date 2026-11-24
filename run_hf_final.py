import json, numpy as np
import run_hf, run_extension as X, policies as PL
d = json.load(open("results/best_parameters_hf.json"))
X.COL.update({"GA-RainAware-HF": "#006d2c", "GA-Fixed-HF": "#54278f"})
extra = {k: (lambda p=np.array(list(v.values())): PL.ParamPolicy(p)) for k, v in d.items()}
run_hf.main(extra)
