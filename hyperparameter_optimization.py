from hyperopt import fmin, tpe, Trials, hp, rand
import numpy as np
from objective_fn import objective, iteration
import pickle
import os

# Keep track of results
if os.path.isfile("rand_trials.bin"):
    with open("rand_trials.bin", "rb") as f:
        trials = pickle.load(f)
else:
    trials = Trials()

iteration += len(trials.trials)

space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
    'dropout': hp.uniform('dropout', 0, 0.6),
    'l2': hp.loguniform('l2', np.log(1e-6), np.log(1e-2)),
    'model_size': hp.choice('model_size', [[512, 32], [256, 256, 32], [128], [64, 64], [32]]),
    'del_num_times': hp.quniform('del_num_times', 0, 2, 1),
    'synonym_num_times': hp.quniform('synonym_num_times', 0, 2, 1)
}

max_evals = 120

# Run optimization
best = fmin(fn=objective, space=space, algo=rand.suggest,
            max_evals=max_evals, trials=trials)
print(best)

with open("rand_trials.bin", "wb+") as f:
    pickle.dump(trials, f)
