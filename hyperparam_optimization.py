from hyperopt import fmin, tpe, Trials, hp
import numpy as np
from objective_fn import objective
import pickle
import os

# Keep track of results
if os.path.isfile("bayes_trials.bin"):
    with open("bayes_trials.bin", "rb") as f:
        bayes_trials = pickle.load(f)
else:
    bayes_trials = Trials()

space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
    'dropout': hp.uniform('dropout', 0, 0.5),
    'l2': hp.loguniform('l2', np.log(1e-6), np.log(1e-2)),
    'model_size': hp.choice('model_size', [[512, 32], [256, 256, 32], [128], [64, 64], [32]]),
    'del_num_times': hp.quniform('del_num_times', 1, 2, 1),
    'synonym_num_times': hp.quniform('synonym_num_times', 1, 2, 1)
}

max_evals = 5

# Run optimization
best = fmin(fn=objective, space=space, algo=tpe.suggest,
            max_evals=max_evals, trials=bayes_trials)
print(best)

with open("bayes_trials.bin", "wb+") as f:
    pickle.dump(bayes_trials, f)
