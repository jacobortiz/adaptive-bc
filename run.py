from model import Model
from node import Node
import numpy as np
from numpy.random import SeedSequence

def kwparams(C, beta, trial):
    params = {
        "trial" : trial,
        "max_steps" : 10000,
        "N" : 10,
        "p" : 0.1,
        "tolerance" : 10e-5,
        "alpha" : 0.1,
        "C" : C,
        "beta" : beta,
        "M" : 1,
        "K" : 1,
        "full_time_series": True
    }
    return params

def run_model(seed_sequence, model_params):
    model = Model(seed_sequence, **model_params)
    model.run(test=True)
    model.save_model('data/test_run.pbz2')

if __name__ == '__main__':

    seed = 123456789

    C = 0.3
    beta = 1
    trial = 1
    print(f'running with params: \n{kwparams(C, beta, trial)}')

    run_model(seed_sequence=seed, model_params=kwparams(C, beta, trial))

