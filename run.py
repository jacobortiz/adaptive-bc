from model import Model
from node import Node
import numpy as np
from numpy.random import SeedSequence

# record data for baseline results
def kwparams(C, beta, trial):
    params = {
        "trial" : trial,
        "max_steps" : 1000000,
        "N" : 1000,
        "p" : 0.1,
        "tolerance" : 1e-5,
        "alpha" : 0.1, # 0.5 is consensus parameter
        "beta" : beta,
        "C" : C,
        "M" : 1,
        "K" : 5,
        "full_time_series": True
    }
    return params

def run_model(seed_sequence, model_params, filename=None):
    model = Model(seed_sequence, **model_params)
    model.run(test=True)
    model.save_model(f'data/{filename}.pbz2')

if __name__ == '__main__':

    seed = 123456789

    C = 1
    beta = 1
    trial = 1
    print(f'running with params: \n{kwparams(C, beta, trial)}')

    run_model(seed_sequence=seed, model_params=kwparams(C, beta, trial), filename='m3-run')
