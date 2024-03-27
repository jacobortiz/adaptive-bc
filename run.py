import multiprocessing
from model import Model
from node import Node
import numpy as np
from numpy.random import SeedSequence

# record data for baseline results
def kwparams(N, C, beta, trial, K):
    params = {
        "trial" : trial,
        "max_steps" : 1000000,
        "N" : N,
        "p" : 0.1,
        "tolerance" : 1e-5,
        "alpha" : 0.1, # 0.5 is consensus parameter
        "beta" : beta,
        "C" : C,
        "M" : 1,
        "K" : K,
        "full_time_series": True
    }
    return params

def run_model(seed_sequence, model_params, filename=None):
    model = Model(seed_sequence, **model_params)
    model.run(test=True)
    model.save_model(f'data/{filename}.pbz2')

    if model.beta != 1:
        print(f'Network assortativity: {model.start_assortativity}')
        print(f'End assortativity: {model.end_assortativity}')

if __name__ == '__main__':
    seed = 123456789

    N = 1000
    RNG = np.random.default_rng(seed=seed)

    # confidence_intervals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # C = RNG.choice(confidence_intervals, N)
    C = 1

    # beta = 0.25
    beta = 1
    trial = 1

    # params,     
    K_list = {1, 5, 10, 20}


    # run_model(seed_sequence=seed, model_params=kwparams(N, C, beta, trial), filename=f'baseline-ABC-K_{K}-C_1-beta_1')

    print('Running model...')
    processes = []
    for K in K_list:
        print(f'testing K={K}...')
        process = multiprocessing.Process(target=run_model, args=(seed, kwparams(N, C, beta, trial, K), f'baseline-ABC-K_{K}-C_1-beta_1'))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
    