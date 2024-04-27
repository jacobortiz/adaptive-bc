import multiprocessing
from model import Model
import numpy as np
from multiprocessing import Pool

import matplotlib.pyplot as plt
import networkx as nx

import pickle
import bz2

# record data for baseline results
def kwparams(N, c, beta, trial, K):
    params = {
        "trial" : "ABC",
        "max_steps" : 100000,
        "N" : N,
        "p" : 0.01,
        "tolerance" : 1e-5,
        "alpha" : 0.1,
        "beta" : beta,
        "c" : c,
        "M" : 1,
        "K" : K,
        "full_time_series": True,
        "gamma": .1,     # 0 is DW
        "delta": .9,     # 1 is DW
    }
    return params

if __name__ == '__main__':
    seed = 51399
    N = 1000
    trial = 1
    beta = .25
    K = 5 # K node pairs interact

    RNG = np.random.default_rng(seed=seed)
    c = np.round(RNG.uniform(0.1, 1, N), decimals=1)

    filename = f'Erdős–Rényi_N-{N}.pbz2'

    # model = Model(seed_sequence=seed, **kwparams(N, c, beta, 1, K), )
    # model.run(test=True)
    # model.save_model(filename=filename)

    loaded_model = bz2.BZ2File(filename, 'rb')
    loaded_model = pickle.load(loaded_model)

    print(loaded_model.convergence_time)
    
    # loaded_model.print_graph(time=0, opinions=True)

    for i in range(0, loaded_model.convergence_time, 100):
        loaded_model.print_graph(time=i, opinions=True)

    # Plot opinion evolutions of loaded_model
    # print(np.shape(loaded_model.X_data))
    # plt.figure(figsize=(12, 8))    
    # for i in range(loaded_model.N):
    #     plt.plot(range(len(loaded_model.X_data)), [opinions[i] for opinions in loaded_model.X_data])
    # plt.xlabel('Time')
    # plt.ylabel('Opinion')
    # plt.title('Opinion Evolution')
    # plt.legend()
    # plt.show()

    print('done')

