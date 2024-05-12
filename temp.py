# from itertools import product
from tqdm.contrib.itertools import product
from model import Model
import networkx as nx
from time import sleep

import numpy as np

import pickle
import bz2

SEED = 51399

baseline_params = {
            "trial" : "BASELINE_KC_OM",
            "max_steps" : 1_000_000,
            "tolerance" : 1e-5,
            "mu" : 0.1,
            "beta" : 1,     # 1 is DW
            "c" : 0.3,
            "M" : 1,
            "K" : 1,
            "gamma": 0.01,     # 0 is DW
            "delta": 0.99,     # 1 is DW
    }

if __name__ == '__main__':
    # G = nx.karate_club_graph()
    # model = Model(seed_sequence=SEED, G=G, **baseline_params)


    k_values = [0, 5, 10, 15, 20, 25, 30]
    c_values = [0.1, 0.3, 0.5]
    beta_values = [0.1, 0.3, 0.5]

    omega_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # # single test
    # # k_values = [0, 15]
    # # c_values = [0.1]

    types = ['degree', 'min_degree', 'greedy', 'min_opinion', 'max_opinion', 'random']
    params = product(k_values, c_values, beta_values, types, omega_values)


    """ USE THIS TO RUN SIMULATIONS TO KEEP TRACK OF THEM???"""
    # model.simulation_info()
    # for k, c, b, t, o in params:


    # filename = f'OM\BASELINE_KC\KC_BASELINE_degree_c-0.1_k-0.pbz2'
    # filename = f'OM\ADAPTIVE_KC\KC_ADAPTIVE_greedy_c-0.0_k-15_β-0.1.pbz2'

    # old_data = f'data/abc/5.3/ER_ABC_beta-0.25_K-10_M-2_c_0.1.pbz2'
    # model = bz2.BZ2File(old_data, 'rb')
    # model = pickle.load(model)
    # print(f'loading {old_data}')
    # # print(model.convergence_time, np.mean(model.c), model.assortativity_history[-1])
    # print(np.mean(model.X_data[-1]))

    # c_values = np.round(np.arange(0.1, 1, 0.1), decimals=1)
    # for i in c_values:
    #     old_data = f'data/abc/5.3/ER_ABC_beta-0.25_K-10_M-2_c_{i}.pbz2'
    #     model = bz2.BZ2File(old_data, 'rb')
    #     model = pickle.load(model)
    #     print(f'loading {old_data}')
    #     # print(np.mean(model.X_data[-1]))
    #     print(np.mean(model.c))