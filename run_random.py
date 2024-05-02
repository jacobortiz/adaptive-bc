import multiprocessing
from model import Model
import numpy as np
from multiprocessing import Pool

import matplotlib.pyplot as plt
import networkx as nx

import pickle
import bz2

import seaborn as sns

SEED = 51399
RNG = np.random.default_rng(seed=SEED)

# record data for baseline results
def kwparams(N, c, beta, trial, K):
    params = {
        "trial" : "ABC",
        "max_steps" : 100000,
        "N" : N,
        "p" : 0.01,
        "tolerance" : 1e-5,
        "mu" : 0.1,
        "beta" : beta,
        "c" : c,
        "M" : 1,
        "K" : K,
        "full_time_series": True,
        "gamma": .1,     # 0 is DW
        "delta": .9,     # 1 is DW
    }
    return params

def baseline_params(N, c, K, i):
    baseline = {
        "trial" : i,
        "max_steps" : 100000,
        "N" : N,
        "p" : 0.01,
        "tolerance" : 1e-5,
        "mu" : 0.1,
        "beta" : 1,
        "c" : c,
        "M" : 1,
        "K" : K,
        "gamma": 0,     # 0 is DW
        "delta": 1,     # 1 is DW
    }
    return baseline

def load_baseline_files():
    N = 1000
    # c = np.round(RNG.uniform(0.1, 1, N), decimals=1)
    c = 0.1
    K = 5

    filename = f'ER_baseline_c-{c}.pbz2'
    path = 'data/baseline/' + filename
    print(f'loading {filename} ...')
    loaded_model = bz2.BZ2File(path, 'rb')
    loaded_model = pickle.load(loaded_model)
    print('done.')

    # initial
    loaded_model.print_graph(time=0, opinions=True)
    # final
    loaded_model.print_graph(time=-1, opinions=True)

    # print(loaded_model.X_data[0][:50])
    # print(loaded_model.X_data[loaded_model.convergence_time //2][:50])
    # print(loaded_model.X_data[-1][:50])

def run_model(c, i):
    model = Model(seed_sequence=SEED, **baseline_params(1000, c, 5, i))
    model.run(test=True)
    return model

def ER_random():
    N = 1000
    # c = np.round(RNG.uniform(0.1, 1, N), decimals=1)
    # c = 0.1
    K = 5
    
    # filename = f'ER_baseline_c-{c}.pbz2'

    # Define the range of c values to test
    c_values = [0.1,0.3,0.5,1]

    # model = Model(seed_sequence=SEED, **baseline_params(N, c, K))

    # with Pool(processes=len(c_values)) as pool:
    #     results = pool.map(run_model, c_values)

    # for i, result in enumerate(results):
    #     filename = f'ER_baseline_c-{c_values[i]}_K-{K}.pbz2'
    #     with bz2.BZ2File(f'data/baseline/{filename}', 'w') as f:
    #         pickle.dump(result, f)
    #         print(f'saved to file: {filename}')

    """ improved? """
    pool = multiprocessing.Pool(processes=10)

    simulations = range(len(c_values))
    results = []

    for i in simulations:
        result = pool.apply_async(run_model, args=(c_values[i], i))
        results.append(result)
    
    # Close the pool
    pool.close()
    pool.join()

    # Wait for all the simulation tasks to complete
    for result in results:
        model = result.get()
        filename = f'ER_baseline_c-{c_values[model.trial]}_K-{K}.pbz2'
        with bz2.BZ2File(f'data/baseline/{filename}', 'w') as f:
            pickle.dump(model, f)
            print(f'saved to file: {filename}')
    
    # model.run(test=True)

    # with bz2.BZ2File(f'data/baseline/{filename}', 'w') as f:
    #         pickle.dump(model, f)
    #         print(f'saved to file: {filename}')

    # print(model.X_data[0])
    # print(model.X_data[-1])

    #model.print_graph(time=0, opinions=True)



if __name__ == '__main__':

    # load_baseline_files()
    ER_random()
    exit()

    seed = 51399
    N = 1000
    trial = 1
    beta = .3
    K = 5   # K node pairs interact

    RNG = np.random.default_rng(seed=seed)
    c = np.round(RNG.uniform(0.1, 1, N), decimals=1)

    # filename = f'Erdős–Rényi_N-{N}_trial_1.pbz2'

    # # model = Model(seed_sequence=seed, **kwparams(N, c, beta, 1, K), )
    # # model.run(test=True)
    # # model.save_model(filename=filename)

    # print(model.initial_c)
    # print(model.c)

    # print(f'loading {filename} ...')
    # loaded_model = bz2.BZ2File(filename, 'rb')
    # loaded_model = pickle.load(loaded_model)
    # print('done.')

    # print(f'convergence time: {loaded_model.convergence_time}')
    
    # print(len(loaded_model.edges))

    # sns.histplot(data=loaded_model.initial_c, x='c')
    # plt.show()

    # print(loaded_model.c)

    # width = 10
    # data = np.round(loaded_model.c, decimals=5) # loaded_model.c
    # plt.hist(data, bins=10, color='#607c8e', rwidth=0.9, density=True) 
    # plt.xlabel('$\it{c}$')
    # plt.ylabel('Frequency')
    # plt.show()

    # print(loaded_model.c)
    
    # loaded_model.print_graph(time=0, opinions=True)

    # for t, old, new in loaded_model.edge_changes:
    #     print(f'time: {t}, edge: old={old}, new={new}')


    # loaded_model.print_graph(opinions=True)
    # for i in range(0, loaded_model.convergence_time, 1000):
    #     loaded_model.print_graph(time=i, opinions=True)

    # print(loaded_model.X_data[0])
    # print(loaded_model.X_data[-1])
    
    # Plot opinion evolutions of loaded_model
    # print(np.shape(loaded_model.X_data))
    # plt.figure(figsize=(12, 8))    
    # for i in range(loaded_model.N):
    #     plt.plot(range(len(loaded_model.X_data)), [opinions[i] for opinions in loaded_model.X_data])
    # plt.xlabel('$\it{t}$')
    # plt.ylabel('$\it{opinions}$')
    # plt.title('Opinion Evolution')
    # plt.legend()
    # plt.show()

    print('done')

