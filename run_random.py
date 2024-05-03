import multiprocessing
from model import Model
import numpy as np
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import networkx as nx

import pickle
import bz2

import seaborn as sns

SEED = 51399
RNG = np.random.default_rng(seed=SEED)

FONT = FontProperties()
FONT.set_family('serif')
FONT.set_name('Times New Roman')
FONT.set_style('italic')
FONTSIZE = 34

# record data for baseline results
def kwparams(N, c, beta, trial, K):
    params = {
        "trial" : "ABC",
        "max_steps" : 500_000,
        "N" : N,
        "p" : 0.01,
        "tolerance" : 1e-5,
        "mu" : 0.1,
        "beta" : beta,
        "c" : c,
        "M" : 2,
        "K" : K,
        "gamma": .1,     # 0 is DW
        "delta": .9,     # 1 is DW
    }
    return params

def baseline_params(N, c, K, i):
    baseline = {
        "trial" : i,
        "max_steps" : 250000,
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

def load_baseline_KC():
    N = 1000
    # c = np.round(RNG.uniform(0.1, 1, N), decimals=1)
    c = 0.1
    K = 5

    filename = f'KC_baseline_c-{c}.pbz2'
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

def load_baseline_ER():
    c=0.5
    filename = f'ER_baseline_c-{c}_K-5.pbz2'
    path = 'data/baseline/' + filename
    print(f'loading {filename} ...')
    loaded_model = bz2.BZ2File(path, 'rb')
    loaded_model = pickle.load(loaded_model)
    print('done.')

    print(loaded_model.convergence_time)

    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_style('italic')

    fontsize=34


    # MAX_PLOT = 160_000
    # plt.figure(figsize=(10, 8))
    # plt.plot(loaded_model.X_data[:MAX_PLOT], linewidth=0.5)
    # plt.xlabel('t', fontsize=fontsize, fontproperties=font)
    # plt.ylabel('x', fontsize=fontsize, fontproperties=font)
    # plt.xticks(fontsize=fontsize-4)
    # plt.yticks(fontsize=fontsize-4) 

    # plt.gca().set_xticks(range(0, MAX_PLOT, 40_000))
    # # plt.gca().set_xticklabels([f'{int(t/1000)}k' if t > 0 else int(t) for t in plt.gca().get_xticks()])

    # plt.show()


def run_model_baseline(c, i):
    """" SET K values here ... """
    model = Model(seed_sequence=SEED, **baseline_params(1000, c, 20, i))
    model.run(test=True)
    return model

def ER_random():
    print('running ER random')
    N = 1000
    # c = np.round(RNG.uniform(0.1, 1, N), decimals=1)
    # c = 0.1
    
    K = 20
    
    # filename = f'ER_baseline_c-{c}.pbz2'

    # Define the range of c values to test
    # c_values = [0.1,0.3,0.5,1]
    c_values = [0.1]

    # model = Model(seed_sequence=SEED, **baseline_params(N, c, K))

    pool = multiprocessing.Pool(processes=10)

    simulations = range(len(c_values))
    results = []

    for i in simulations:
        result = pool.apply_async(run_model_baseline, args=(c_values[i], i))
        results.append(result)
    
    # Close the pool
    pool.close()
    pool.join()

    # Wait for all the simulation tasks to complete
    for result in results:
        model = result.get()
        filename = f'ER_baseline_c-{c_values[model.trial]}_K-{K}_MAX_TIME_STEPS_EXTENDED.pbz2'
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

def run_model_abc(N, c, beta, trial, K):
    model = Model(seed_sequence=SEED, **kwparams(N, c, beta, trial, K))
    model.run(test=True)
    return model

def ER_ABC():
    print('running ER on ABC')
    N = 1000
    c = RNG.uniform(0.1, 1, N)
    K = 10

    pool = multiprocessing.Pool(processes=10)

    beta_values = [0.1, 0.3, 0.5, 0.7, 1]
    simulations = range(len(beta_values))
    results = []

    delta_values = [0.1, 0.3, 0.5, 0.7, 0.9]


    for i in simulations:
        result = pool.apply_async(run_model_abc, args=(N, c, beta_values[i], 1, K))
        results.append(result)
    
    # Close the pool
    pool.close()
    pool.join()

    # Wait for all the simulation tasks to complete
    for result in results:
        model = result.get()
        print(f'beta: {model.beta}, CT: {model.convergence_time}')
        filename = f'ER_ABC_beta-{model.beta}_K-{K}.pbz2'
        with bz2.BZ2File(f'data/abc/{filename}', 'w') as f:
            pickle.dump(model, f)
            print(f'saved to file: {filename}')

def LOAD_ER_ABC():
    beta = 0.7
    K = 5
    filename = f'ER_ABC_beta-{beta}_K-{K}.pbz2'
    print(f'loading: {filename}')
    loaded_model = bz2.BZ2File('data/abc/' + filename, 'rb')
    loaded_model = pickle.load(loaded_model)

    print(loaded_model.convergence_time)
    print(loaded_model.info())

    # loaded_model.print_graph(time=0, opinions=True)
    # loaded_model.print_graph(opinions=True)



    # MAX_PLOT = loaded_model.convergence_time
    
    # plt.figure(figsize=(10, 8))
    # plt.plot(loaded_model.X_data, linewidth=0.5)
    # plt.xlabel('t', fontsize=FONTSIZE, fontproperties=FONT)
    # plt.ylabel('x', fontsize=FONTSIZE, fontproperties=FONT)
    # plt.xticks(fontsize=FONTSIZE-4)
    # plt.yticks(fontsize=FONTSIZE-4) 

    # plt.gca().set_xticks(range(0, MAX_PLOT, 40_000))
    # plt.gca().set_xticklabels([f'{int(t/1000)}k' if t > 0 else int(t) for t in plt.gca().get_xticks()])

    plt.show()


if __name__ == '__main__':

    # load_baseline_files()
    # ER_random()
    # load_baseline_ER()
    ER_ABC()
    # LOAD_ER_ABC()
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

