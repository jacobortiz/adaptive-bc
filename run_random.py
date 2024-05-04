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

# RECORDING DATA FOR ABC MODEL
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
        "gamma": .01,     # 0 is DW
        "delta": .99,     # 1 is DW
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
    # TEST INITIAL CONFIDENCE STARTS
    # c = RNG.uniform(0.1, 1, N)
    c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    K = 10

    pool = multiprocessing.Pool(processes=10)

    # beta_values = [0.1, 0.3, 0.5, 0.7, 1]

    simulations = range(len(c_values))
    results = []

    for i in simulations:
        print(f'testing {c_values[i]}')
        result = pool.apply_async(run_model_abc, args=(N, c_values[i], .25, 1, K))
        results.append(result)
    
    # Close the pool
    pool.close()
    pool.join()

    # Wait for all the simulation tasks to complete
    for result in results:
        model = result.get()
        print(f'beta: {model.beta}, CT: {model.convergence_time}')
        filename = f'ER_ABC_beta-{model.beta}_K-{K}_M-{model.M}_c_{model.initial_c[0]}.pbz2'
        with bz2.BZ2File(f'data/abc/{filename}', 'w') as f:
            pickle.dump(model, f)
            print(f'saved to file: {filename}')

def LOAD_ER_ABC():
    K = 10
    M = 2
    gamma = 0.1
    delta = 0.9

    # beta_values = [0.1, 0.3, 0.5, 0.7, 1]
    beta_values = [1]

    for b in beta_values:
        filename = f'ER_ABC_beta-{b}_K-10.pbz2'
        print(f'loading: {filename}')
        loaded_model = bz2.BZ2File('data/abc/5.2/' + filename, 'rb')
        loaded_model = pickle.load(loaded_model)
        print('Done loading.')

        # loaded_model.print_graph(opinions=True)
        # print(f'CT: {loaded_model.convergence_time}')

        # average_opinions = np.mean(loaded_model.X_data[-1])
        # print(f"Average opinions: {average_opinions}")
        # print(f'ass: init={loaded_model.assortativity_history[0]}, final={loaded_model.assortativity_history[-1]}')
        # average_confidence = np.mean(loaded_model.c)
        # print(f"avg INIT: {np.mean(loaded_model.initial_c)} Average confidence: {average_confidence}")


        FONTSIZE = 30 - 4
        
        data = loaded_model.X_data
        MAX_PLOT = len(data)
        plt.figure(figsize=(10, 8))
        plt.plot(loaded_model.X_data, linewidth=0.5)
        plt.xlabel('t', fontsize=FONTSIZE, fontproperties=FONT)
        plt.ylabel('x', fontsize=FONTSIZE, fontproperties=FONT)
        plt.xticks(fontsize=FONTSIZE-4)
        plt.yticks(fontsize=FONTSIZE-4) 

        # plt.gca().set_xticks(range(0, MAX_PLOT, 10_000))
        # plt.gca().set_xticklabels([f'{int(t/1000)}k' if t > 0 else int(t) for t in plt.gca().get_xticks()])

        plt.show()

        # Plot frequencies of initial c
        # width = 0.1
        # data = RNG.uniform(0.1, 1, 1000)
        # plt.figure(figsize=(10, 8))
        # plt.hist(data, bins=np.arange(0.1, 1.1, width), color='#607c8e', rwidth=0.9, density=False)
        # plt.xlabel('c', fontsize=FONTSIZE, fontproperties=FONT)
        # plt.ylabel('Frequency', fontsize=FONTSIZE - 8)
        # plt.xticks(fontsize=FONTSIZE - 12)
        # plt.yticks(fontsize=FONTSIZE - 12)
        # plt.show()




    # loaded_model.print_graph(time=0, opinions=True)
    # loaded_model.print_graph(opinions=True)

def run_om(k: int, type: str):
    c = RNG.uniform(0.1, 1, 1000)
    params = {
        "trial" : 1,
        "N": 1000,
        "p": 0.01,
        "max_steps" : 250_000,
        "tolerance" : 1e-5,
        "mu" : 0.1,
        "c" : c,
        "beta" : .25,
        "M" : 5,
        "K" : 25,
        "gamma" : 0.01,
        "delta": 0.99
    }

    model = Model(seed_sequence=SEED, **params)

    seed_nodes = None
    if type == 'degree':
        seed_nodes = model.degree_solution(k=k)
    elif type == 'random':
        seed_nodes = model.random_solution(k=k)
    elif type == 'min_opinion':
        seed_nodes = model.min_opinion_solution(k=k)
    elif type == 'max_opinion':
        seed_nodes = model.max_opinion_solution(k=k)
    elif type == 'min_degree':
        seed_nodes = model.min_degree_solution(k=k)

    # seed nodes adopt opinion 1
    if seed_nodes is not None:
        for node, _ in seed_nodes:
            model.X[node] = 1

    model.run(test=True, label='OM')
    return model, k

def test_om(load=False):
    print('running OM')
    if load is False:
        simulations = 20
        pool = multiprocessing.Pool(processes=10)
        results = []
        for i in range(simulations):
            k=50
            result = pool.apply_async(run_om, args=(i*k, 'random'))
            results.append(result)
        
        # Close the pool
        pool.close()
        pool.join()

        # Wait for all the simulation tasks to complete
        max_opinion_random = {}
        for result in results:
            model, k = result.get()
            max_opinion_random[k] = np.sum(model.X.copy())

        print('running min')
        # min_opinion
        pool = multiprocessing.Pool(processes=10)
        results = []
        for i in range(simulations):
            k=50
            result = pool.apply_async(run_om, args=(i*k, 'min_opinion'))
            results.append(result)
        
        # Close the pool
        pool.close()
        pool.join()

        # Wait for all the simulation tasks to complete
        max_opinion_min = {}
        for result in results:
            model, k = result.get()
            max_opinion_min[k] = np.sum(model.X.copy())

        print('saving min')
        with bz2.BZ2File('data/OM/ER_max_opinion_min.pbz2', 'w') as f:
            pickle.dump(max_opinion_min, f)
            print('saved to file')

        """ SAVE THE RESULTS IN AN OBJECT?????????? """
        print('running degree') 
        # degree solution
        pool = multiprocessing.Pool(processes=10)
        results_degree = []
        for i in range(simulations):
            k=50
            result = pool.apply_async(run_om, args=(i*k, 'degree'))
            results_degree.append(result)
        
        # Close the pool
        pool.close()
        pool.join()

        max_opinion_degree = {}
        for result in results_degree:
            model, k = result.get()
            max_opinion_degree[k] = np.sum(model.X.copy())
        
        print('saving degree')
        with bz2.BZ2File('data/OM/ER_max_opinion_degree.pbz2', 'w') as f:
            pickle.dump(max_opinion_degree, f)
            print('saved to file')
        

        print('running max')
        # max solution
        pool = multiprocessing.Pool(processes=10)
        results_max_opinion = []
        for i in range(simulations):
            k=50
            result = pool.apply_async(run_om, args=(i*k, 'max_opinion'))
            results_max_opinion.append(result)
        
        # Close the pool
        pool.close()
        pool.join()

        max_opinion_max = {}
        for result in results_max_opinion:
            model, k = result.get()
            max_opinion_max[k] = np.sum(model.X.copy())
        
        print('saving max')
        with bz2.BZ2File('data/OM/ER_max_opinion_max.pbz2', 'w') as f:
            pickle.dump(max_opinion_max, f)
            print('saved to file')

        print('running min-degree')
        # min-degree solution
        pool = multiprocessing.Pool(processes=10)
        results_min_degree = []
        for i in range(simulations):
            k=50
            result = pool.apply_async(run_om, args=(i*k, 'min_degree'))
            results_min_degree.append(result)
        
        # Close the pool
        pool.close()
        pool.join()

        max_opinion_min_degree = {}
        for result in results_min_degree:
            model, k = result.get()
            max_opinion_min_degree[k] = np.sum(model.X.copy())
        
        print('saving min-degree')
        with bz2.BZ2File('data/OM/ER_max_opinion_min_degree.pbz2', 'w') as f:
            pickle.dump(max_opinion_min_degree, f)
            print('saved to file')


        print('random: ', max_opinion_random)
        print('min: ', max_opinion_min)
        print(f'max: {max_opinion_max}')
        print('degree: ', max_opinion_degree)
        print(f'min degree: {max_opinion_min_degree}')

        """ SAVE """
        with bz2.BZ2File('data/OM/ER_max_opinion_random.pbz2', 'w') as f:
            pickle.dump(max_opinion_random, f)
            print('saved to file')

        """ save results objects """
        results_data = {
            "result_degree": results_degree,
            "results_max_opinion": results_max_opinion,
            "results_min_degree": results_min_degree
        }

        with bz2.BZ2File('data/OM/ER_MODEL_OBJECTS_FINAL.pbz2', 'w') as f:
            pickle.dump(results_data, f)
            print('saved to file')


    else:
        # load data 
        max_opinion_random = bz2.BZ2File('data/OM/ER_max_opinion_random.pbz2', 'rb')
        max_opinion_random = pickle.load(max_opinion_random)

        max_opinion_min = bz2.BZ2File('data/OM/ER_max_opinion_min.pbz2', 'rb')
        max_opinion_min = pickle.load(max_opinion_min)

        max_opinion_degree = bz2.BZ2File('data/OM/ER_max_opinion_degree.pbz2', 'rb')
        max_opinion_degree = pickle.load(max_opinion_degree)

        max_opinion_max = bz2.BZ2File('data/OM/ER_max_opinion_max.pbz2', 'rb')
        max_opinion_max = pickle.load(max_opinion_max)

        max_opinion_min_degree = bz2.BZ2File('data/OM/ER_max_opinion_min_degree.pbz2', 'rb')
        max_opinion_min_degree = pickle.load(max_opinion_min_degree)

    # Line plot of OM algorithms
    plt.figure(figsize=(10, 8))
    plt.plot(list(max_opinion_degree.keys()), list(max_opinion_degree.values()), color='red', marker='d', label='Degree')
    plt.plot(list(max_opinion_min.keys()), list(max_opinion_min.values()), color='orange', marker='x', label='Min-Opinion')
    plt.plot(list(max_opinion_max.keys()), list(max_opinion_max.values()), color='black', marker='v', label='Max-Opinion')
    plt.plot(list(max_opinion_random.keys()), list(max_opinion_random.values()), color='green', marker='o', label='Random')
    plt.plot(list(max_opinion_min_degree.keys()), list(max_opinion_min_degree.values()), color='blue', marker='s', label='Min-Degree')
    plt.xlabel('k', fontsize=FONTSIZE, fontproperties=FONT)
    plt.ylabel('g(x)', fontsize=FONTSIZE, fontproperties=FONT)
    
    plt.xticks(fontsize=FONTSIZE-4)
    plt.yticks(fontsize=FONTSIZE-4) 

    plt.legend(fontsize=FONTSIZE-16, frameon=False)

    plt.show()

if __name__ == '__main__':

    # load_baseline_files()
    # ER_random()
    # load_baseline_ER()
    # ER_ABC()
    # LOAD_ER_ABC()
    test_om(load=False)
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

