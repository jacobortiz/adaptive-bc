import multiprocessing
from model import Model
import numpy as np
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import networkx as nx

import pickle
import bz2

# from itertools import product
from tqdm.contrib.itertools import product

FONT = FontProperties()
FONT.set_family('serif')
FONT.set_name('Times New Roman')
FONT.set_style('italic')
FONTSIZE = 34


def load(filename):
    data = bz2.BZ2File(filename, 'rb')
    data = pickle.load(data)
    return data

def run_baselines(c, t):
    SEED = 51399 + t
    G = nx.karate_club_graph()
    baseline_params = {
            "trial" : f"BASELINE_KC_{t}",
            "max_steps" : 1_000_000,
            "tolerance" : 1e-5,
            "mu" : 0.1,
            "beta" : 1,     # 1 is DW
            "c" : c,
            "M" : 1,
            "K" : 1,
            "gamma": 0,     # 0 is DW
            "delta": 1,     # 1 is DW
    }

    model = Model(seed_sequence=SEED, G=G, **baseline_params)
    model.run(test=True)

    return model, t

def run_om(k: int, c, t, omega=None):
    SEED = 51399
    G = nx.karate_club_graph()
    baseline_params = {
            "trial" : f"BASELINE_KC_OM_{t}",
            "max_steps" : 1_000_000,
            "tolerance" : 1e-5,
            "mu" : 0.1,
            "beta" : 1,     # 1 is DW
            "c" : c,
            "M" : 1,
            "K" : 1,
            "gamma": 0,     # 0 is DW
            "delta": 1,     # 1 is DW
    }

    model = Model(seed_sequence=SEED, G=G, **baseline_params)
    seed_nodes = None
    if t == 'degree':
        seed_nodes = model.degree_solution(k=k)
    elif t == 'random':
        seed_nodes = model.random_solution(k=k)
    elif t == 'min_opinion':
        seed_nodes = model.min_opinion_solution(k=k)
    elif t == 'max_opinion':
        seed_nodes = model.max_opinion_solution(k=k)
    elif t == 'greedy':
        seed_nodes = model.greedy_solution(k=k)
    elif t == 'min_degree':
        seed_nodes = model.min_degree_solution(k=k)
    elif t == 'proposed':
        seed_nodes = model.proposed_solution(k=k, omega=omega)

    # seed nodes adopt opinion 1, and confidence 0
    for node, _ in seed_nodes:
        model.X[node] = 1
        model.c[node] = 0

    label = None
    # if omega is None:
    #     label = f'BASELINE, k={k}, c={c}'
    # else:
    #     label = f'BASELINE, k={k}, c={c}, omega={omega}'

    model.run(test=True)
    if omega is None:
        return model, k, t
    else:
        return model, k, t, omega
    
def load_all_om():
    """ LOADING DATA """
    k_values = [0, 5, 10, 15, 20, 25, 30]
    c_values = [0.1, 0.3, 0.5]

    path = 'OM/BASELINE_KC/'

    types = ['degree']
    params = product(k_values, c_values, types)
    degree_om_values = {}
    for k, c, t in params:
        # result = pool.apply_async(run_om, args=(k, c, t, o))
        # results.append(result)
        filename = f'KC_BASELINE_{t}_c-{c}_k-{k}.pbz2'
        loaded_model = load(path + filename)

        if c not in degree_om_values:
            degree_om_values[c] = {}

        degree_om_values[c][k] = loaded_model.overall_opinions
    print(degree_om_values)

    types = ['min_degree']
    params = product(k_values, c_values, types)
    min_degree_om_values = {}
    for k, c, t in params:
        # result = pool.apply_async(run_om, args=(k, c, t, o))
        # results.append(result)
        filename = f'KC_BASELINE_{t}_c-{c}_k-{k}.pbz2'
        loaded_model = load(path + filename)

        if c not in min_degree_om_values:
            min_degree_om_values[c] = {}

        min_degree_om_values[c][k] = loaded_model.overall_opinions
    print(min_degree_om_values)

    types = ['greedy']
    params = product(k_values, c_values, types)
    greedy_om_values = {}
    for k, c, t in params:
        # result = pool.apply_async(run_om, args=(k, c, t, o))
        # results.append(result)
        filename = f'KC_BASELINE_{t}_c-{c}_k-{k}.pbz2'
        loaded_model = load(path + filename)

        if c not in greedy_om_values:
            greedy_om_values[c] = {}

        greedy_om_values[c][k] = loaded_model.overall_opinions
    print(greedy_om_values)

    types = ['min_opinion']
    params = product(k_values, c_values, types)
    min_opinion_om_values = {}
    for k, c, t in params:
        # result = pool.apply_async(run_om, args=(k, c, t, o))
        # results.append(result)
        filename = f'KC_BASELINE_{t}_c-{c}_k-{k}.pbz2'
        loaded_model = load(path + filename)

        if c not in min_opinion_om_values:
            min_opinion_om_values[c] = {}

        min_opinion_om_values[c][k] = loaded_model.overall_opinions
    print(min_opinion_om_values)

    types = ['max_opinion']
    params = product(k_values, c_values, types)
    max_opinion_om_values = {}
    for k, c, t in params:
        # result = pool.apply_async(run_om, args=(k, c, t, o))
        # results.append(result)
        filename = f'KC_BASELINE_{t}_c-{c}_k-{k}.pbz2'
        loaded_model = load(path + filename)

        if c not in max_opinion_om_values:
            max_opinion_om_values[c] = {}

        max_opinion_om_values[c][k] = loaded_model.overall_opinions
    print(max_opinion_om_values)

    types = ['random']
    params = product(k_values, c_values, types)
    random_om_values = {}
    for k, c, t in params:
        # result = pool.apply_async(run_om, args=(k, c, t, o))
        # results.append(result)
        filename = f'KC_BASELINE_{t}_c-{c}_k-{k}.pbz2'
        loaded_model = load(path + filename)

        if c not in random_om_values:
            random_om_values[c] = {}

        random_om_values[c][k] = loaded_model.overall_opinions
    print(random_om_values)

    types = ['proposed']
    params = product(k_values, c_values, types)
    proposed_om_values = {}
    for k, c, t in params:
        # result = pool.apply_async(run_om, args=(k, c, t, o))
        # results.append(result)
        filename = f'KC_BASELINE_{t}_c-{c}_k-{k}_Ω-1.pbz2'
        loaded_model = load(path + filename)

        if c not in proposed_om_values:
            proposed_om_values[c] = {}

        proposed_om_values[c][k] = loaded_model.overall_opinions
    print(proposed_om_values)

    # Save all dictionaries in an object
    all_om_values = {
        'degree': degree_om_values,
        'min_degree': min_degree_om_values,
        'greedy': greedy_om_values,
        'min_opinion': min_opinion_om_values,
        'max_opinion': max_opinion_om_values,
        'random': random_om_values,
        'proposed': proposed_om_values
    }

    filename = f'ALL_OM_VALUES_BASELINE_KC_Ω-1.pbz2'
    with bz2.BZ2File(path + filename, 'w') as f:
        pickle.dump(all_om_values, f)
        print(f'saved to file, {filename}')

def plot_om():
    filename = f'ALL_OM_VALUES_BASELINE_KC_Ω-1.pbz2'
    path = f'OM/BASELINE_KC/'
    # all_oms = load(path + filename)

    # save object
    filename = f'ALL_PROPOSED_OM_VALUES_BASELINE_KC.pbz2'

    all_proposed_om = load(path + filename)
    markers = {1: 'v', 3: 'o', 5: 's', 10: 'd'}
    plt.figure(figsize=(10, 8))
    for o, k_c_values in all_proposed_om.items():
        temp_k = {}
        for k_c, overall_opinions in k_c_values.items():
            k, c = k_c 
            if c == 0.5:
                if c not in temp_k:
                    temp_k[c] = {}
                else:                        
                    temp_k[c][k] = overall_opinions
        for initial_confidence, v in temp_k.items():
            # initial opinion not recorded for something reason
            v[0] = 16.74736677251868
            v = dict(sorted(v.items()))

            plt.plot(list(v.keys()), list(v.values()), marker=markers[o], label=f'ω={o}')
    plt.xlabel('k', fontsize=FONTSIZE, fontproperties=FONT)
    plt.ylabel('g(x)', fontsize=FONTSIZE, fontproperties=FONT)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE) 
    plt.legend(fontsize=FONTSIZE-16, frameon=False)
    plt.show()

    """ OTHER OMS """
    # plt.figure(figsize=(10, 8))
    # for t, values in all_oms.items():
    #     if t == 'proposed': t = 'TRIADS' 
            
    #     for c, k_values in values.items():
    #         if c == 0.1:
    #             plt.plot(list(k_values.keys()), list(k_values.values()), marker='d', label=t)

    #             # for k, overall_opinions in k_values.items():
    #             #     # print(f'{t}, k={k}, c={c}, g(x)={overall_opinions}')
    #             #     plt.plot(k, overall_opinions, label=f'{t}, k={k}, c={c}')

    # plt.xlabel('k', fontsize=FONTSIZE, fontproperties=FONT)
    # plt.ylabel('g(x)', fontsize=FONTSIZE, fontproperties=FONT)
    
    # plt.xticks(fontsize=FONTSIZE)
    # plt.yticks(fontsize=FONTSIZE) 

    # plt.legend(fontsize=FONTSIZE-14, frameon=False)

    # plt.show()
    
    # plt.plot(om_values, label=f'{t}, k={k}, c={c}')


if __name__ == '__main__':

    plot_om()
    exit()

    # c_values = [0.1, 0.3, 0.5]
    k_values = [0, 5, 10, 15, 20, 25, 30]
    c_values = [0.1, 0.3, 0.5]
    types = ['random']
    beta_values = [0.1, 0.3, 0.5]
    # trials = list(range(1, 51))


    num_trials = (len(k_values) * len(c_values) * len(types) * len(beta_values))

    params = product(k_values, c_values, beta_values, types)

    print(num_trials)

    # exit()


    # path = 'OM/BASELINE_KC/'
    path = 'OM/ADAPTIVE_KC/'
    AVERAGE_T = 0
    AVERAGE_X = 0
    for k, c, b, t in params:
        filename = f'KC_ADAPTIVE_{t}_c-{c}_k-{k}_β-{b}.pbz2'
        # filename = f'KC_BASELINE_{t}_c-{c}_k-{k}_Ω-1.pbz2'
        # filename = f'KC_BASELINE_{t}_c-{c}_k-{k}.pbz2'
        data = load(path + filename)

        AVERAGE_X += np.sum(data.X.copy())
        AVERAGE_T += data.convergence_time

    print(AVERAGE_T / num_trials)
    print(AVERAGE_X / num_trials)

    exit()

    # print(data.simulation_info())

    # # types = ['degree', 'min_degree', 'greedy', 'min_opinion', 'max_opinion', 'random']
    # # types = ['degree']
    # # params = product(k_values, c_values, types)

    # # # uncomment for proposed method
    # types = ['proposed']
    # omega_values = [1, 3, 5, 10]
    # params = product(k_values, c_values, types, omega_values)

    pool = Pool(processes=multiprocessing.cpu_count())
    results = []

    for c, t in params:
        print(f'running: {c}, trial: {t}')
        result = pool.apply_async(run_baselines, args=(c, t))
        results.append(result)

    # # # # add o if running proposed in for loop and args() 
    # for k, c, t, o in params:
    #     # print(f'running k={k}, c={c}, type: {t}')
    #     result = pool.apply_async(run_om, args=(k, c, t, o))
    #     results.append(result)

    pool.close()
    pool.join()

    path = 'data/comparisons/'
    for result in results:
        model, t = result.get()
        filename = f'KC_BASELINE_c-{model.initial_c[0]}_trial_{t}.pbz2'
                
        with bz2.BZ2File(path + filename, 'w') as f:
            pickle.dump(model, f)
            print(f'saved to file, {filename}')


    # for result in results:
    #     model, k, t, o = result.get()
    #     filename = f'KC_BASELINE_{t}_c-{model.initial_c[0]}_k-{k}_Ω-{o}.pbz2'
    #     # filename = f'KC_BASELINE_{t}_c-{model.initial_c[0]}_k-{k}.pbz2'
    #     path = 'OM/BASELINE_KC/'
        
    #     with bz2.BZ2File(path + filename, 'w') as f:
    #         pickle.dump(model, f)
    #         print(f'saved to file, {filename}')

    # plot_om()


    pass