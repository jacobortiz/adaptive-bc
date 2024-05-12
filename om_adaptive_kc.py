import multiprocessing
from model import Model
import numpy as np
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import networkx as nx

import pickle
import bz2

from tqdm.contrib.itertools import product

FONT = FontProperties()
FONT.set_family('serif')
FONT.set_name('Times New Roman')
FONT.set_style('italic')
FONTSIZE = 34

SEED = 51399

def load(filename):
    model = bz2.BZ2File(filename, 'rb')
    model = pickle.load(model)
    return model


def run_adaptive(c, t):
    SEED = 51399 + t
    G = nx.karate_club_graph()
    params = {
            "trial" : f"ADAPTIVE_KC_{t}",
            "max_steps" : 1_000_000,
            "tolerance" : 1e-5,
            "mu" : 0.1,
            "beta" : 0.25,     # 1 is DW
            "c" : c,
            "M" : 1,
            "K" : 1,
            "gamma": 0.01,     # 0 is DW
            "delta": 0.99,     # 1 is DW
    }

    model = Model(seed_sequence=SEED, G=G, **params)
    model.run(test=True)

    return model, t

def run_om(k: int, c, b, t, omega=None):
    G = nx.karate_club_graph()
    params = {
            "trial" : f"ADAPTIVE_KC_OM_{t}",
            "max_steps" : 1_000_000,
            "tolerance" : 1e-5,
            "mu" : 0.1,
            "beta" : b,     # 1 is DW
            "c" : c,
            "M" : 1,
            "K" : 1,
            "gamma": 0.01,     # 0 is DW
            "delta": 0.99,     # 1 is DW
    }

    model = Model(seed_sequence=SEED, G=G, **params)

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

    model.run(test=True)
    if omega is None:
        return model, k, b, t
    else:
        return model, k, b, t, omega
    
def load_all_om(beta = 0.3):
    """ LOADING DATA """
    k_values = [0, 5, 10, 15, 20, 25, 30]
    c_values = [0.1, 0.3, 0.5]

    beta_values = [beta]

    path = 'OM/ADAPTIVE_KC/'

    types = ['degree']
    params = product(k_values, c_values, beta_values, types)
    degree_om_values = {}
    for k, c, b, t in params:
        filename = f'KC_ADAPTIVE_{t}_c-{c}_k-{k}_β-{b}.pbz2'
        loaded_model = load(path + filename)

        if c not in degree_om_values:
            degree_om_values[c] = {}

        degree_om_values[c][k] = loaded_model.overall_opinions
        
    print(degree_om_values)

    types = ['min_degree']
    params = product(k_values, c_values, beta_values, types)
    min_degree_om_values = {}
    for k, c, b, t in params:
        filename = f'KC_ADAPTIVE_{t}_c-{c}_k-{k}_β-{b}.pbz2'
        loaded_model = load(path + filename)

        if c not in min_degree_om_values:
            min_degree_om_values[c] = {}

        min_degree_om_values[c][k] = loaded_model.overall_opinions
    print(min_degree_om_values)

    types = ['greedy']
    params = product(k_values, c_values, beta_values, types)
    greedy_om_values = {}
    for k, c, b, t in params:
        filename = f'KC_ADAPTIVE_{t}_c-{c}_k-{k}_β-{b}.pbz2'
        loaded_model = load(path + filename)

        if c not in greedy_om_values:
            greedy_om_values[c] = {}

        greedy_om_values[c][k] = loaded_model.overall_opinions
    print(greedy_om_values)

    types = ['min_opinion']
    params = product(k_values, c_values, beta_values, types)
    min_opinion_om_values = {}
    for k, c, b, t in params:
        filename = f'KC_ADAPTIVE_{t}_c-{c}_k-{k}_β-{b}.pbz2'
        loaded_model = load(path + filename)

        if c not in min_opinion_om_values:
            min_opinion_om_values[c] = {}

        min_opinion_om_values[c][k] = loaded_model.overall_opinions
    print(min_opinion_om_values)

    types = ['max_opinion']
    params = product(k_values, c_values, beta_values, types)
    max_opinion_om_values = {}
    for k, c, b, t in params:
        filename = f'KC_ADAPTIVE_{t}_c-{c}_k-{k}_β-{b}.pbz2'
        loaded_model = load(path + filename)

        if c not in max_opinion_om_values:
            max_opinion_om_values[c] = {}

        max_opinion_om_values[c][k] = loaded_model.overall_opinions
    print(max_opinion_om_values)

    types = ['random']
    params = product(k_values, c_values, beta_values, types)
    random_om_values = {}
    for k, c, b, t in params:
        filename = f'KC_ADAPTIVE_{t}_c-{c}_k-{k}_β-{b}.pbz2'
        loaded_model = load(path + filename)

        if c not in random_om_values:
            random_om_values[c] = {}

        random_om_values[c][k] = loaded_model.overall_opinions
    print(random_om_values)

    types = ['proposed']
    params = product(k_values, c_values, beta_values, types)
    proposed_om_values = {}
    for k, c, b, t in params:
        filename = f'KC_ADAPTIVE_{t}_c-{c}_k-{k}_β-{b}_Ω-1.pbz2'
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

    filename = f'ALL_OM_VALUES_ADAPTIVE_KC_beta-{beta}_Ω-1.pbz2'
    with bz2.BZ2File(path + filename, 'w') as f:
        pickle.dump(all_om_values, f)
        print(f'saved to file, {filename}')

def plot_om():
    beta = 0.3
    confidence = 0.5
    load_all_om(beta)

    path = f'OM/ADAPTIVE_KC/'
    filename = f'ALL_OM_VALUES_ADAPTIVE_KC_beta-{beta}_Ω-1.pbz2'


    all_adaptive_om = load(path + filename)

    # markers = {1: 'v', 3: 'o', 5: 's', 10: 'd'}
    # plt.figure(figsize=(10, 8))
    # for o, k_c_values in all_proposed_om.items():
    #     temp_k = {}
    #     for k_c, overall_opinions in k_c_values.items():
    #         k, c = k_c 
    #         if c == 0.5:
    #             if c not in temp_k:
    #                 temp_k[c] = {}
    #             else:                        
    #                 temp_k[c][k] = overall_opinions
    #     for initial_confidence, v in temp_k.items():
    #         # initial opinion not recorded for something reason
    #         v[0] = 16.74736677251868
    #         v = dict(sorted(v.items()))

    #         plt.plot(list(v.keys()), list(v.values()), marker=markers[o], label=f'ω={o}')
    # plt.xlabel('k', fontsize=FONTSIZE, fontproperties=FONT)
    # plt.ylabel('g(x)', fontsize=FONTSIZE, fontproperties=FONT)
    # plt.xticks(fontsize=FONTSIZE)
    # plt.yticks(fontsize=FONTSIZE) 
    # plt.legend(fontsize=FONTSIZE-16, frameon=False)
    # plt.show()

    """ OTHER OMS """
    plt.figure(figsize=(10, 8))
    for t, values in all_adaptive_om.items():
        if t == 'proposed': t = 'TRIADS' 
            
        for c, k_values in values.items():
            # change this for different initial confidence
            if c == confidence:
                plt.plot(list(k_values.keys()), list(k_values.values()), marker='d', label=t)

                # for k, overall_opinions in k_values.items():
                #     # print(f'{t}, k={k}, c={c}, g(x)={overall_opinions}')
                #     plt.plot(k, overall_opinions, label=f'{t}, k={k}, c={c}')

    plt.xlabel('k', fontsize=FONTSIZE, fontproperties=FONT)
    plt.ylabel('g(x)', fontsize=FONTSIZE, fontproperties=FONT)
    
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE) 

    plt.legend(fontsize=FONTSIZE-14, frameon=False)

    plt.show()
    
    # plt.plot(om_values, label=f'{t}, k={k}, c={c}')


if __name__ == '__main__':

    # k_values = [0, 5, 10, 15, 20, 25, 30]
    # c_values = [0.1, 0.3, 0.5]
    # beta_values = [0.1, 0.3, 0.5]

    c_values = [0.7, 0.9]
    trials = list(range(1, 51))

    params = product(c_values, trials)
    
    # # single test
    # # k_values = [0, 15]
    # # c_values = [0.1]

    # # types = ['degree', 'min_degree', 'greedy', 'min_opinion', 'max_opinion', 'random']
    # # params = product(k_values, c_values, beta_values, types)

    # # uncomment for proposed method
    # types = ['proposed']
    # omega_values = [1, 5, 10]
    # params = product(k_values, c_values, beta_values, types, omega_values)

    pool = Pool(processes=multiprocessing.cpu_count())
    results = []

    for c, t in params:
        # print(f'running k={k}, c={c}, b={b}, o={o}, type: {t}')
        result = pool.apply_async(run_adaptive, args=(c, t))
        results.append(result)

    # # add o if running proposed in for loop and args() 
    # for k, c, b, t, o in params:
    #     print(f'running k={k}, c={c}, b={b}, o={o}, type: {t}')
    #     result = pool.apply_async(run_om, args=(k, c, b, t, o))
    #     results.append(result)

    pool.close()
    pool.join()

    path = 'data/comparisons/'
    for result in results:
        model, t = result.get()
        filename = f'comparisonsKC_ADAPTIVE_c-{model.initial_c[0]}_trial-{t}.pbz2'
        
        with bz2.BZ2File(path + filename, 'w') as f:
            pickle.dump(model, f)
            print(f'saved to file, {filename}')


    """ LOAD FILES """

    # filename = f'KC_ADAPTIVE_degree_c-0.1_k-15_β-0.5.pbz2'
    # path = 'OM/ADAPTIVE_KC/'

    # load(path + filename)

    # load_all_om()

    # plot_om()