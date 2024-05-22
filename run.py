import multiprocessing
from model import Model
import numpy as np
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import networkx as nx

import pickle
import bz2

from sklearn.cluster import KMeans
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

# xmeans package warning error bypass
import warnings
from collections import Counter
np.warnings = warnings

SEED=51399
RNG = np.random.default_rng(seed=SEED)

FONT = FontProperties()
FONT.set_family('serif')
FONT.set_name('Times New Roman')
FONT.set_style('italic')
FONTSIZE = 34

""" CHANGE TO RUN TESTS HERE"""
def kwparams(N, c, beta, trial, K):
    params = {
        "trial" : "ABC",
        "max_steps" : 500_000,
        "N" : N,
        "p" : 0.1,
        "tolerance" : 1e-5,
        "mu" : 0.1,
        "beta" : beta,
        "c" : c,
        "M" : 1,
        "K" : K,
        "gamma": .01,     # 0 is DW
        "delta": .99,     # 1 is DW
    }
    return params

def baseline_params(N):
    baseline = {
        "trial" : "BASELINE",
        "max_steps" : 100000,
        "N" : N,
        "p" : 0.1,
        "tolerance" : 1e-5,
        "mu" : 0.1,
        "beta" : 1,
        "c" : 0.3,
        "M" : 1,
        "K" : 5,
        "gamma": 0,     # 0 is DW
        "delta": 1,     # 1 is DW
    }
    return baseline

def run_model(seed_sequence, model_params, filename=None):
    model = Model(seed_sequence, **model_params)
    model.run(test=True)

    if filename:    
        model.save_model(f'data/{filename}.pbz2')
        return model

    if model.full_time_series:
        model.X_data = model.X_data[:model.convergence_time, :]
    else:
        model.X_data = model.X_data[:int(model.convergence_time / 250) + 1, :]

    model.num_discordant_edges = model.num_discordant_edges[:model.convergence_time - 1]
    model.num_discordant_edges = np.trim_zeros(model.num_discordant_edges)

    if model.beta != 1:
        print(f'Network assortativity: {model.start_assortativity}')
        print(f'End assortativity: {model.end_assortativity}')

    return model

def run_multi_baseline_dw(c=None):
    seed = 51399
    N = 1000

    beta = 1
    trial = 1
    K = 5

    simulations = 50
    average_opinions = 0
    pool = Pool(processes=10)  # Set the maximum number of workers to 10
    results = []

    for i in range(simulations):
        result = pool.apply_async(run_model, args=(seed+i, kwparams(N, c, beta, trial, K)))
        results.append(result)

    pool.close()
    pool.join()

    percentage_increases = []
    for result in results:
        model = result.get()
        sum_opinions = np.sum(model.X_data, axis=1)

        increase_opinions = sum_opinions[-1] - sum_opinions[0]
        percentage_increase = (increase_opinions / sum_opinions[0]) * 100
        average_opinions += percentage_increase
        percentage_increases.append(percentage_increase)

    average_percentage_increase = average_opinions / simulations
    print(f"Average percentage increase in opinions: {average_percentage_increase:.2f}%")

    # Plot percentage increases
    plt.figure(figsize=(12, 8))
    plt.plot(percentage_increases)
    plt.xlabel('Simulation')
    plt.ylabel('Percentage Increase')
    plt.title('Percentage Increase in Opinions')
    plt.show()

def run_om(k: int, type: str):
    G = nx.karate_club_graph()

    c = RNG.uniform(0.1, 1, G.number_of_nodes())
    params = {
        "trial" : 1,
        "max_steps" : 1_000_000,
        "tolerance" : 1e-5,
        "mu" : 0.1,
        "c" : c,
        "beta" : .25,
        "M" : 2,
        "K" : 10,
        "gamma" : 0.01,
        "delta": 0.99
    }

    model = Model(seed_sequence=SEED, G=G, **params)

    seed_nodes = None
    if type == 'degree':
        seed_nodes = model.degree_solution(k=k)
    elif type == 'random':
        seed_nodes = model.random_solution(k=k)
    elif type == 'min_opinion':
        seed_nodes = model.min_opinion_solution(k=k)
    elif type == 'max_opinion':
        seed_nodes = model.max_opinion_solution(k=k)
    elif type == 'greedy':
        seed_nodes = model.greedy_solution(k=k)
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
        simulations = 7
        pool = multiprocessing.Pool(processes=10)
        results = []
        for i in range(simulations):
            k=5
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


        # min_opinion
        pool = multiprocessing.Pool(processes=10)
        results = []
        for i in range(simulations):
            k=5
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


        # degree solution
        pool = multiprocessing.Pool(processes=10)
        results = []
        for i in range(simulations):
            k=5
            result = pool.apply_async(run_om, args=(i*k, 'degree'))
            results.append(result)
        
        # Close the pool
        pool.close()
        pool.join()

        max_opinion_degree = {}
        for result in results:
            model, k = result.get()
            max_opinion_degree[k] = np.sum(model.X.copy())


        # max solution
        pool = multiprocessing.Pool(processes=10)
        results = []
        for i in range(simulations):
            k=5
            result = pool.apply_async(run_om, args=(i*k, 'max_opinion'))
            results.append(result)
        
        # Close the pool
        pool.close()
        pool.join()

        max_opinion_max = {}
        for result in results:
            model, k = result.get()
            max_opinion_max[k] = np.sum(model.X.copy())

        # min-degree solution
        pool = multiprocessing.Pool(processes=10)
        results = []
        for i in range(simulations):
            k=5
            result = pool.apply_async(run_om, args=(i*k, 'min_degree'))
            results.append(result)
        
        # Close the pool
        pool.close()
        pool.join()

        max_opinion_min_degree = {}
        for result in results:
            model, k = result.get()
            max_opinion_min_degree[k] = np.sum(model.X.copy())


        # greedy solution
        pool = multiprocessing.Pool(processes=10)
        results = []
        for i in range(simulations):
            k=5
            result = pool.apply_async(run_om, args=(i*k, 'greedy'))
            results.append(result)
        
        # Close the pool
        pool.close()
        pool.join()

        max_opinion_greedy = {}
        for result in results:
            model, k = result.get()
            max_opinion_greedy[k] = np.sum(model.X.copy())


        print('random: ', max_opinion_random)
        print('min: ', max_opinion_min)
        print(f'max: {max_opinion_max}')
        print('degree: ', max_opinion_degree)
        print(f'min degree: {max_opinion_min_degree}')
        print(f'greedy: {max_opinion_greedy}')

        """ SAVE """
        with bz2.BZ2File('data/OM/KC_max_opinion_random.pbz2', 'w') as f:
            pickle.dump(max_opinion_random, f)
            print('saved to file')

        with bz2.BZ2File('data/OM/KC_max_opinion_min.pbz2', 'w') as f:
            pickle.dump(max_opinion_min, f)
            print('saved to file')

        with bz2.BZ2File('data/OM/KC_max_opinion_degree.pbz2', 'w') as f:
            pickle.dump(max_opinion_degree, f)
            print('saved to file')
        
        with bz2.BZ2File('data/OM/KC_max_opinion_max.pbz2', 'w') as f:
            pickle.dump(max_opinion_max, f)
            print('saved to file')

        with bz2.BZ2File('data/OM/KC_max_opinion_greedy.pbz2', 'w') as f:
            pickle.dump(max_opinion_greedy, f)
            print('saved to file')

        with bz2.BZ2File('data/OM/KC_max_opinion_min_degree.pbz2', 'w') as f:
            pickle.dump(max_opinion_min_degree, f)
            print('saved to file')

    else:
        # load data 
        max_opinion_random = bz2.BZ2File('data/OM/KC_max_opinion_random.pbz2', 'rb')
        max_opinion_random = pickle.load(max_opinion_random)

        max_opinion_min = bz2.BZ2File('data/OM/KC_max_opinion_min.pbz2', 'rb')
        max_opinion_min = pickle.load(max_opinion_min)

        max_opinion_degree = bz2.BZ2File('data/OM/KC_max_opinion_degree.pbz2', 'rb')
        max_opinion_degree = pickle.load(max_opinion_degree)

        max_opinion_max = bz2.BZ2File('data/OM/KC_max_opinion_max.pbz2', 'rb')
        max_opinion_max = pickle.load(max_opinion_max)

        max_opinion_greedy = bz2.BZ2File('data/OM/KC_max_opinion_greedy.pbz2', 'rb')
        max_opinion_greedy = pickle.load(max_opinion_greedy)

        max_opinion_min_degree = bz2.BZ2File('data/OM/KC_max_opinion_min_degree.pbz2', 'rb')
        max_opinion_min_degree = pickle.load(max_opinion_min_degree)

    # Line plot of OM algorithms
    plt.figure(figsize=(10, 8))
    plt.plot(list(max_opinion_greedy.keys()), list(max_opinion_greedy.values()), color='red', marker='d', label='Greedy')
    plt.plot(list(max_opinion_degree.keys()), list(max_opinion_degree.values()), color='blue', marker='+', label='Degree')
    plt.plot(list(max_opinion_min.keys()), list(max_opinion_min.values()), color='orange', marker='x', label='Min-Opinion')
    plt.plot(list(max_opinion_max.keys()), list(max_opinion_max.values()), color='black', marker='v', label='Max-Opinion')
    plt.plot(list(max_opinion_random.keys()), list(max_opinion_random.values()), color='green', marker='o', label='Random')
    plt.plot(list(max_opinion_min_degree.keys()), list(max_opinion_min_degree.values()), color='brown', marker='s', label='Min-Degree')
    plt.xlabel('k', fontsize=FONTSIZE, fontproperties=FONT)
    plt.ylabel('g(x)', fontsize=FONTSIZE, fontproperties=FONT)
    
    plt.xticks(fontsize=FONTSIZE-4)
    plt.yticks(fontsize=FONTSIZE-4) 

    plt.legend(fontsize=FONTSIZE-16, frameon=False)

    plt.show()

def test_assortativity():
    seed = 51399
    params = {
        "trial" : 1,
        "max_steps" : 100000,
        "tolerance" : 1e-5,
        "mu" : 0.1,
        "c" : .3,
        "beta" : .25,
        "M" : 1,
        "K" : 5,
        "full_time_series": True,
        "gamma" : 0.1,
        "delta": 0.9
    }

    G = nx.karate_club_graph()
    model = Model(seed_sequence=seed, G=G, **params)
    model.run(test=True)

    plt.figure(figsize=(8, 6))
    plt.plot(model.assortativity_history[:200])
    plt.xlabel('$\it{t}$')
    plt.ylabel('assortativity')
    plt.show()

def baselines_karate():
    seed = 51399
    c = 1
    params = {
        "trial" : 1,
        "max_steps" : 100000,
        "tolerance" : 1e-5,
        "mu" : 0.1,
        "c" : c,
        "beta" : 1,
        "M" : 1,
        "K" : 1,
        "gamma" : 0,
        "delta": 1
    }
    filename = f'KC_baseline_c-{c}.pbz2'

    # save runs
    G = nx.karate_club_graph()
    model = Model(seed_sequence=seed, G=G, **params)
    model.run(test=True)

    with bz2.BZ2File(f'data/baseline/{filename}', 'w') as f:
            pickle.dump(model, f)
            print(f'saved to file: {filename}')

    # load run
    print(f'loading {filename}')
    model = bz2.BZ2File(f'data/baseline/{filename}', 'rb')
    model = pickle.load(model)
    print('done.')
    # print(f'c={c}, : {model.convergence_time}')

    # num_clusters = find_clusters(model)


    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_style('italic')

    fontsize = 34

    # Visualize opinion evolution
    plt.figure(figsize=(10, 8))
    plt.plot(model.X_data)
    plt.xlabel('t', fontsize=fontsize, fontproperties=font)
    plt.ylabel('x', fontsize=fontsize, fontproperties=font)
    plt.xticks(fontsize=fontsize-4)
    plt.yticks(fontsize=fontsize-4) 

    plt.gca().set_xticks(range(0, model.convergence_time, 3000))
    # plt.gca().set_xticklabels([f'{int(t/1000)}k' if t > 0 else int(t) for t in plt.gca().get_xticks()])

    plt.show()
    
    # Visualize opinion evolution
    # plt.figure(figsize=(10, 8))
    # plt.plot(model.X_data)
    # plt.xlabel('t', fontsize=fontsize, fontproperties=font)
    # plt.ylabel('x', fontsize=fontsize, fontproperties=font)
    # plt.xticks(fontsize=fontsize-4)
    # plt.yticks(fontsize=fontsize-4)
    # plt.show()

def run_abc_karate(c: float):
    print('running abc_karate')
    G = nx.karate_club_graph()
    N = G.number_of_nodes()
    # c = np.round(RNG.uniform(0.1, 1, N), decimals=1)

    model = Model(seed_sequence=SEED, G=G, **kwparams(N, c, 0.25, 1, 5))
    model.run(test=True)
    return model

def ABC_karate():
    G = nx.karate_club_graph()
    N = G.number_of_nodes()
    # c = np.round(RNG.uniform(0.1, 1, N), decimals=1)
    K = 5

    # b_values = [0.1, 0.3, 0.5, 0.7]
    # c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    c = 0.3

    model = Model(SEED, G, **kwparams(N, c, 0.25, 1, K))
    model.run(test=True)

    print(f'{model.convergence_time}')

    # pool = multiprocessing.Pool(processes=10)
    # simulations = range(len(c_values))

    # results = []
    # for i in simulations:
    #     result = pool.apply_async(run_abc_karate, args=(c_values[i]))
    #     results.append(result)
    
    # # Close the pool
    # pool.close()
    # pool.join()

    # # Wait for all the simulation tasks to complete
    # for result in results:
    #     model = result.get()
    #     filename = f'KC_ABC_beta-{model.beta}_K-{model.K}.pbz2'
    #     with bz2.BZ2File(f'data/abc/{filename}', 'w') as f:
    #         pickle.dump(model, f)
    #         print(f'saved to file: {filename}')


    print(r'%%%%%%%%% DONE WITH SIMULATIONS %%%%%%%%%')

    # model = Model(seed_sequence=SEED, G=G, **kwparams(None, c, beta, 1, 5))
    # model.run(test=True)

    # filename = f'KC_ABC_beta-{beta}_K-{K}.pbz2'
    # with bz2.BZ2File(f'data/abc/{filename}', 'w') as f:
    #     pickle.dump(model, f)
    #     print(f'saved to file: {filename}')

    # loaded_model.print_graph(opinions=True)

    # # initial plots
    # # plot confidence
    # data = loaded_model.initial_c # np.round(loaded_model.initial_c, decimals=1)
    # plt.figure(figsize=(10, 8))
    # plt.hist(data, bins=10, color='#607c8e', rwidth=0.9)
    # plt.xlabel('c', fontsize=fontsize, fontproperties=font)
    # plt.ylabel('Frequency', fontsize=fontsize-2)
    # plt.xticks(fontsize=fontsize-4)
    # plt.yticks(fontsize=fontsize-4)
    # plt.show()
    
    print('done saving')

def load_ABC_karate():
    K = 5
    beta = .25
    gamma = 0.01
    delta = 0.99
    # c_values = [0.1, 0.3, 0.5, 0.9, 1]

    
    # for c in c_values:
    #     filename = f'KC_ABC_beta-0.25_K-{K}_g-{gamma}_d-{delta}_c-{c}_TIME-1M.pbz2'

    #     # load run
    #     print(f'loading {filename}')
    #     loaded_model = bz2.BZ2File(f'data/abc/5.3/{filename}', 'rb')
    #     loaded_model = pickle.load(loaded_model)
    #     print('done.')


    #     print(loaded_model.convergence_time)
        
    #     loaded_model.print_graph(opinions=True)
        

    


    # G = loaded_model.original_graph
    # degrees = dict(G.degree(G.nodes()))

    # # check for corrrectness
    # # counting freq of opinion weighted by their degree
    # initial_opinions = {}
    # for node in range(G.number_of_nodes()):
    #     opinion = loaded_model.X_data[0][node]
    #     degree = degrees[node]
    #     if opinion in initial_opinions:
    #         initial_opinions[opinion] += degree
    #     else:
    #         initial_opinions[opinion] = degree

    
    # leading_opinion = max(initial_opinions, key=initial_opinions.get)

    # print(f"Leading Opinion: {leading_opinion}")
    # print(f"Degree-Weighted Count: {initial_opinions[leading_opinion]}")

    # # Visualize opinion evolution
    # plt.figure(figsize=(10, 8))
    # plt.plot(loaded_model.X_data)
    # plt.xlabel('t', fontsize=FONTSIZE, fontproperties=FONT)
    # plt.ylabel('x', fontsize=FONTSIZE, fontproperties=FONT)
    # plt.xticks(fontsize=FONTSIZE-4)
    # plt.yticks(fontsize=FONTSIZE-4) 

    # plt.gca().set_xticks(range(0, loaded_model.convergence_time, 1500))
    # # plt.gca().set_xticklabels([f'{int(t/1000)}k' if t > 0 else int(t) for t in plt.gca().get_xticks()])

    # plt.show()



def find_clusters(model):
    if model is None:
        raise ValueError('Model cannot be None.')
        
    data = model.X_data[-1]

    # Specify the range of possible numbers of clusters

    kmin = 1  # Minimum number of clusters
    kmax = 5  # Maximum number of clusters

    # Initialize the initial centers using the K-means++ method
    initial_centers = kmeans_plusplus_initializer(data.reshape(-1, 1), kmin).initialize()

    # Create an instance of the Xmeans algorithm
    xmeans_instance = xmeans(data.reshape(-1, 1), initial_centers, kmax)

    # Run the Xmeans algorithm
    xmeans_instance.process()

    # Get the clusters and their centers
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()

    # Print the number of clusters found
    num_clusters = len(clusters)
    print(f"Number of clusters found: {num_clusters}")

    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {cluster}, {model.X[cluster]}")

    return num_clusters

def TEST_GREEDY():
    G = nx.karate_club_graph()
    N = G.number_of_nodes()
    c = RNG.uniform(0.1, 1, G.number_of_nodes())

    params = {
        "trial" : 'om greedy',
        "max_steps" : 1_000_000,
        "tolerance" : 1e-5,
        "mu" : 0.1,
        "c" : c,
        "beta" : .25,
        "M" : 2,
        "K" : 10,
        "gamma" : 0.01,
        "delta": 0.99
    }

    k_values = list(range(0, N+1, 5))

    model = Model(SEED, G=G, **params)
    seed_nodes = model.greedy_solution(k=10)

    print(seed_nodes)


if __name__ == '__main__':

    # baselines_karate()
    # ABC_karate()
    # load_ABC_karate()

    test_om(load=True)
    # test_assortativity()

    # TEST_GREEDY()
    exit()

    seed = 51399
    N = 1000
    trial = 1
    beta = .25
    K = 5 # K node pairs interact 

    # pool = multiprocessing.Pool(processes=10)

    # Run the simulation task 50 times
    # simulations = 10
    # results = []

    # for i in range(simulations):
    #     result = pool.apply_async(run_model, args=(seed+i, baseline_params(N)))
    #     results.append(result)

    for i in range(simulations):
        RNG = np.random.default_rng(seed=seed+i)
        c = np.round(RNG.uniform(0.1, 1, N), decimals=1)    
        result = pool.apply_async(run_model, args=(seed+i, kwparams(N, c, beta, 1, K)))
        results.append(result)
    
    # Close the pool
    pool.close()
    pool.join()

    # Wait for all the simulation tasks to complete
    for result in results:
        model = result.get()
        print(f'model: {model.trial}, convergence time: {model.convergence_time}')

    # Plot opinion evolution for baseline model
    # plt.figure(figsize=(12, 8))
    # plt.plot(baseline_model.X_data)
    # plt.xlabel('Time')
    # plt.ylabel('Opinion')
    # plt.title(f'Opinion Evolution from trial {baseline_model.trial}')
    # plt.show()

    # Plot opinion evolution for kwparams model
    # plt.figure(figsize=(12, 8))
    # plt.plot(kw_model.X_data)
    # plt.xlabel('Time')
    # plt.ylabel('Opinion')
    # plt.title(f'Opinion Evolution from trial {kw_model.trial}')
    # plt.show()

    G = nx.karate_club_graph()

    RNG = np.random.default_rng(seed=seed)
    # c = np.round(RNG.uniform(0.1, 1, G.number_of_nodes()), decimals=1)
    
    # run abc model on network
    params = {
        "trial" : 1,
        "max_steps" : 100000,
        "tolerance" : 1e-5,
        "mu" : 0.1,
        "c" : .3,
        "beta" : .25,
        "M" : 1,
        "K" : 5,
        "full_time_series": True,
        "gamma" : 0.1,
        "delta": 0.9
    }

    # model = Model(seed_sequence=seed, **kwparams(10, 0.3, 0.25, 1, 1))
    # model.run(test=True)
    # print(model.edge_changes)

    # plt.xlabel('$\it{t}$')
    # plt.ylabel('Assortativity coefficient')
    # # plt.title('Assortativity Evolution')
    # plt.show()

    # for i in range(0, model.convergence_time, 10):
    #     model.print_graph(time=i, opinions=True)

    # for node in G.nodes():
    #     print(node, model.initial_X[node])
     
    # pos = nx.spring_layout(G, seed=seed)
    # cmap = plt.cm.Blues
    # colors = [model.initial_X[node] for node in list(G.nodes())]
    # plt.title('Network with Initial Opinions')
    # nx.draw(G, pos=pos, node_color=colors, cmap=cmap, with_labels=True, edge_color='gray')

    # Add colorbar
    # sm = plt.cm.ScalarMappable(cmap=cmap)
    # sm.set_array(colors)
    # cbar = plt.colorbar(
    #     sm, ax=plt.gca(), 
    #     shrink=0.65
    # )
    # plt.axis('off')
    # plt.show()

    # model.run(test=True)
    # model.X_data = model.X_data[:model.convergence_time, :]

    # pos = nx.spring_layout(H, seed=seed)
    # nx.draw(H, pos=pos, with_labels=True)
    # plt.show()
    # H = G.copy()
    # nx.set_edge_attributes(H, {(edge[0], edge[1]): {'color': 'gray'} for edge in H.edges})
    # # edge_colors = [H[u][v]['color'] for u, v in H.edges()]

    # for (t, old, new) in model.edge_changes:
    #     colors = [model.X_data[t][node] for node in list(H.nodes())]
    #     cmap = plt.cm.Blues
    #     plt.figure(figsize=(12, 8))
    #     plt.title(f'Karate Club network time: {t}')
    #     pos = nx.spring_layout(H, seed=seed)
    #     nx.draw(H, pos=pos, with_labels=True, edge_color='gray', cmap=cmap, node_color=colors)
    #     # Add colorbar
    #     sm = plt.cm.ScalarMappable(cmap=cmap)
    #     sm.set_array(colors)
    #     cbar = plt.colorbar(
    #         sm, ax=plt.gca(), 
    #         shrink=0.65
    #     )
    #     plt.axis('off')

    #     plt.show()
    #     H.remove_edge(*old)
    #     H.add_edge(*new)

    # else:
    # model.X_data = model.X_data[:int(model.convergence_time / 250) + 1, :]

    # model.num_discordant_edges = model.num_discordant_edges[:model.convergence_time - 1]
    # model.num_discordant_edges = np.trim_zeros(model.num_discordant_edges)

    # for time, g in model.G_snapshots:
    #     colors = [model.X_data[time][node] for node in list(g.nodes())]
    #     plt.figure(figsize=(12, 8))
    #     pos = nx.spring_layout(g, seed=seed)
    #     nx.draw(g, pos=pos, node_colors=colors, with_labels=True)
    #     plt.show()

    # pos = nx.spring_layout(rewired_G, seed=seed)
    # plt.figure(figsize=(12, 8))
    # # # plt.title(G.name)
    # nx.draw(rewired_G, pos=pos, node_color=colors, cmap=cmap, with_labels=True, edge_color='gray')

    # # Add colorbar
    # sm = plt.cm.ScalarMappable(cmap=cmap)
    # sm.set_array(colors)
    # cbar = plt.colorbar(
    #     sm, ax=plt.gca(), 
    #     shrink=0.65
    # )
    # plt.axis('off')
    # plt.show()