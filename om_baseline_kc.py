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
import time

FONT = FontProperties()
FONT.set_family('serif')
FONT.set_name('Times New Roman')
FONT.set_style('italic')
FONTSIZE = 34

SEED_SET = 51399

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

def run_om(k: int, c, t, trial, omega=None):
    SEED = 51399 + trial
    G = nx.karate_club_graph()
    baseline_params = {
            "trial" : f"OM-{trial}",
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
        seed_nodes = model.proposed_solution(k=k)
    elif t == 'new':
        seed_nodes = model.test_new(k=k)

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
    # if omega is None:
    return model, k, t
    # else:
    #     return model, k, t, omega
    
def load_all_om():
    """ LOADING DATA """
    path = 'new_proposed/'


    k_values = [0, 5, 10, 15, 20]
    c_values = [0.3]
    trials = list(range(1, 51))
    types = ['degree']
    num = len(trials)

    params = product(k_values, c_values, types, trials)
    degree_om_values = {}
    degree_om_values_ct = {}
    for k in k_values:
        degree_om_values[k] = 0
        degree_om_values_ct[k] = 0

    for k, c, t, trials in params:
        filename = f'KC_BASELINE_{t}_c-{c}_k-{k}_trial-OM-{trials}.pbz2'
        loaded_model = load(path + filename)
        degree_om_values[k] += loaded_model.overall_opinions
        degree_om_values_ct[k] += loaded_model.convergence_time

    for k in k_values:
        degree_om_values[k] /= num
        degree_om_values_ct[k] /= num



    types = ['min_degree']
    k_values = [0, 5, 10, 15, 20]
    c_values = [0.3]
    trials = list(range(1, 51))
    params = product(k_values, c_values, types, trials)
    min_degree_om_values = {}
    min_degree_om_values_ct = {}
    for k in k_values:
        min_degree_om_values[k] = 0
        min_degree_om_values_ct[k] = 0
        
    for k, c, t, trials in params:
        filename = f'KC_BASELINE_{t}_c-{c}_k-{k}_trial-OM-{trials}.pbz2'
        loaded_model = load(path + filename)
        min_degree_om_values[k] += loaded_model.overall_opinions
        min_degree_om_values_ct[k] += loaded_model.convergence_time

    for k in k_values:
        min_degree_om_values[k] /= num
        min_degree_om_values_ct[k] /= num


    types = ['greedy']
    k_values = [0, 5, 10, 15, 20]
    c_values = [0.3]
    trials = list(range(1, 51))
    params = product(k_values, c_values, types, trials)
    greedy_om_values = {}
    greedy_om_values_ct = {}
    for k in k_values:
        greedy_om_values[k] = 0
        greedy_om_values_ct[k] = 0


    for k, c, t, trials in params:
        filename = f'KC_BASELINE_{t}_c-{c}_k-{k}_trial-OM-{trials}.pbz2'
        loaded_model = load(path + filename)

        greedy_om_values[k] += loaded_model.overall_opinions
        greedy_om_values_ct[k] += loaded_model.convergence_time

    for k in k_values:
        greedy_om_values[k] /= num
        greedy_om_values_ct[k] /= num

    types = ['min_opinion']
    k_values = [0, 5, 10, 15, 20]
    c_values = [0.3]
    trials = list(range(1, 51))
    params = product(k_values, c_values, types, trials)
    min_opinion_om_values = {}
    min_opinion_om_values_ct = {}
    for k in k_values:
        min_opinion_om_values[k] = 0
        min_opinion_om_values_ct[k] = 0

    for k, c, t, trials in params:
        filename = f'KC_BASELINE_{t}_c-{c}_k-{k}_trial-OM-{trials}.pbz2'
        loaded_model = load(path + filename)

        min_opinion_om_values[k] += loaded_model.overall_opinions
        min_opinion_om_values_ct[k] += loaded_model.convergence_time

    for k in k_values:
        min_opinion_om_values[k] /= num
        min_opinion_om_values_ct[k] /= num


    types = ['max_opinion']
    k_values = [0, 5, 10, 15, 20]
    c_values = [0.3]
    trials = list(range(1, 51))
    params = product(k_values, c_values, types, trials)
    max_opinion_om_values = {}
    max_opinion_om_values_ct = {}
    for k in k_values:
        max_opinion_om_values[k] = 0
        max_opinion_om_values_ct[k] = 0

    for k, c, t, trials in params:
        filename = f'KC_BASELINE_{t}_c-{c}_k-{k}_trial-OM-{trials}.pbz2'
        loaded_model = load(path + filename)

        max_opinion_om_values[k] += loaded_model.overall_opinions
        max_opinion_om_values_ct[k] += loaded_model.convergence_time

    for k in k_values:
        max_opinion_om_values[k] /= num
        max_opinion_om_values_ct[k] /= num


    types = ['random']
    k_values = [0, 5, 10, 15, 20]
    c_values = [0.3]
    trials = list(range(1, 51))
    params = product(k_values, c_values, types, trials)
    random_om_values = {}
    random_om_values_ct = {}
    for k in k_values:
        random_om_values[k] = 0
        random_om_values_ct[k] = 0

    for k, c, t, trials in params:
        filename = f'KC_BASELINE_{t}_c-{c}_k-{k}_trial-OM-{trials}.pbz2'
        loaded_model = load(path + filename)

        random_om_values[k] += loaded_model.overall_opinions
        random_om_values_ct[k] += loaded_model.convergence_time

    for k in k_values:
        random_om_values[k] /= num
        random_om_values_ct[k] /= num


    types = ['proposed']
    k_values = [0, 5, 10, 15, 20]
    c_values = [0.3]
    trials = list(range(1, 51))
    params = product(k_values, c_values, types, trials)
    proposed_om_values = {}
    proposed_om_values_ct = {}
    for k in k_values:
        proposed_om_values[k] = 0
        proposed_om_values_ct[k] = 0

    for k, c, t, trials in params:
        filename = f'KC_BASELINE_{t}_c-{c}_k-{k}_trial-OM-{trials}.pbz2'
        loaded_model = load(path + filename)

        proposed_om_values[k] += loaded_model.overall_opinions
        proposed_om_values_ct[k] += loaded_model.convergence_time

    for k in k_values:
        proposed_om_values[k] /= num
        proposed_om_values_ct[k] /= num


    # Save all dictionaries in an object
    all_om_values = {
        'degree': degree_om_values,
        'degree_ct': degree_om_values_ct,
        'min_degree': min_degree_om_values,
        'min_degree_ct': min_degree_om_values_ct,
        'greedy': greedy_om_values,
        'greedy_ct': greedy_om_values_ct,
        'min_opinion': min_opinion_om_values,
        'min_opinion_ct': min_opinion_om_values_ct,
        'max_opinion': max_opinion_om_values,
        'max_opinion_ct': max_opinion_om_values_ct,
        'random': random_om_values,
        'random_ct': random_om_values_ct,
        'proposed': proposed_om_values,
        'proposed_ct': proposed_om_values_ct
    }

    filename = f'ALL_OM_VALUES_BASELINE_KC_NEW.pbz2'
    with bz2.BZ2File(path + filename, 'w') as f:
        pickle.dump(all_om_values, f)
        print(f'saved to file, {filename}')

def plot_om():
    filename = f'ALL_OM_VALUES_BASELINE_KC_NEW.pbz2'
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

def seed_selection_test(k, t=None):
    """ seed selection test """
    baseline_params = {
        "trial" : f"BASELINE_KC",
        "max_steps" : 1_000_000,
        "tolerance" : 1e-5,
        "mu" : 0.1,
        "beta" : 1,     # 1 is DW
        "c" : 0.3,
        "M" : 1,
        "K" : 1,
        "gamma": 0,     # 0 is DW
        "delta": 1,     # 1 is DW
    }
    G = nx.karate_club_graph()
    model = Model(seed_sequence=SEED_SET, G=G, **baseline_params)

    # time algos take to finish
    start_time = time.time()
    model.proposed_solution(k)
    end_time = time.time()

    execution_time = end_time - start_time
    return execution_time

def run_om_test():
    """ new runs"""
    k_values = [0, 5, 10, 15, 20]
    c_values = [0.3]
    trials = list(range(1, 51))

    types = ['proposed']
    params = product(k_values, c_values, types, trials)
    pool = Pool(processes=multiprocessing.cpu_count())
    results = []

    for k, c, type, trials in params:
        print(f'running new: {c}, trial: {type}')
        result = pool.apply_async(run_om, args=(k, c, type, trials))
        results.append(result)

    pool.close()
    pool.join()

    path = 'triads_greedy/'
    for result in results:
        model, k, t = result.get()
        filename = f'KC_BASELINE_greedy-{t}_c-{model.initial_c[0]}_k-{k}_trial-{model.trial}.pbz2'                
        
        with bz2.BZ2File(path + filename, 'w') as f:
            pickle.dump(model, f)
            print(f'saved to file, {t}, {filename}')

def algo_runtime():

    # k_ranges = [0, 5, 10, 15, 20, 25, 30]
    k_ranges = [25]
    trials = list(range(1, 51))

    num_runs = len(k_ranges) * len(trials)

    params = product(k_ranges, trials)

    pool = Pool(processes=multiprocessing.cpu_count())
    results = []

    for k, t in params:
        result = pool.apply_async(seed_selection_test, args=(k, t))
        results.append(result)

    pool.close()
    pool.join()
    TOTAL = 0
    for result in results:
        time = result.get()
        TOTAL += time

    print(TOTAL)
    print(TOTAL / num_runs)
    

if __name__ == '__main__':
    seed_selection_test(5)
    exit()
    # for k in range(0, 31, 5):
    #     start_time = time.time()
    #     seed_selection_test(k)
    #     end_time = time.time()
    #     execution_time = end_time - start_time
    #     print(f"Execution time, k: {execution_time} seconds")
    # algo_runtime()
    # exit()

    # run_om_test()
    """ testing new proposed with heuristics """
    # load_all_om()
    # exit()

    # data = load('new_proposed\ALL_OM_VALUES_BASELINE_KC_NEW.pbz2')
    # loads = ['degree', 'degree_ct', 'min_degree', 'min_degree_ct', 'greedy', 'greedy_ct', 'proposed', 'proposed_ct']

    # for k, v in data.items():
    #     if k in loads:
    #         print(k, v)
    # exit()


    # c_values = [0.1, 0.3, 0.5]
    k_values = [0, 5, 10, 15, 20]
    c_values = [0.3]
    # # types = ['proposed']
    # # # beta_values = [0.1, 0.3, 0.5]
    trials = list(range(1, 51))
    num = len(trials)

    path = 'triads_greedy/'

    # # # num_trials = (len(k_values) * len(c_values) * len(types) * len(beta_values))

    types = ['proposed']
    params = product(k_values, c_values, types, trials)

    # # """ load data see if changes worked """

    new_keys = {}    
    for k, c, type, trials in params:
        filename = f'KC_BASELINE_greedy-{type}_c-{c}_k-{k}_trial-OM-{trials}.pbz2'
        data = load(path + filename)
        if k in new_keys:
            new_keys[k] += data.overall_opinions
        else:
            new_keys[k] = 0
    
    for key, val in new_keys.items():
        new_keys[key] = val / num
    print(new_keys)

    print('done')
    # exit()

    types = ['degree']
    path = 'new_proposed/'
    trials = list(range(1, 51))
    new_params = product(k_values, c_values, types, trials)
    old_keys = {}
    for k, c, type, trials in new_params:
        filename = f'KC_BASELINE_{type}_c-{c}_k-{k}_trial-OM-{trials}.pbz2'
        data = load(path + filename)
        
        if k in old_keys:
            old_keys[k] += data.overall_opinions
        else:
            old_keys[k] = 0
    

    for key, val in old_keys.items():
        old_keys[key] = val / num
    print(old_keys)

    types = ['greedy']
    path = 'new_proposed/'
    trials = list(range(1, 51))
    new_params = product(k_values, c_values, types, trials)
    old_keys = {}
    for k, c, type, trials in new_params:
        filename = f'KC_BASELINE_{type}_c-{c}_k-{k}_trial-OM-{trials}.pbz2'
        data = load(path + filename)
        
        if k in old_keys:
            old_keys[k] += data.overall_opinions
        else:
            old_keys[k] = 0
    

    for key, val in old_keys.items():
        old_keys[key] = val / num
    print(old_keys)


""" old stuff """
# def proposed_solution(self, k, omega=1):
#     triads = self.__find_triads()
#     node_scores = self.__top_k_nodes_from_triads(triads)

#     for key, value in node_scores.items():
#         score = value
#         node_scores[key] = score + (self.original_graph.degree(key)) * omega

#     node_scores = {k: v for k, v in sorted(node_scores.items(), key=lambda item: item[1], reverse=True)}
#     selected_nodes = list(node_scores.keys())[:k]

#     # return selected_nodes
#     return list(self.original_graph.degree(selected_nodes))

# def __top_k_nodes_from_triads(self, triads):
#     node_scores = {}
#     for triad in triads:
#         for node in triad:
#             if node in node_scores:
#                 node_scores[node] += 1
#             else:
#                 node_scores[node] = 1

#     return node_scores

# def __find_triads(self):
#     """
#     Identifies triadic closures in the network G.
#     Returns a list of triads (sets of three nodes).
#     """
#     triads = []
#     for node in self.original_graph.nodes():
#         neighbors = list(self.original_graph.neighbors(node))
#         for i, neighbor in enumerate(neighbors):
#             for other_neighbor in neighbors[i+1:]:
#                 if self.original_graph.has_edge(neighbor, other_neighbor):
#                     triads.append((node, neighbor, other_neighbor))
#     return triads     


    
# old does not work
"""
def TRIADS_TEST(self, k):
        G_copy = self.original_graph.copy()
        for node in G_copy.nodes():
            G_copy.nodes[node]['opinion'] = self.X[node]
            G_copy.nodes[node]['confidence'] = self.c[node]

        selected_nodes = []
        for _ in range(k):
            triad_influence = self.__triadic_influence(G_copy)
            # print(triad_influence)
            max_node = max(triad_influence, key=triad_influence.get)
            selected_nodes.append(max_node)
            G_copy = self.__update_nodes(G_copy)
            
        return selected_nodes

    def __triadic_influence(self, G):
        triadic_influence = {}
        for node in G.nodes():
            triadic_influence[node] = 0
            neighbors = list(G.neighbors(node))
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if G.has_edge(neighbors[i], neighbors[j]):
                        # Calculate influence based on opinion closeness within the triad
                        closeness = min(abs(G.nodes[node]['opinion'] - G.nodes[neighbors[i]]['opinion']),
                                        abs(G.nodes[node]['opinion'] - G.nodes[neighbors[j]]['opinion']),
                                        abs(G.nodes[neighbors[i]]['opinion'] - G.nodes[neighbors[j]]['opinion']))
                        triadic_influence[node] += closeness
        return triadic_influence
    
    # rewire and update opinions
    def __update_nodes(self, G):
        discordant_edges = [(i, j) for i, j in G.edges if abs(G.nodes[i]['opinion'] - G.nodes[j]['opinion']) > self.beta][:self.M]
        for (u, v) in discordant_edges:
            G.remove_edge(u, v)
            # Attempt to rewire
            for candidate in G.nodes():
                if candidate != u and not G.has_edge(u, candidate) and abs(G.nodes[u]['opinion'] - G.nodes[candidate]['opinion']) <= self.beta:
                    G.add_edge(u, candidate)
                    break
        
        # node_pairs = list(G.edges())[:self.K]
        node_pairs = G.edges()

        # for each pair, update opinions in Model and Node
        for u, w in node_pairs:
            # using confidence bound of the receiving agent
            if abs(G.nodes[u]['opinion'] - G.nodes[w]['opinion']) < G.nodes[u]['confidence']:
                # update opinions
                G.nodes[u]['opinion'] = G.nodes[u]['opinion'] + self.mu * (G.nodes[w]['opinion'] - G.nodes[u]['opinion'])
                # update confidence
                G.nodes[u]['confidence'] = G.nodes[u]['confidence'] + self.gamma * (1 - G.nodes[u]['confidence'])
            else:
                # update confidence using repulsion parameter, delta
                G.nodes[u]['confidence'] = self.delta * G.nodes[u]['confidence']

            # check other agent is withing their own bounds
            if abs(G.nodes[w]['opinion'] - G.nodes[u]['opinion']) < G.nodes[w]['confidence']:
                G.nodes[w]['opinion'] = G.nodes[w]['opinion'] + self.mu * (G.nodes[u]['opinion'] - G.nodes[w]['opinion'])

                G.nodes[w]['confidence'] = G.nodes[w]['confidence'] + self.gamma * (1 - G.nodes[w]['confidence'])
            else:
                # update confidence using repulsion parameter, delta
                G.nodes[w]['confidence'] = self.delta * G.nodes[w]['confidence']

        return G
"""


""" 
old new improvements not improving 
def test_new(self, k):
        seed_nodes = []
        node_potentials = {node: self.__calculate_potential(node, self.original_graph) for node in self.original_graph.nodes()}
        
        # Identify communities in the network
        communities = nx.community.greedy_modularity_communities(self.original_graph)

        while len(seed_nodes)  < k:  
            # Select top nodes from each community
            for community in communities:
                community_nodes = list(community)
                if community_nodes:
                    selected_node = max(community_nodes, key=lambda node: node_potentials[node])
                    if selected_node not in seed_nodes:
                        seed_nodes.append(selected_node)
                    
                    # Update potentials based on opinion spread velocity
                    for node in nx.single_source_shortest_path_length(self.original_graph, selected_node).keys():
                        node_potentials[node] = self.__calculate_potential(node, self.original_graph)
                    
                    node_potentials[selected_node] = float('-inf')  # Set potential to very low value

                if len(seed_nodes) >= k:
                    break
            
            # Escape local optima with simulated annealing
            current_score = np.sum([self.__calculate_potential(node, self.original_graph) for node in seed_nodes])
            
            for temp in np.logspace(0, 5, num=100)[::-1]:
                candidate_node = random.choice(list(node_potentials.keys()))
                candidate_score = np.sum([self.__calculate_potential(node, self.original_graph) for node in seed_nodes + [candidate_node]])
                
                if candidate_score > current_score:
                    if candidate_node not in seed_nodes:
                        seed_nodes.append(candidate_node)
                    node_potentials[candidate_node] = float('-inf')  # Set potential to very low value
                    break
                else:
                    prob = np.exp((candidate_score - current_score) / temp)
                    if random.random() < prob:
                        if candidate_node not in seed_nodes:
                            seed_nodes.append(candidate_node)
                        node_potentials[candidate_node] = float('-inf')  # Set potential to very low value
                        break

        return list(self.original_graph.degree(seed_nodes))
"""