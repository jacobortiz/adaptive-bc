import networkx as nx
from node import Node
import numpy as np
import random
from random import shuffle

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import pickle
import bz2

import time

# from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

class Model:
    def __init__(self, seed_sequence, G=None, **kwparams) -> None:
        # set random state for model instance to ensure repeatability
        self.seed_sequence = seed_sequence
        self.RNG = np.random.default_rng(seed_sequence)
        random.seed(seed_sequence)

        # set model params
        self.trial = kwparams['trial']                      # trial ID (for saving model)
        self.max_steps = kwparams['max_steps']              # max time steps model runs
        if not G: self.N = kwparams['N']                    # number of nodes
        if not G: self.p = kwparams['p']                    # probability of edge creation, p in G(N, p)
        self.tolerance = kwparams['tolerance']              # convergence tolerance (1-e^5)
        self.mu = kwparams['mu']                            # convergence parameter (0.1)
        self.c = kwparams['c']                              # confidence bound
        self.beta = kwparams['beta']                        # rewiring threshold
        self.M = kwparams['M']                              # num of edges to rewire each step
        self.K = kwparams['K']                              # num of node pairs to update opinions at each step
        self.gamma = kwparams['gamma']                      # confidence attraction parameter
        self.delta = kwparams['delta']                      # confidence repulsion parameter

        
        # store network assortativity, pearson correlation coefficient
        self.assortativity_history = []

        # generate network and set attributes:
        # opinions, initial_opinions, initial_edges, nodes, edges
        self.__initialize_network(G)

        # X is opinion data
        self.X_data = np.ndarray((self.max_steps, self.N))   # storing time series opinion data
        self.X_data[0, :] = self.X                           # record initial opinions
        self.edge_changes = []                                  # record edge changes

        self.num_discordant_edges = np.empty(self.max_steps)# track number of discordant edges
        self.stationary_counter = 0                         # determining if we reached a stationary state
        self.stationary_flag = 0                            # flagging stationary state
        self.convergence_time = None                        # record convergence time

        # if beta == 1, no rewiring, standard BC applies
        self.rewiring = False if int(kwparams['beta'] == 1) else True

        # self.start_assortativity = nx.degree_assortativity_coefficient(nx.Graph(self.edges.copy()))

    def __initialize_network(self, G: nx.Graph) -> None:
        # generate G(N, p) random graph
        if G:
            print(f'===== Running on {G.name} =====')
            self.N = G.number_of_nodes()
            self.original_graph = G.copy()
            self.graph_type = G.name
        else:
            print('===== initializing network =====')
            G = nx.fast_gnp_random_graph(n=self.N, p=self.p, seed=self.seed_sequence, directed=False)
            self.graph_type = 'random Erdős–Rényi graph'
            self.original_graph = G.copy()

        # random initial opinions from [0, 1] uniformly
        opinions = self.RNG.random(self.N)

        # random confidence bounds for each agent if providing list
        if type(self.c) is not np.ndarray:
            self.c = [self.c] * self.N

        self.initial_c = self.c.copy()
        
        nodes = []
        for i in range(self.N):
            node_neighbors = list(G[i])
            node = Node(id=i, initial_opinion=opinions[i], neighbors=node_neighbors, confidence_bound=self.c[i])
            nodes.append(node)

        edges = [(u, v) for u, v in G.edges()]

        self.X = opinions
        self.initial_X = opinions
        self.edges = edges.copy()
        self.initial_edges = edges.copy()
        self.nodes = nodes

        # self.sum_opinions = np.sum(opinions)

        self.assortativity_history.append(nx.degree_pearson_correlation_coefficient(G))

    # run the model
    def run(self, test=False) -> None:
        time = 0
        def rewire_step():
            # get discordant edges
            discordant_edges = [(i, j) for i, j in self.edges if abs(self.X[i] - self.X[j]) > self.beta]
            self.num_discordant_edges[time] = len(discordant_edges)

            # if len of discordant edges >= M, choose M at random to rewire
            # else choose all discordant edges to rewire
            if len(discordant_edges) > self.M:
                shuffle(discordant_edges)
                edges_to_cut = discordant_edges[:self.M]
            else:
                edges_to_cut = discordant_edges

            # remove and make new connections
            for edge in edges_to_cut:
                self.edges.remove(edge)
                i, j = edge[0], edge[1]
                self.nodes[i].erase_neighbor(j)
                self.nodes[j].erase_neighbor(i)

                # pick either i or j to rewire
                i = random.choice([i, j])
                selected_node = self.nodes[i]
                new_neighbor = selected_node.rewire(self.X, self.RNG)
                self.nodes[new_neighbor].add_neighbor(i)
                new_edge = (i, new_neighbor)

                # record data
                self.edges.append(new_edge)

                self.edge_changes.append((time, edge, new_edge))

        # update opinions using DW
        def dw_step():
            # choose K edges for nodes to update opinions
            index = self.RNG.integers(low=0, high=len(self.edges), size=self.K)
            node_pairs = [self.edges[i] for i in index]

            # for each pair, update opinions in Model and Node
            X_new = self.X.copy()
            for u, w in node_pairs:
                # assumptions here
                # using confidence bound of the receiving agent
                if abs(self.X[u] - self.X[w]) < self.c[u]:
                    # update opinions
                    X_new[u] = self.X[u] + self.mu * (self.X[w] - self.X[u])
                    self.nodes[u].update_opinion(X_new[u])
                    # update confidence
                    self.c[u] = self.c[u] + self.gamma * (1 - self.c[u])
                else:
                    # update confidence using repulsion parameter, delta
                    self.c[u] = self.delta * self.c[u]

                # check other agent is withing their own bounds
                if abs(self.X[w] - self.X[u]) < self.c[w]:
                    X_new[w] = self.X[w] + self.mu * (self.X[u] - self.X[w])
                    self.nodes[w].update_opinion(X_new[w])
                    self.c[w] = self.c[w] + self.gamma * (1 - self.c[w])
                else:
                    # update confidence using repulsion parameter, delta
                    self.c[w] = self.delta * self.c[w]

            # update data
            self.X_prev = self.X.copy()
            self.X = X_new

            # update data in time series
            self.X_data[time + 1, :] = X_new
    
        def check_convergence():
            state_change = np.sum(np.abs(self.X - self.X_prev))
            self.stationary_counter = self.stationary_counter + 1 if state_change < self.tolerance else 0
            self.stationary_flag = 1 if self.stationary_counter >= 100 else 0

        l = self.max_steps
        printProgressBar(0, l, prefix = 'Running Model:', suffix = 'Complete', length = 50)
        # run model
        while time < self.max_steps - 1 and self.stationary_flag != 1:
            if self.rewiring: rewire_step()
            dw_step()
            check_convergence()
            time += 1
            self.assortativity_history.append(nx.degree_pearson_correlation_coefficient(nx.Graph(self.edges.copy())))
            printProgressBar(time, l, prefix = 'Running Model:', suffix = 'Complete', length = 50)

        printProgressBar(time, time, prefix = 'Running Model:', suffix = 'Complete', length = 50)

        print(f'Model finished. \nConvergence time: {time}')

        self.convergence_time = time
        self.X_data = self.X_data[:time, :]
        self.num_discordant_edges = self.num_discordant_edges[:self.convergence_time - 1]
        self.num_discordant_edges = np.trim_zeros(self.num_discordant_edges)
        if not test: self.save_model()

    def get_edges(self, time: int = None) -> list:
        if time == None or time >= self.convergence_time:
            return self.edges.copy()
        elif time == 0:
            return self.initial_edges.copy()
        else:
            edges = self.initial_edges.copy()
            # find all edge changes up until t < time
            edge_changes = [(t, e1, e2) for (t, e1, e2) in self.edge_changes if t < time]
            # iteratively make changes to network
            for (_, old_edge, new_edge) in edge_changes:
                edges.remove(old_edge)
                edges.append(new_edge)

            return edges

    def get_network(self, time: int = None) -> nx.Graph:
        if time == 0:
            return self.original_graph
        
        G = nx.Graph()
        G.add_nodes_from(range(self.N))

        edges = self.get_edges(time)
        G.add_edges_from(edges)
        return G

    def get_opinions(self, time: int = None) -> np.ndarray:
        if time == None or time >= self.convergence_time:
            return self.X_data.copy()
        else:
            return self.X_data[:time + 1, :]
        
    def print_graph(self, time: int = None, opinions: bool = False, k: float = 0.25) -> None:
        G = self.get_network(time)

        # print last time step if not providing time
        if time is None:
            time = -1

        labels = False
        width = 1
        cmap = None if not opinions else plt.cm.get_cmap('viridis')
        colors = 'skyblue' if not opinions else [self.X_data[time][node] for node in list(G.nodes())]

        # print(f'{time}: {self.X_data[time]}')
        node_size = 600

        if G.number_of_nodes() < 100:
            labels=True

        elif G.number_of_edges() > 500:
            width = 0.2
            node_size = 50

        font = FontProperties()
        font.set_family('serif')
        font.set_name('Times New Roman')
        font.set_style('italic')

        fontsize = 20

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=self.seed_sequence, k=k)
        nx.draw(G, pos=pos, node_color=colors, node_size=node_size, edge_color='gray', width=width, cmap=cmap, with_labels=labels, node_shape='o')

        # move opinion colors here
        if opinions:
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array(colors)
            cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.75)
            cbar.ax.tick_params(labelsize=fontsize)  # set fontsize of colorbar
            plt.axis('off')

        plt.show()
        

    def save_model(self, filename=None):
        self.X_data = self.X_data[:self.convergence_time, :]
        self.num_discordant_edges = self.num_discordant_edges[:self.convergence_time - 1]
        self.num_discordant_edges = np.trim_zeros(self.num_discordant_edges)

        if not filename:
            filename='data/test.pbz2'

        print(f'saving model to {filename}')
        with bz2.BZ2File(filename, 'w') as f:
            pickle.dump(self, f)

    def info(self) -> None:
        print('===== Model parameters =====')
        print(f'Seed: {self.seed_sequence}')
        print(f'Trial number: {self.trial}')
        print(f'Max time steps: {self.max_steps}')
        print(f'Number of nodes: {self.N}')
        print(f'Edge creation probability: {self.p}')
        print(f'Convergence tolerance: {self.tolerance}')
        print(f'Convergence parameter: {self.mu}')
        print(f'Rewiring threshold: {self.beta}')
        print(f'Edges to rewire at each time step, M: {self.M}')
        print(f'Node pairs to update opinions, K: {self.K}')

    def degree_solution(self, k: int):
        " select top k nodes ranked in decreasing order by their degree "        
        return sorted(self.original_graph.degree, key=lambda x: x[1], reverse=True)[:k]
    
    def random_solution(self, k: int):
        " randomly select k nodes "
        nodes = random.sample(list(self.original_graph.nodes()), k)
        return list(self.original_graph.degree(nodes))
    
    def min_opinion_solution(self, k: int):
        " select nodes with smallest opinions to adopt opinion 1"
        selected_nodes = sorted(range(len(self.X)), key=lambda i: self.X[i])[:k]
        return list(self.original_graph.degree(selected_nodes))

    def max_opinion_solution(self, k: int):
        " select nodes with smallest opinions to adopt opinion 1"
        selected_nodes = sorted(range(len(self.initial_X)), key=lambda i: self.initial_X[i], reverse=True)[:k]
        return list(self.original_graph.degree(selected_nodes))
    
    def greedy_solution(self, k: int):
        pass