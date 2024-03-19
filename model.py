import networkx as nx
from node import Node
import numpy as np
from numpy.random import RandomState, MT19937
import random

class Model:
    def __init__(self, seed_sequence, **kwparams) -> None: 
        # set random state for model instance to ensure repeatability
        self.seed_sequence = seed_sequence
        try:
            self.spawn_key = seed_sequence.spawn_key[0]
        except:
            self.spawn_key = None

        # each instance gets its own RNG
        self.RNG = np.random.default_rng(seed_sequence)
        # set state of random module
        random.seed(seed_sequence)
        self.random_state = RandomState(MT19937(seed_sequence))

        # set model params
        self.trial = kwparams['trial']                      # trial ID (for saving model)
        self.max_steps = kwparams['max_steps']              # bailout time
        self.N = kwparams['N']                              # number of nodes
        self.p = kwparams['p']                              # p in G(N, p)
        self.tolerance = kwparams['tolerance']              # convergence tolerance
        self.alpha = kwparams['alpha']                      # convergence parameter
        self.C = kwparams['C']                              # confidence bound
        self.beta = kwparams['beta']                        # rewiring threshold
        self.M = kwparams['M']                              # num of edges to rewire each step
        self.K = kwparams['K']                              # num of node pairs to update opinions at each step
        self.full_time_series = kwparams['full_time_series']       # save time series opinion data

        # generate network and set attributes:
        # opinions, initial_opinions, initial_edges, nodes, edges
        self.__initialize_network()

        # X = opinion data
        if self.full_time_series:
            self.X_data = np.ndarray((self.max_steps, self.N))   # storing time series opinion data
            self.X_data[0, :] = self.X                           # record initial opinions
            self.edge_changes = []                          # record edge changes
        else:
            self.X_data = np.ndarray((
                int(self.max_steps / 250) + 1, self.N))     # store opinions every 250 time steps
            self.X_data[0, :] = self.X                           # record initial opinions
            self.G_snapshots = []                           # record network snapshots every 250 time steps

        self.num_discordant_edges = np.empty(self.max_steps)# track number of discordant edges
        self.stationary_counter = 0                         # determining if we reached a stationary state
        self.stationary_flag = 0                            # flagging stationary state
        self.convergence_time = None                        # record convergence time

        # if beta == 1, no rewiring, standard BC applies
        self.rewiring = False if int(kwparams['beta'] == 1) else True

    def __initialize_network(self) -> None:
        # random initial opinions from [0, 1] uniformly
        opinions = self.RNG.random(self.N)
        # generate G(N, p) random graph
        G = nx.fast_gnp_random_graph(n=self.N, p=self.p, seed=self.random_state, directed=False)
        
        nodes = []
        for i in range(self.N):
            n_neigh = list(G[i])
            # print(f'node neighbors {n_neigh}')
            node = Node(id=i, initial_opinion=opinions[i], neighbors=n_neigh)
            nodes.append(node)

        edges = [(u, v) for u, v in G.edges()]

        self.X = opinions
        self.initial_X = opinions
        self.edges = edges.copy()
        self.initial_edges = edges.copy()
        self.nodes = nodes
        
    def run(self) -> None:
        pass
    
    def get_edges(self):
        pass

    def get_network(self):
        pass

    def save_model(self):
        pass

