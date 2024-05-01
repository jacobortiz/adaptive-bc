import numpy as np
import random
import matplotlib.pyplot as plt

import networkx as nx

SEED = 51399
random.seed(SEED)
RNG = np.random.default_rng(SEED)

class DeffuantWeisbuch:
    def __init__(self, G, mu, epsilon, max_iterations, tol):
        self.G = G
        self.num_agents = G.number_of_nodes()
        self.mu = mu  # convergence parameter
        self.epsilon = epsilon  # confidence bound
        self.max_iterations = max_iterations
        self.opinions = RNG.random(self.num_agents)
        self.history = [self.opinions.copy()]
        self.tol = tol
        self.converged = False

    def run(self):
        edges = list(self.G.edges())
        for _ in range(self.max_iterations):
            i, j = random.choice(edges)

            print(i, j)
            if abs(self.opinions[i] - self.opinions[j]) < self.epsilon:
                self.opinions[i] += self.mu * (self.opinions[j] - self.opinions[i])
                self.opinions[j] += self.mu * (self.opinions[i] - self.opinions[j])
            self.history.append(self.opinions.copy())

            # Check for convergence
            if self.check_convergence():
                self.converged = True
                return

    def check_convergence(self):
        if len(self.history) > 1:
            last_opinions = self.history[-1]
            second_last_opinions = self.history[-2]
            if np.allclose(last_opinions, second_last_opinions, atol=self.tol):
                return True
        return False

    def plot(self):
        plt.figure(figsize=(8, 6))
        for i in range(self.num_agents):
            plt.plot(range(len(self.history)), [h[i] for h in self.history])
        plt.xlabel('Iteration')
        plt.ylabel('Opinion')
        plt.title(f'Deffuant-Weisbuch Model (N={self.num_agents}, μ={self.mu}, ε={self.epsilon})')
        plt.show()

# Parameters
tol = 1e-5
mu = 0.1
epsilon = .5
max_iterations = 100_000

G = nx.karate_club_graph()

# Run the model
model = DeffuantWeisbuch(G, mu, epsilon, max_iterations, tol)
for i in range(max_iterations):
    model.run()
    if model.converged: break

print(model.history[0])
print(model.history[-1])
model.plot()