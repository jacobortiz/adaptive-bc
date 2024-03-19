import numpy as np

class Node:
    def __init__(self, id, initial_opinion, neighbors=None) -> None:
        self.id = id
        self.initial_opinion = initial_opinion
        self.neighbors = neighbors if neighbors is not None else []

        self.current_opinion = initial_opinion
        self.total_opinion_change = 0

    def add_neighbor(self, id) -> None:
        self.neighbors.append(id)

    def erase_neighbor(self, id) -> None:
        self.neighbors.remove(id)

    def check_neighbor(self, id) -> bool:
        return id in self.neighbors
    
    def update_opinino(self, new_opinion) -> None:
        self.total_opinion_change += abs(self.current_opinion - new_opinion)
        self.current_opinion = new_opinion

    def rewire(self, X, RNG):
        pass

    def rewire_probability(self, X):
        pass