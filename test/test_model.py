import networkx as nx
from node import Node
from model import Model
import numpy as np
from numpy.random import RandomState, MT19937, SeedSequence
import random
import pytest
from copy import deepcopy

# fix test seed
seed = 12345

@pytest.fixture
def model_params():
    params = {
        "trial" : 1,
        "max_steps" : 10000,
        "N" : 10,
        "p" : 0.1,
        "tolerance" : 10e-5,
        "alpha" : 0.1,
        "C" : 0.3,
        "beta" : 0.25,
        "M" : 1,
        "K" : 1,
        "full_time_series": False
    }
    return params

@pytest.fixture
def seed_sequence():
    return SeedSequence(seed)

# test nodes if they are equivalent
def __equivalent_nodes(node_1, node_2):
    node_attributes = ['id', 'initial_opinion', 'current_opinion', 'neighbors']
    # return True if node_1, node_2 attributes are equivalent
    return np.all([ getattr(node_1, attr) == getattr(node_2, attr) for attr in node_attributes])

# test model contains params that are passed in model
def test_model_params(seed_sequence, model_params):
    model = Model(seed_sequence, **model_params)
    for k, v in model_params.items():
        assert getattr(model, k) == v
