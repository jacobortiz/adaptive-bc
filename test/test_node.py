from node import Node
import numpy as np
import pytest

def test_init():
    initial_opinion = 0.5
    neighbors = [2, 3]
    node_1 = Node(1, initial_opinion, neighbors)
    node_2 = Node(2, initial_opinion)

    assert node_1.id == 1
    assert node_1.initial_opinion == initial_opinion
    assert node_1.current_opinion == initial_opinion
    assert node_1.neighbors == neighbors
    assert len(node_1.neighbors) == 2
    assert len(node_2.neighbors) == 0
    assert node_2.neighbors == []

@pytest.fixture
def test_node():
    id = 1
    initial_opinion = 0.5
    neighbors = [2, 3]
    return Node(id, initial_opinion, neighbors)

def test_add_neighbor(test_node: Node):
    test_node.add_neighbor(4)
    assert test_node.neighbors == [2, 3, 4]

def test_erase_neighbor(test_node: Node):
    test_node.erase_neighbor(2)
    assert test_node.neighbors == [3]

def test_check_neighbor(test_node: Node):
    assert test_node.check_neighbor(2) == True
    assert test_node.check_neighbor(4) == False