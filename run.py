import multiprocessing
from model import Model
import numpy as np
from multiprocessing import Pool

import matplotlib.pyplot as plt
import networkx as nx

# record data for baseline results
def kwparams(N, c, beta, trial, K):
    params = {
        "trial" : "ABC",
        "max_steps" : 100000,
        "N" : N,
        "p" : 0.1,
        "tolerance" : 1e-5,
        "alpha" : 0.1,
        "beta" : beta,
        "c" : c,
        "M" : 1,
        "K" : K,
        "full_time_series": True,
        "gamma": .1,     # 0 is DW
        "delta": .9,     # 1 is DW
    }
    return params

def baseline_params(N):
    baseline = {
        "trial" : "BASELINE",
        "max_steps" : 100000,
        "N" : N,
        "p" : 0.1,
        "tolerance" : 1e-5,
        "alpha" : 0.1,
        "beta" : 1,
        "c" : 0.3,
        "M" : 1,
        "K" : 5,
        "full_time_series": True,
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


if __name__ == '__main__':
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

    # for i in range(simulations):
    #     RNG = np.random.default_rng(seed=seed+i)
    #     c = np.round(RNG.uniform(0.1, 1, N), decimals=1)    
    #     result = pool.apply_async(run_model, args=(seed+i, kwparams(N, c, beta, 1, K)))
    #     results.append(result)
    
    # Close the pool
    # pool.close()
    # pool.join()

    # Wait for all the simulation tasks to complete
    # for result in results:
    #     model = result.get()
    #     print(f'model: {model.trial}, convergence time: {model.convergence_time}')

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
    c = np.round(RNG.uniform(0.1, 1, G.number_of_nodes()), decimals=1)
    
    # run abc model on network
    params = {
        "trial" : 1,
        "max_steps" : 100000,
        "tolerance" : 1e-5,
        "alpha" : 0.1,
        "c" : c,
        "beta" : .25,
        "M" : 1,
        "K" : 1,
        "full_time_series": True,
        "gamma" : 0.1,
        "delta": 0.9
    }

    # TODO: look at edge changes, and graph them for report on karate club network

    # model = Model(seed_sequence=seed, **params)
    model = Model(seed_sequence=seed, **kwparams(10, 0.3, 0.25, 1, 1))
    model.run(test=True)

    # model.print_graph(time=0, opinions=False)

    for i in range(0, model.convergence_time, 10):
        model.print_graph(time=i, opinions=True)

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