import multiprocessing
from model import Model
from node import Node
import numpy as np
from multiprocessing import Pool

import matplotlib.pyplot as plt

# record data for baseline results
def kwparams(N, c, beta, trial, K):
    params = {
        "trial" : trial,
        "max_steps" : 100000,
        "N" : N,
        "p" : 0.1,
        "tolerance" : 1e-5,
        "alpha" : 0.1, # 0.5 is consensus parameter
        "beta" : beta,
        "c" : c,
        "M" : 1,
        "K" : K,
        "full_time_series": True,
        "gamma": 0.01,
        "delta": 0.1
    }
    return params

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
    beta = 1
    K = 5

    # convergence times are too high, investigate!
    simulations = 1
    average_opinions = 0
    pool = Pool(processes=10)  # Set the maximum number of workers to 10
    results = []

    for i in range(simulations):
        RNG = np.random.default_rng(seed=seed+i)
        c = np.round(RNG.uniform(0.1, 1, N), decimals=1)
        model_params = kwparams(N, c, beta, trial, K)
        result = pool.apply_async(run_model, args=(seed+i, kwparams(N, c, beta, trial+i, K)))
        results.append(result)

    pool.close()
    pool.join()

    # percentage_increases = []
    for result in results:
        model = result.get()
        print(model.convergence_time)
        # sum_opinions = np.sum(model.X_data, axis=1)

        # increase_opinions = sum_opinions[-1] - sum_opinions[0]
        # percentage_increase = (increase_opinions / sum_opinions[0]) * 100
        # average_opinions += percentage_increase
        # percentage_increases.append(percentage_increase)

        # # Plot opinion evolution
        # plt.figure(figsize=(12, 8))
        # plt.plot(model.X_data)   # convergence time is too 
        # plt.xlabel('Time')
        # plt.ylabel('Opinion')
        # # plt.legend(['Nodes in Network'], bbox_to_anchor=(1.3, 1), loc='upper right')
        # # plt.annotate(f'$confidence$ $\epsilon = {loaded_model.C}$', xy=(1.05,.8), xycoords='axes fraction', fontsize=12)
        # plt.title(f'Opinion Evolution from trial {model.trial}')
        # plt.show()

    

    # average_percentage_increase = average_opinions / simulations
    # print(f"Average percentage increase in opinions: {average_percentage_increase:.2f}%")

    # # Plot percentage increases
    # plt.figure(figsize=(12, 8))
    # plt.plot(percentage_increases)
    # plt.xlabel('Simulation')
    # plt.ylabel('Percentage Increase')
    # plt.title('Percentage Increase in Opinions')
    # plt.show()

    # sum_opinions = np.sum(model.X_data, axis=1)
    # print(sum_opinions[0], sum_opinions[-1])

    # plt.figure(figsize=(12, 8))
    # plt.plot(sum_opinions)
    # plt.xlabel('Time')
    # plt.ylabel('Sum of Opinions')
    # plt.title('Sum of Opinions Evolution')
    # plt.show()
