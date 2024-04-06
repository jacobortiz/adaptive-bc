import numpy as np
import pickle
import bz2

import matplotlib.pyplot as plt

import networkx as nx

if __name__ == '__main__':
    # K_list = [1, 5, 10, 20]
    # for K in K_list:
    #     file = f'baseline/baseline-ABC-K_{K}-C_1-beta_1-N_100k'
    #     loaded_model = bz2.BZ2File(f'data/{file}.pbz2', 'rb')
    #     loaded_model = pickle.load(loaded_model)
    #     print('\n')
    #     loaded_model.info()
    #     print(f'Convergence time for K={K}: {loaded_model.convergence_time}')
    #     # print(f'Opinion evolution for K={K}: \n{loaded_model.X_data[-1]}')
    #     if loaded_model.beta == 1:
    #         print(f'No rewiring for K={K}, beta is 1')
    #     else:
    #         print(f'Rewiring threshold for K={K}: {loaded_model.beta}')
    #         print(f'Edge changes for K={K}: \n{loaded_model.edge_changes}')

        
    # load the model
    file = 'adaptive_bc-no_rewire-no_c_changes'

    loaded_model = bz2.BZ2File(f'data/{file}.pbz2', 'rb')
    loaded_model = pickle.load(loaded_model)

    loaded_model.info()

    # convergence time
    print(f'\nConvergence time: {loaded_model.convergence_time}')

    # see opinion history
    print(f'Opinion evolution: \n{loaded_model.X_data}')

    # see all edge changes, if beta is 1, no wiring since all opinions are within threshold
    if loaded_model.beta == 1:
        print('No rewiring, beta is 1')
    else:
        print(f'Rewiring threshold: {loaded_model.beta}')
        print(f'Edge changes: \n{loaded_model.edge_changes}')

    

    # Plot opinion evolution
    plt.figure(figsize=(12, 8))
    plt.plot(loaded_model.X_data)   # convergence time is too 
    plt.xlabel('Time')
    plt.ylabel('Opinion')
    # plt.legend(['Nodes in Network'], bbox_to_anchor=(1.3, 1), loc='upper right')
    # plt.annotate(f'$confidence$ $\epsilon = {loaded_model.C}$', xy=(1.05,.8), xycoords='axes fraction', fontsize=12)
    plt.title(f'Opinion Evolution - Adaptive-BC: no rewire, no confidence updates')
    plt.show()
