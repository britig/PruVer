from matplotlib import pyplot as plt
import matplotlib
import numpy as np


def visualize_model_weight(policy):

    '''
    Visualize the weights of the sparse model
    For weights with values of 0, they will be represented by colour white
    '''

    my_cmap = matplotlib.cm.get_cmap('rainbow')
    my_cmap.set_under('w')
    data = {}

    for name, param in policy.named_parameters():
        if 'weight' in name:
            # print(f'Name========={name} ======= param ====== {param.detach().numpy().size}')
            name = name.split(".")[0]
            data[name] = param.detach().numpy()

    for key,val in data.items():
        weight_matrix = data[key]
        weight_matrix = np.resize(weight_matrix,(1, weight_matrix.size))
        plt.imshow(np.abs(weight_matrix),
        interpolation='none',
        aspect = "auto",
        cmap = my_cmap,
        vmin = 1e-26) # lower bound is closed to but not at 0
        plt.colorbar()
        plt.title(key)
        plt.show()

