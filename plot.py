from re import T
import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot_rew_comp():
    x = np.array([0.99, 0.98, 0.97, 0.96, 0.95])
    y = np.array([9.31, 11.44, 141.47, 186.42, 196.46])

    # first plot with X and Y data
    plt.plot(x, y, "-b", label="Cartpole")

    x1 = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60])
    y1 = np.array([-91.07, -130.08, -58.51, -8.67, 23.50, 93.47, 178.29, 238.30])

    x2 = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50])
    y2 = np.array([-92.70, -119.38, -39.02, -17.92, -79.15, -53.15, 44.69, 85.12, 139.43, 200.88])

    # second plot with x1 and y1 data
    plt.plot(x1, y1, "-r", label="LunarLander")
    plt.plot(x2, y2, "-g", label="BipedalWalker")
    plt.legend(loc="lower left")

    plt.xlabel("Prune Percentage")
    plt.ylabel("Rewards")
    plt.title('Prune Percentage vs Rewards')
    plt.show()

def plot_traj_deviation(org_traj,pred_traj,t):
    x = t
    y = org_traj
    plt.plot(x, y, "-b", label="Original traj")
    x1 = t 
    y1 = pred_traj
    plt.plot(x1, y1, "-r", label="Pred Traj")
    plt.legend(loc="upper left")
    plt.show()


def plot_val(x_lower,x_upper):
    file = open('verlunardata.pkl', 'rb')
    x_lower1 = pickle.load(file)
    x_upper1 = pickle.load(file)
    print(f'len===={len(x_lower1)}========{len(x_upper1)}')
    x_axis = np.arange(start=0, stop=145, step=1)

    y = np.array(x_lower)
    plt.plot(x_axis, y, "-m", marker = 'v')

    y2 = np.array(x_upper)

    y3 = np.array(x_lower1)
    y4 = np.array(x_upper1)

    plt.plot(x_axis, y2, "-b", marker = '^',alpha=1.0)
    plt.plot(x_axis, y3, "-r" , marker = 'v', markersize=2, alpha=1.0)
    plt.plot(x_axis, y4, "-g" , marker = '^', markersize=2, alpha=1.0)
    plt.fill_between(x_axis, y2, y, color='yellow', alpha=0.2)
    plt.fill_between(x_axis, y3, y4, color='orange', alpha=0.2)
    #plt.fill_between(x_axis, y, where=x_axis<172, color='yellow', alpha=0.2)
    #plt.fill_between(x_axis, y2, color='yellow', alpha=0.2)
    plt.axhline(y = 0.4, color = 'r', linestyle = '-')
    plt.axhline(y = -0.4, color = 'r', linestyle = '-')

    plt.show()





