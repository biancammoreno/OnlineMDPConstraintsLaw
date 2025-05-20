from FourRooms import FourRoomsEnv
from online_mirror_descent import MirrorDescent
from online_mirror_descent_set2 import MirrorDescentSet2
import matplotlib.pyplot as plt
import os
import numpy as np


def color_walls_white(env, mu):
    for coor in env.grid.wall_cells:
        x = coor[1] * env.size + coor[0]
        mu[x] = -1000
    mu = mu.reshape((env.size,env.size))
    mu_masked = np.ma.masked_less(mu, 0)

    return mu_masked

def show_final_distribution(env, mu, algo, max_steps, reward_type, iter, online):
    _EPSILON = 10**(-3)
    # Print the whole final distribution
    mu_traj = 0
    for step in range(max_steps):
        mu_traj_step = color_walls_white(env, mu[step,:])
        plt.imshow(mu_traj_step)
        plt.colorbar()
        plt.savefig('results/frames/' + reward_type + '_algo_' + algo + '_mu_traj_iter_' + str(iter)  + '_step_' + str(step) + '.png')
        plt.close()


def plot_regret(gamma, n_iterations, algo, obj_function, rho_diff, max_steps, reward_type):
    # load info about optimal periodic policy
    obj_function_known_p = np.load('knownp/' + reward_type + '/obj_function_' + reward_type + '_step_' + str(max_steps) + '_known_p.npy')
    rho_diff_known_p = np.load('knownp/' + reward_type + '/rho_diff_' + reward_type + '_step_' + str(max_steps) + '_known_p.npy')
    optimal_obj_function = obj_function_known_p[-1]

    regret = np.cumsum(obj_function + gamma * rho_diff - optimal_obj_function - gamma * rho_diff_known_p[0:n_iterations])

    # plot
    plt.plot(regret, label=algo, c='b')
    plt.xlabel('Episodes')
    plt.title('Regret')
    plt.legend()
    plt.savefig('results/regret_' + reward_type + '_algo_' + algo + '.png')
    plt.close()


def main(algo, reward_type, n_iterations, gamma=1000, online=True):

    if reward_type == 'entropy_max':
        max_steps = 40
    elif reward_type == 'obstacles':
        max_steps = 80
    else:
        print('reward does not exist, try among: entropy_max, obstacles.')
        return

    # # Create directory to add results if it does not exist
    isExist = os.path.exists('results')
    if not isExist:
        os.makedirs('results')

    # # Create directory to add state distributions if it does not exist
    isExist = os.path.exists('results/frames')
    if not isExist:
        os.makedirs('results/frames')

    # define environment 
    env = FourRoomsEnv(max_steps=max_steps)
    # original initial state distribution
    nu0 = env.initial_state_dist()
    obs = env.reset(nu0)
    P_model = env.P(env.p) # true probability transition kernel
    if algo == 'episodic':
        model = MirrorDescent(env, online, gamma, episodic = True, reward_type=reward_type)
    elif algo == 'MDPP-K':
        model = MirrorDescent(env, online, gamma, episodic = False, reward_type=reward_type)
    elif algo == 'MDPP-U':
        model = MirrorDescentSet2(env, online, reward_type=reward_type)
    else: 
        print('algorithm does not exist, try among: MDPP-K, MDPP-U, episodic.')
        return 

    # run for n_iterations
    model.iteration(n_iterations=n_iterations)
    
    # save state distribution after n_iterations for all time steps
    init_dist = model.init_dist
    mu = model.mu_induced(model.policy, P_model,  init_dist)
    show_final_distribution(env, mu, algo, max_steps, reward_type,  model.count_step, online)

    # plot the regret 
    obj_function = np.array(model.error)
    rho_diff = np.array(model.rho_diff)
    plot_regret(gamma, n_iterations, algo, obj_function, rho_diff, max_steps, reward_type)


if __name__ == '__main__':
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--algo', type=str, required=True, help='algorithm to use: MDPP-K, MDPP-U or episodic')
    parser.add_argument('--reward_type', type=str, required=True, help='type of reward: entropy_max or obstacles')
    parser.add_argument('--n_iterations', type=int, required=True, help='number of iterations')
    parser.add_argument('--samples', type=int, required=False, help='number of repetitions per experiment, default is 1')
    parser.add_argument('--online', type=ast.literal_eval, required=False, help='True or False: if online (unknown MDP) or not (known MDP) default is True')
    args = parser.parse_args()

    main(algo=args.algo, reward_type=args.reward_type, n_iterations=args.n_iterations, online=args.online)
