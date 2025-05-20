from dataclasses import dataclass

import gym
import numpy as np
from gym import spaces
from numpy import linalg as LA

_EPSILON = 10**(-25)


class MirrorDescent:
    """
    Class to compute Online Mirror Descent with changing transition costs
    """

    def __init__(self, env, online, gamma, episodic=False, n_agents=100, lr= 0.01, reward_type='entropy_max'):
        self.env = env
        self.lr = lr
        
        # useful as a shortcut
        self.S = env.size * env.size
        self.A = env.action_space.n
        self.N_steps = env.max_steps
        
        # number of agents to observe
        self.n_agents = n_agents

        # regularization parameter for episodic algorithm
        self.gamma = gamma

        # state action counts
        self.n_counts = np.zeros((self.S, self.A))
        self.m_counts = np.zeros((self.S, self.A, self.S))

        # initial probability transition
        if online == True:
            self.P = np.ones((self.S, self.A, self.S))/self.S # online case where we estimate p
        else:
            self.P = self.env.P(self.env.p) # true P for offline case

        # initial policy
        self.policy = np.ones((self.N_steps, self.S, self.A))/self.A

        # initial state distribution sequence
        self.mu = np.zeros((self.N_steps, self.S))

        # initial state-action value function
        self.Q = np.zeros((self.N_steps, self.S, self.A))
        self.sum_Q = np.zeros((self.N_steps, self.S, self.A))

        # inital algorithm count step
        self.count_step = 0

        # set target
        self.target = self.S - 1

        # count number of couples (s,a) visited
        self.n_state_action_visited = []

        # reward type
        self.reward_type = reward_type

        # true probability kernel
        self.true_P = self.env.P(self.env.p) # true P for objective function

        # constrained states
        if self.reward_type == 'obstacles':
            self.constrained = self.env.grid.constrained_states()

        # multiple objectives
        if self.reward_type == 'multi_objectives':
            self.multi_objectives = [(self.env.size-1) * self.env.size, self.env.size-1, (self.env.size-1) * self.env.size + self.env.size-1]

        # noise parameters initialization
        self.noise_params = np.zeros(5)

        # Lagrange multiplier
        self.lambda_ = 0 

        # dual learning rate
        self.tau = 0.01

        # original initial state distribution
        self.nu0 = self.env.initial_state_dist()

        # if online case (with unknown p)
        self.online = online

        # contraction parameter
        self.alpha = 0.1

        # if episodic algorithm or not
        self.episodic = episodic


    def reward(self, mu, x):
        """
        mu = vector of size N (\mu(x) for all n \in [N])
        """
        if self.reward_type == 'entropy_max':
            r_mu = - np.log(mu[:,x] + _EPSILON)

        elif self.reward_type == 'multi_objectives':
            if x in self.multi_objectives:
                r_mu = 2*(1-mu[:,x])
            else:
                r_mu = np.zeros(self.N_steps)

        elif self.reward_type == 'obstacles':
            if x == ((self.env.size-1) * self.env.size + self.env.size - 1):
                r_mu = np.ones(self.N_steps) * 50
            elif x in self.constrained:
                r_mu = -50 *sum(50 * mu[:,state] for state in self.constrained)
            else:
                r_mu = np.zeros(self.N_steps)

        if self.episodic == False:
            r_mu[-1] -= self.lambda_ * 2 * (1 - self.nu0[x])
        else:
            # compute regularization in the last distribution
            r_mu[-1] -= (mu[-1,x] - self.nu0[x]) * self.gamma 

        return r_mu

    def objective_function(self, policy, dist):
        """
        computes the objective function
        dist = current initial distribution
        """
        mu_dist = self.mu_induced(self.policy, self.true_P, dist)
        step = -1

        if self.reward_type == 'entropy_max':
            obj = 0
            for n in range(self.N_steps):
                mu = mu_dist[n,:]
                obj += np.dot(mu, np.log(mu + _EPSILON))/np.log(self.S - len(self.env.grid.wall_cells)+1)
            return -obj 

        elif self.reward_type == 'multi_objectives':
            obj = 0
            for n in range(self.N_steps):
                for x in self.multi_objectives:
                    obj += (mu_dist[n,x])**2
                
            return -obj 

        elif self.reward_type == 'obstacles':
            x_target = (self.env.size-1) * self.env.size + self.env.size - 1
            obj = 0
            # Sum loss for each time step
            for n in range(self.N_steps):
                obj += mu_dist[n, x_target] * 10
                obj -= np.maximum(0,sum(50 * mu_dist[n,x] for x in self.constrained))**2/2
            return -obj


    def softmax(self, y, pi):
        """softmax function
        Args:
          y: vector of len |A|
          pi: vector of len |A|
        """
        max_y = max(y)
        exp_y = [np.exp(self.lr * (y[a] - max_y)) for a in range(y.shape[0])]
        norm_exp = sum(exp_y)
        return [l / norm_exp for l in exp_y]

    def policy_from_logit(self, Q, prev_policy):
        """Compute policy from Q function
        """
        policy = np.zeros((self.N_steps, self.S, self.A))
        for n in range(self.N_steps):
            for x in range(self.S):
                policy[n,x,:] = self.softmax(Q[n,x,:], prev_policy[n,x,:])
                # assert np.sum(policy[n,x,:]) == 1,  'policy should sum to 1'
        
        return policy


    def dual_update(self, mu, old_lambda):
        """
        Update the Lagrange multiplier used in the dual iteration
        lambda = vector of size 2 * |X| (number of constraints)
        """
        if self.online == True:
            prod = 0
            for n in range(self.N_steps):
                n_counts_x = np.sum(self.n_counts, axis=1)
                bonus = 1/np.maximum(np.sqrt(n_counts_x),1) 
                prod += np.dot(bonus.flatten(), mu[n,:].flatten())
            self.gap = LA.norm(mu[-1,:] - self.nu0, 1) - self.alpha * LA.norm(mu[0,:] - self.nu0, 1) - prod
            new_lambda = np.maximum(old_lambda + 1/self.tau * self.gap , 0)
        else:
            self.gap =  LA.norm(mu[-1,:] - self.nu0, 1)
            new_lambda = np.maximum(old_lambda + 1/self.tau * self.gap, 0)

        return new_lambda

    def state_action_value(self, mu, policy):
        """
        Computes the state-action value function
        (without updating pi)
        """
        Q = np.zeros((self.N_steps, self.S, self.A))

        reward = np.zeros((self.N_steps, self.S))
        for x in range(self.S):
            reward[:,x] = self.reward(mu,x)
            Q[self.N_steps-1,x,:] = reward[self.N_steps-1,x] 

        for n in range(self.N_steps - 1, 0, -1):
            for x in range(self.S):
                for a in range(self.A):
                    Q[n-1,x,a] = reward[n-1,x] 
                    for x_next in range(self.S):
                        Q[n-1,x,a] += self.P[x,a,x_next] * np.dot(policy[n, x_next,:], Q[n,x_next,:])

        return Q


    def bonus_state_action_value(self, mu, policy):
        """
        Computes the state-action value function
        (without updating pi)
        """
        Q = np.zeros((self.N_steps, self.S, self.A))

        reward = np.zeros((self.N_steps, self.S))
        for x in range(self.S):
            reward[:,x] = self.reward(mu,x)
            Q[self.N_steps-1,x,:] = reward[self.N_steps-1,x] 

        for n in range(self.N_steps - 1, 0, -1):
            for x in range(self.S):
                for a in range(self.A):
                    Q[n-1,x,a] = reward[n-1,x] + self.C * (1) * (self.N_steps - n)/np.maximum(np.sqrt(self.n_counts[x,a]),1) 
                    for x_next in range(self.S):
                        Q[n-1,x,a] += self.P[x,a,x_next] * np.dot(policy[n, x_next,:], Q[n,x_next,:])

        return Q

    def mu_induced(self, policy, P, initial_dist):
        """
        Computes the state distribution induced by a policy
        """
        mu = np.zeros((self.N_steps, self.S))
        mu[0,:] = initial_dist
        for n in range(1,self.N_steps):
            for x in range(self.S):
                for x_prev in range(self.S):
                    mu[n, x] += mu[n-1, x_prev] * np.dot(policy[n-1, x_prev, :], P[x_prev, :, x])   

        # np.testing.assert_array_equal(np.sum(mu, axis=1), np.ones(self.N_steps), 'proba density should sum to 1')
        return mu 

    
    def iteration(self, n_iterations):
        """
        """
        if self.count_step == 0:
            self.error = []
            self.mu = self.mu_induced(self.policy, self.P, self.nu0)
            self.gap = 10
            self.init_dist = self.nu0
            self.rho_diff = []
            self.sum_cum_rho_diff = []
            self.C = 1
            
        for iter in range(n_iterations):
            print('iteration', iter)
            self.count_step += 1

            # 1) Update the state-value function
            if self.online == True:
                self.Q = self.bonus_state_action_value(self.mu, self.policy)
            else:
                self.Q = self.state_action_value(self.mu, self.policy)
            # 2) Compute the policy associated
            sum_Q_lamb = self.Q + self.sum_Q
            policy_lamb = self.policy_from_logit(sum_Q_lamb, self.policy)
            # 3) Update the state-action distribution (the initial dist. equal to the final one), or the offline case 
            if self.episodic == False:
                mu_lamb = self.mu_induced(policy_lamb, self.P, self.init_dist)
            else:
                mu_lamb = self.mu_induced(policy_lamb, self.P, self.nu0)
            # 4) Update the lagrange multiplier
            if self.episodic == False:
                self.lambda_ = self.dual_update(mu_lamb, self.lambda_)

            # final updates
            self.sum_Q = sum_Q_lamb
            self.policy = policy_lamb
            self.mu = mu_lamb # distribution induced in the estimated MDP
            # 5) Compute objective function value at the last time step
            self.error.append(self.objective_function(self.policy, self.init_dist))
            # 6) Update the probability transitions if needed (online case)
            if self.online == True:
                self.P, self.mu_empirical = self.estimate_transition()
            # 7) Update next initial distribution as the final distribution on previous episode on the true MDP
            true_mu = self.mu_induced(self.policy, self.true_P, self.init_dist)
            self.init_dist = true_mu[-1,:]
            # 8) Compute L1 difference between initial distribution and nu0 to plot
            self.rho_diff.append(LA.norm(self.init_dist - self.nu0, 1))
            self.sum_cum_rho_diff.append(np.sum(self.rho_diff))
        

    def sample_policy(self, n, state):
        return np.random.choice(self.A, p=self.policy[n, state,:])

    def estimate_transition(self):
        """
        Estimate transitions from n_agents playing the current policy
        """
        n_steps = self.n_agents * self.env.max_steps

        P = np.zeros((self.S, self.A, self.S))
        mu_empirical = np.zeros((self.N_steps, self.S))

        observation = self.env.reset(self.init_dist)
        state = self.env.obs_to_state(observation)
        for n in range(n_steps):
            # 1. Sample an action using the policy
            time_step = n % self.env.max_steps
            action = self.sample_policy(time_step, state)
            # 2. Step in the env using this random action
            observation, reward, terminated, truncated, info = self.env.step(action)
            next_state = self.env.obs_to_state(observation)
            # 3. Update state-action counts
            mu_empirical[time_step, state] += 1
            self.n_counts[state, action] += 1
            self.m_counts[state, action, next_state] += 1
            state = next_state.copy()

            if terminated or truncated: # we reach the end of an episode
                # env should be reset with the actual initial dist for this episode
                observation = self.env.reset(self.init_dist)
                state = self.env.obs_to_state(observation)

        for s in range(self.S):
            P[:,:,s] = self.m_counts[:,:,s]/np.maximum(1, self.n_counts)

        mu_empirical = mu_empirical/self.n_agents

        n_a_s = np.argwhere(np.sum(P, axis =2) == 0)
        for i in range(len(n_a_s)):
            P[n_a_s[i][0], n_a_s[i][1],:] =  1/self.S 
        self.n_state_action_visited.append(self.S*self.A - len(n_a_s))

        return P, mu_empirical

        


        
