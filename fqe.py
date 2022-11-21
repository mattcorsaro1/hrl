import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import os
import time

import torch
import torch.nn as nn

from hrl.agent.td3.TD3AgentClass import TD3
from hrl.agent.td3.utils import load as load_agent

import pdb

import argparse

RESULT_DIR = "/users/mcorsaro/scratch/"

class QFitter(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        value = self.linear_relu_stack(x)
        return value

def chunked_policy_prediction(policy, states, action_dim, device, chunk_size=100):
    data_size = states.size(dim=0)
    actions = torch.zeros(data_size, action_dim).to(device)
    num_whole_chunks = data_size // chunk_size
    for i in range(num_whole_chunks):
        actions[i*chunk_size:(i+1)*chunk_size-1, :] = policy(states[i*chunk_size:(i+1)*chunk_size-1, :])
    if data_size % chunk_size != 0:
        actions[num_whole_chunks*chunk_size:, :] = policy(states[num_whole_chunks*chunk_size:, :])
    return actions

class FQE:
    def __init__(self,
                 state_dim,
                 action_dim,
                 pi_eval,
                 learning_rate=0.001,
                 device='cpu',
                 exp_name="tmp"):
        self.pi_eval = pi_eval
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.q_fitter = QFitter(self.state_dim, self.action_dim).to(device)

        self.learning_rate = learning_rate
        self.exp_name = exp_name

        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_fitter.parameters(), lr=self.learning_rate)

    def generate_sample_idx(self, batch_size=50, oversample_goal=True):
        num_samples = len(self.state)
        if oversample_goal:
            zero_r_idx = np.where(self.reward == 1)
            sample_idx = list(zero_r_idx[0]) + random.sample(range(num_samples), batch_size - len(zero_r_idx[0]))
        else:
            sample_idx = random.sample(range(num_samples), batch_size)
        return sample_idx

    def optimize_model(self, gamma, batch_size, num_batches, oversample_goal, save_fname=None, no_bootstrap_within_iteration=False):
        # Compute target
        if no_bootstrap_within_iteration:
            target_q_full = self.reward + (1. - self.done) * gamma * self.q_fitter(
                self.next_state_action).detach()
        if save_fname is not None:
            loss_list = []
        for idx_batch in range(num_batches):
            # Sample a batch with the transitions reaching the goal
            if oversample_goal == 'always':
                sample_idx = self.generate_sample_idx(batch_size=batch_size, oversample_goal=True)
            elif oversample_goal == 'never':
                sample_idx = self.generate_sample_idx(batch_size=batch_size, oversample_goal=False)
            elif oversample_goal == 'first_sample_only':
                if idx_batch == 0:
                    sample_idx = self.generate_sample_idx(batch_size=batch_size, oversample_goal=True)
                else:
                    sample_idx = self.generate_sample_idx(batch_size=batch_size, oversample_goal=False)

            # Feed forward
            q_pred = self.q_fitter(self.state_action[sample_idx, :].requires_grad_())

            # Compute target
            if no_bootstrap_within_iteration:
                target_q = target_q_full[sample_idx, :]
            else:
                target_q = self.reward[sample_idx, :] + (1. - self.done[sample_idx, :]) * gamma * self.q_fitter(
                    self.next_state_action[sample_idx, :]).detach()

            # Calculate the loss
            loss = self.loss_func(q_pred, target_q)
            if save_fname is not None:
                loss_list.append(loss.item())

            # Backward propagation: caluclate gradients
            loss.backward()

            # Update the weights
            self.optimizer.step()

            # Clear gradients
            self.optimizer.zero_grad()

            if idx_batch % 10 == 0 or idx_batch == num_batches - 1:
                print('Batch {}: loss = {}'.format(idx_batch, loss.item()))

        if save_fname is not None:
            with open(save_fname, 'wb') as f:
                pickle.dump(loss_list, f)

    def update_rewards(self, termination_indicator, next_state):
        self.done = torch.from_numpy(termination_indicator(next_state).astype(np.float32)).view(-1, 1).to(self.device)
        self.reward = self.done


    def fit(self, data, termination_indicator=None, num_iter=100, gamma=0.995, batch_size=256, num_batches=1000, save_interval=np.inf, oversample_goal='never', no_bootstrap_within_iteration=False):
        self.state = torch.from_numpy(data["state"].astype(np.float32)).to(self.device)
        if termination_indicator is None:
            self.reward = torch.from_numpy(data["reward"].astype(np.float32)).view(-1, 1).to(self.device)
            self.done = torch.from_numpy(data["done"].astype(np.float32)).to(self.device)
        else:
            self.update_rewards(termination_indicator, data["next_state"])
        self.state_action = torch.from_numpy(
            np.concatenate((data["state"], data["action"]), axis=1).astype(np.float32)).to(self.device)
        next_action = chunked_policy_prediction(self.pi_eval, self.state, self.action_dim, self.device)
        self.next_state_action = torch.cat(
            (torch.from_numpy(data["next_state"].astype(np.float32)).to(self.device), next_action), dim=1).to(self.device)

        for iteration in range(num_iter):
            print('Iteration: {}'.format(iteration))
            loss_save_fname = '{}/saved_results/{}/loss_iter_{}.pkl'.format(RESULT_DIR, self.exp_name, iteration)
            self.optimize_model(gamma, batch_size, num_batches, oversample_goal, save_fname=loss_save_fname, no_bootstrap_within_iteration=no_bootstrap_within_iteration)
            if (save_interval < np.inf and iteration % save_interval == 0) or iteration == num_iter-1:
                torch.save(self.q_fitter.state_dict(), "{}/saved_results/{}/weights_{}".format(RESULT_DIR, self.exp_name, iteration))

    def get_values(self, state):
        next_action = self.pi_eval(state)
        state_policy_action = torch.cat(
            (state, next_action), dim=1)
        return self.q_fitter(state_policy_action)


if __name__ == '__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='tmp')
    parser.add_argument('--save_interval', type=int, default=np.inf)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_iter', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_batches', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--oversample_goal', type=str, default='never')
    parser.add_argument('--move_goal', action='store_true')
    parser.add_argument('--no_bootstrap_within_iteration', action='store_true')
    args = parser.parse_args()

    buffer_fname = 'td3_replay_buffer.pkl'
    data = pickle.load(open(buffer_fname, 'rb'))
    data["reward"] += 1
    data["done"] = (data["reward"] == 1).astype(float)


    exp_name = "{}_gamma_{}_lr_{}".format(args.exp_name, args.gamma, args.learning_rate)

    agent = TD3(state_dim=29,
                action_dim=8,
                max_action=1.,
                use_output_normalization=False,
                device=torch.device(args.device))
    agent_fname = 'antreacher_dense_save_rbuf_policy/0/td3_episode_500'
    load_agent(agent, agent_fname)

    state_dim = data["state"].shape[1]
    action_dim = data["action"].shape[1]

    if not os.path.exists('{}/saved_results/{}/'.format(RESULT_DIR, exp_name)):
        os.makedirs('{}/saved_results/{}/'.format(RESULT_DIR, exp_name))

    if args.move_goal:
        def termination_indicator(next_state):
            return np.sqrt((next_state[:, 0] - 2)**2 + (next_state[:, 1] - 2)**2) <= 0.5
    else:
        termination_indicator = None

    fqe = FQE(state_dim=state_dim,
              action_dim=action_dim,
              pi_eval=agent.actor,
              learning_rate=args.learning_rate,
              exp_name=exp_name,
              device=args.device)
    fqe.fit(data,
            termination_indicator=termination_indicator,
            num_iter=args.num_iter,
            gamma=args.gamma,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            save_interval=args.save_interval,
            oversample_goal=args.oversample_goal,
            no_bootstrap_within_iteration=args.no_bootstrap_within_iteration)

    with open('{}/saved_results/{}/args.txt'.format(RESULT_DIR, exp_name), 'w') as f:
        for arg in vars(args):
            f.write("{}: {}".format(arg, getattr(args, arg)))
            f.write("\n")
        t1 = time.time()
        f.write("\n")
        f.write("Runtime: {0:.2f} minutes.".format((t1-t0)/60))