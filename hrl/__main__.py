import argparse
import numpy as np
from copy import deepcopy
import os
import random
import sys

import seeding
import gym
import d4rl
import torch
from hrl.utils import create_log_dir
from hrl.agent.td3.TD3AgentClass import TD3
from hrl.agent.td3.utils import make_chunked_value_function_plot
from hrl.wrappers.antmaze_wrapper import D4RLAntMazeWrapper

def extract_goal_dimensions(mdp, goal):
    def _extract(goal):
        goal_features = goal
        if "ant" in mdp.unwrapped.spec.id:
            return goal_features[:2]
        raise NotImplementedError(f"{mdp.env_name}")
    if isinstance(goal, np.ndarray):
        return _extract(goal)
    return goal.pos

def get_augmented_state(state, goal, mdp):
    assert goal is not None and isinstance(goal, np.ndarray), f"goal is {goal}"

    goal_position = extract_goal_dimensions(mdp, goal)
    return np.concatenate((state, goal_position))

def experience_replay(agent, mdp, trajectory, goal):
    for state, action, _, next_state in trajectory:
        reward, done = mdp.sparse_gc_reward_func(next_state, goal)
        agent.step(get_augmented_state(state, goal, mdp), action, reward, get_augmented_state(next_state, goal, mdp), done)

def rollout(agent, mdp, goal, steps):
    score = 0.
    mdp.reset()
    trajectory = []

    for step in range(steps):
        state = deepcopy(mdp.cur_state)
        action = agent.act(state)

        next_state, reward, done, _ = mdp.step(action)

        score = score + reward
        trajectory.append((state, action, reward, next_state))

        if done:
            break

    return score, trajectory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--results_dir", type=str, default='results',
                        help='the name of the directory used to store results')
    parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
    parser.add_argument("--environment", type=str, choices=["antmaze-umaze-v0", "antmaze-medium-play-v0", "antmaze-large-play-v0"], 
                        help="name of the gym environment")
    parser.add_argument("--seed", type=int, help="Random seed")

    parser.add_argument("--use_dense_rewards", action="store_true", default=False)
    parser.add_argument("--buffer_length", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--logging_frequency", type=int, default=50, help="Plot after every _ episodes")

    parser.add_argument("--goal_state", nargs="+", type=float, default=[0, 8],
                        help="specify the goal state of the environment, (0, 8) for example")
    args = parser.parse_args()

    # TODO(mcorsaro): Add additional parameters (learning rate, use HER, HER parameters, num episodes)
    # TODO(mcorsaro): Implement HER correctly
    # TODO(mcorsaro): Use all parameters (logging_frequency)

    saving_dir = os.path.join(args.results_dir, args.experiment_name)
    create_log_dir(saving_dir)

    env = gym.make(args.environment)
    # pick a goal state for the env
    goal_state = np.array(args.goal_state)
    mdp = D4RLAntMazeWrapper(
        env,
        start_state=np.array((0, 0)),
        goal_state=goal_state,
        #init_truncate="position" in args.init_classifier_type,
        use_dense_reward=args.use_dense_rewards
    )

    torch.manual_seed(0)
    seeding.seed(0, random, np)
    seeding.seed(args.seed, gym, env)

    agent = TD3(state_dim=mdp.state_space_size(),
                action_dim=mdp.action_space_size(),
                max_action=1.,
                use_output_normalization=False)

    per_episode_scores = []

    for episode in range(args.episodes):
        mdp.reset()
        goal = goal_state
        score, trajectory = rollout(agent, mdp, goal, args.steps)
        experience_replay(agent, mdp, trajectory, goal)

        per_episode_scores.append(score)
        print(f"Episode: {episode} | Score: {score}")
        if episode > 0 and episode % 100 == 0:
            make_chunked_value_function_plot(agent, episode, 0, saving_dir)
