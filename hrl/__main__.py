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
from hrl.utils import create_log_dir, MetaLogger
from hrl.agent.td3.TD3AgentClass import TD3
from hrl.agent.td3.utils import make_chunked_value_function_plot
from hrl.wrappers.antmaze_wrapper import D4RLAntMazeWrapper
from hrl.wrappers.MujocoGraspEnv_wrapper import D4RLGraspEnvWrapper

#sys.path.insert(1, os.path.join(sys.path[0], '../GraspInitiation/scripts/'))
#sys.path.insert(1, os.path.join(sys.path[0], '../GraspInitiation/'))
#sys.path.insert(1, os.path.join(sys.path[0], '../../GraspInitiation/scripts/'))
#sys.path.insert(1, os.path.join(sys.path[0], '../../GraspInitiation/'))
#sys.path.insert(1, os.path.join(sys.path[0], '~/Software/GraspInitiation/scripts/'))
#sys.path.insert(1, os.path.join(sys.path[0], '~/Software/GraspInitiation/'))
sys.path.insert(1, os.path.join(sys.path[0], '/users/mcorsaro/Software/GraspInitiation/scripts/'))
sys.path.insert(1, os.path.join(sys.path[0], '/users/mcorsaro/Software/GraspInitiation/'))
from MujocoGraspEnv import MujocoGraspEnv

# TODO(mcorsaro): Wrap all this in a class
def get_position(state):
    """
    position in the antmaze is the x, y coordinates
    """
    return state[:2]

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

def experience_replay(agent, mdp, trajectory, goal, dense_reward):
    for state, action, _, next_state in trajectory:
        reward_func = mdp.dense_gc_reward_func if dense_reward else mdp.sparse_gc_reward_func
        reward, done = reward_func(next_state, goal)
        agent.step(get_augmented_state(state, goal, mdp), action, reward, get_augmented_state(next_state, goal, mdp), done)

def rollout(agent, mdp, goal, steps):
    score = 0.
    mdp.reset()
    trajectory = []

    for step in range(steps):
        state = deepcopy(mdp.cur_state)
        action = agent.act(get_augmented_state(state, goal, mdp))

        next_state, reward, done, _ = mdp.step(action)

        score = score + reward
        trajectory.append((state, action, reward, next_state))

        if done:
            break

    return done, score, trajectory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--results_dir", type=str, default='results',
                        help='the name of the directory used to store results')
    parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
    parser.add_argument("--environment", type=str, choices=["door", "switch", "antmaze-umaze-v0", "antmaze-medium-play-v0", "antmaze-large-play-v0"], 
                        help="name of the gym environment")
    parser.add_argument("--seed", type=int, help="Random seed")

    parser.add_argument("--use_dense_rewards", action="store_true", default=False)
    parser.add_argument("--use_HER", action="store_true", default=False)
    parser.add_argument("--use_output_norm", action="store_true", default=False)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--logging_frequency", type=int, default=50, help="Plot after every _ episodes")

    parser.add_argument("--lr_c", type=float, default=3e-4)
    parser.add_argument("--lr_a", type=float, default=3e-4)

    parser.add_argument("--goal_state", nargs="+", type=float, default=[0, 8],
                        help="specify the goal state of the environment, (0, 8) for example")
    args = parser.parse_args()

    # TODO(mcorsaro): Add additional parameters (learning rate, HER parameters)

    saving_dir = os.path.join(args.results_dir, args.experiment_name)
    create_log_dir(saving_dir)
    meta_logger = MetaLogger(saving_dir)
    logging_filename = f"seed_{args.seed}.pkl"

    meta_logger.add_field("episodic_success_rate", logging_filename)
    meta_logger.add_field("episodic_score", logging_filename)
    meta_logger.add_field("episodic_final_dist", logging_filename)

    if "ant" in args.environment:
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
    elif args.environment == "door" or args.environment == "switch":
        env = MujocoGraspEnv(args.environment, False, reward_sparse=(not args.use_dense_rewards), gravity=True, lock_fingers_closed=True,
                         sample_method="random", state_space="friendly")
        start_state = np.array([0])
        goal_state = None
        if args.environment == "door":
            goal_state = np.array([0.5])
        elif args.environment == "switch":
            goal_state = np.array([3])
        mdp = D4RLGraspEnvWrapper(
            env,
            start_state=start_state,
            goal_state=goal_state,
            use_dense_rewards=args.use_dense_rewards)
    else:
        raise ValueError(f'Unknown environment {args.environment}')

    torch.manual_seed(0)
    seeding.seed(0, random, np)
    seeding.seed(args.seed, gym, env)

    agent = TD3(state_dim=mdp.state_space_size()+goal_state.shape[0],
                action_dim=mdp.action_space_size(),
                max_action=1.,
                device=args.device,
                lr_c=args.lr_c, lr_a=args.lr_a,
                use_output_normalization=args.use_output_norm)

    for episode in range(args.episodes):
        mdp.reset()
        goal = goal_state
        done, score, trajectory = rollout(agent, mdp, goal, args.steps if "ant" in args.environment else mdp.env._max_episode_steps)
        experience_replay(agent, mdp, trajectory, goal, args.use_dense_rewards)

        last_sars = trajectory[-1]
        final_reached_state = last_sars[-1]
        reached_goal = get_position(final_reached_state)
        if args.use_HER:
            experience_replay(agent, mdp, trajectory, reached_goal, args.use_dense_rewards)

        distance_to_goal = None
        if "ant" in args.environment:
            distance_to_goal = np.linalg.norm(reached_goal - goal, axis=-1)
        elif args.environment == "door" or args.environment == "switch":
            distance_to_goal = reached_goal[0]
        meta_logger.append_datapoint("episodic_success_rate", done, write=True)
        meta_logger.append_datapoint("episodic_score", score, write=True)
        meta_logger.append_datapoint("episodic_final_dist", distance_to_goal, write=True)

        print(f"Episode: {episode} | Score: {score}")
        '''if episode > 0 and episode % args.logging_frequency == 0:
            make_chunked_value_function_plot(agent, episode, 0, saving_dir)'''
