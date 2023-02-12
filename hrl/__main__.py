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
from hrl.wrappers.mlp_classifier import BinaryMLPClassifier

def compute_weights_unbatched(states, labels, values, threshold):
    n_states = states.shape[0]
    weights = numpy.zeros((n_states,))
    for i in range(n_states):
        label = labels[i]
        state_value = values[i]
        if label == 1:  # These signs are assuming that we are thresholding *steps*, not values.
            flip_mass = state_value[state_value < threshold].sum()
        else:
            flip_mass = state_value[state_value > threshold].sum()
        weights[i] = flip_mass / state_value.sum()
        #print("^^^^^^flip_mass, state_value", flip_mass, state_value.sum(), weights[i])
    return weights

def get_weights(states, labels, learner):
    """
    Given state, threshold, value function, compute the flipping prob for each state
    Return 1/flipping prob which is the weights
        The formula for weights is a little more complicated, see paper Akhil will send in
        channel
    Args:
      states (torch tensor): num states, state_dim
      labels (list[int]): num states
    """
    # Compute updated weights
    """ Get the value distribution for the input states. """
    # shape: (num grasps, 200)
    value_distribution = learner.get_values(states).detach().cpu().numpy()
    print(";;;;;;;;;;VD SHAPE", value_distribution.shape)
    #print("^^^^^^")
    #print("^^^^^^VD", value_distribution)

    '''# We have to mmake sure that the distribution and threshold are in the same units
    step_distribution = value2steps(value_distribution)
    print("^^^^^^SD", step_distribution)'''

    # Determine the threshold. It has units of # steps.
    threshold = numpy.median(value_distribution)  # TODO: This should be a percentile based on class ratios
    #print(f"Set the threshold to {threshold}")
    #print("^^^^^^Threshold", threshold)

    probabilities = compute_weights_unbatched(states, labels, value_distribution, threshold)
    weights = 1. / (probabilities + 1e-1)
    #print("^^^^^^Computed weights", weights)
    #print("^^^^^^")
    return weights

# TODO(mcorsaro): Wrap all this in a class
def get_antmaze_position(state):
    """
    position in the antmaze is the x, y coordinates
    """
    return state[:2]

def extract_goal_dimensions(mdp, goal):
    def _extract(goal):
        goal_features = goal
        if isinstance(mdp.unwrapped, MujocoGraspEnv):
            return goal
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
    final_info = None

    for step in range(steps):
        state = deepcopy(mdp.cur_state)
        action = agent.act(get_augmented_state(state, goal, mdp))

        next_state, reward, done, final_info = mdp.step(action)

        score = score + reward
        trajectory.append((state, action, reward, next_state))

        if done:
            break

    return done, score, trajectory, final_info

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

    parser.add_argument("--sample_method",
                        required=False,
                        choices=["random", "oracle", "classifier", "classifier_unweighted"],
                        type=str,
                        default="random")

    parser.add_argument("--lr_c", type=float, default=3e-4)
    parser.add_argument("--lr_a", type=float, default=3e-4)

    parser.add_argument("--goal_state", nargs="+", type=float, default=[0, 8],
                        help="specify the goal state of the environment, (0, 8) for example")
    args = parser.parse_args()

    # TODO(mcorsaro): Add additional parameters (learning rate, HER parameters)

    saving_dir = os.path.join(args.results_dir, args.experiment_name)
    create_log_dir(saving_dir)
    meta_logger = MetaLogger(saving_dir)
    classifier_prob_dir = None
    if "classifier" in args.sample_method:
        classifier_prob_dir = utils.create_log_dir(
            os.path.join(saving_dir, "classifier_probs"))
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
        start_state = np.array([0])
        goal_state = np.array([0.5])if args.environment == "door" else np.array([3])
        env = MujocoGraspEnv(args.environment, False, reward_sparse=(not args.use_dense_rewards), gravity=True, lock_fingers_closed=True,
                         sample_method=args.sample_method, state_space="friendly", goal_state=goal_state)
        mdp = D4RLGraspEnvWrapper(
            env,
            start_state=start_state,
            goal_state=goal_state)
    else:
        raise ValueError(f'Unknown environment {args.environment}')

    torch.manual_seed(0)
    seeding.seed(args.seed, random, np)
    seeding.seed(args.seed, gym, env)

    agent = TD3(state_dim=mdp.state_space_size()+goal_state.shape[0],
                action_dim=mdp.action_space_size(),
                max_action=1.,
                device=args.device,
                lr_c=args.lr_c, lr_a=args.lr_a,
                use_output_normalization=args.use_output_norm)

    clf = None
    if "classifier" in args.sample_method:
        clf = BinaryMLPClassifier(\
            env.cache_torch_state.shape[1], \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'), \
            threshold=0.5, \
            batch_size=5)

    # Dictionary of labels, indexed by grasp index
    classifier_training_dict = {}

    for episode in range(args.episodes):
        mdp.reset()
        goal = goal_state
        done, score, trajectory, final_info = rollout(agent, mdp, goal, args.steps if "ant" in args.environment else mdp.env._max_episode_steps)

        experience_replay(agent, mdp, trajectory, goal, args.use_dense_rewards)

        if "classifier" in args.sample_method:
            grasp_index = int(final_info["grasp_index"])
            success_label = float(final_info["success"])
            # Dictionary of latest grasp success label for each grasp index
            classifier_training_dict[grasp_index] = success_label

            grasp_indices = classifier_training_dict.keys()
            # List of ints
            classifier_training_labels = numpy.array([classifier_training_dict[grasp_index] for grasp_index in grasp_indices])
            if clf.should_train(classifier_training_labels):
                # List of tensors of lists
                grasp_indices_tensor = torch.LongTensor(list(grasp_indices))
                classifier_training_examples = env.cache_torch_state.index_select(0, grasp_indices_tensor)

                if args.sample_method == "classifier_unweighted":
                    clf.fit(classifier_training_examples.to(clf.device).float(), classifier_training_labels, n_epochs=10)
                else:
                    W = get_weights(classifier_training_examples.to(agent.device), classifier_training_labels, agent).astype(float)
                    print(";;;;;;;Received weights of shape", W.shape)
                    clf.fit(classifier_training_examples.to(clf.device).float(), classifier_training_labels, W, n_epochs=10)

                # Set weights for agent to draw new examples
                env.classifier_probs = clf.predict_proba(env.cache_torch_state.to(clf.device).float()).detach().cpu().numpy()
                env.classifier_probs = env.classifier_probs.reshape((-1))
                if (numpy.isnan(env.classifier_probs).any()):
                    print("~~~~~Array contains nans, setting nan probs to 0")
                    env.classifier_probs[numpy.isnan(env.classifier_probs)] = 0
                env.classifier_probs = softmax(env.classifier_probs)
                if episodes % 1000 == 0:
                    prob_output_file = classifier_prob_dir + "/cached_grasp_{}_prob_{}_seed_{}_episode_{}.npy".format(args.sample_method, args.environment, args.seed, episode)
                    print("Now writing probabilities to", prob_output_file)
                    numpy.save(prob_output_file, env.classifier_probs)

        last_sars = trajectory[-1]
        final_reached_state = last_sars[-1]
        reached_goal = get_antmaze_position(final_reached_state) if "ant" in args.environment else (mdp.get_door_position(final_reached_state) if args.environment == "door" else mdp.get_switch_position(final_reached_state))
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
