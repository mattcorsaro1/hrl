import os

import torch
import scipy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from treelib import Tree, Node

class SkillTree(object):
    def __init__(self, options):
        self._tree = Tree()
        self.options = options

        if len(options) > 0:
            [self.add_node(option) for option in options]

    def add_node(self, option):
        if option.name not in self._tree:
            print(f"Adding {option} to the skill-tree")
            self.options.append(option)
            parent = option.parent.name if option.parent is not None else None
            self._tree.create_node(tag=option.name, identifier=option.name, data=option, parent=parent)

    def get_option(self, option_name):
        if option_name in self._tree.nodes:
            node = self._tree.nodes[option_name]
            return node.data

    def get_depth(self, option):
        return self._tree.depth(option.name)

    def get_children(self, option):
        return self._tree.children(option.name)

    def traverse(self):
        """ Breadth first search traversal of the skill-tree. """
        return list(self._tree.expand_tree(mode=self._tree.WIDTH))

    def show(self):
        """ Visualize the graph by printing it to the terminal. """
        self._tree.show()


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def get_grid_states(mdp):
    ss = []
    x_low_lim, y_low_lim = mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = mdp.get_x_y_high_lims()
    for x in np.arange(x_low_lim, x_high_lim+1, 1):
        for y in np.arange(y_low_lim, y_high_lim+1, 1):
            ss.append(np.array((x, y)))
    return ss


def get_initiation_set_values(option):
    values = []
    x_low_lim, y_low_lim = option.overall_mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = option.overall_mdp.get_x_y_high_lims()
    for x in np.arange(x_low_lim, x_high_lim+1, 1):
        for y in np.arange(y_low_lim, y_high_lim+1, 1):
            pos = np.array((x, y))
            init = option.is_init_true(pos)
            if hasattr(option.overall_mdp.env, 'env'):
                init = init and not option.overall_mdp.env.env._is_in_collision(pos)
            values.append(init)
    return values

def plot_one_class_initiation_classifier(option):

    colors = ["blue", "yellow", "green", "red", "cyan", "brown"]

    X = option.initiation_classifier.construct_feature_matrix(option.initiation_classifier.positive_examples)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    Z1 = option.initiation_classifier.pessimistic_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)

    color = colors[option.option_idx % len(colors)]
    plt.contour(xx, yy, Z1, levels=[0], linewidths=2, colors=[color])

def plot_two_class_classifier(option, episode, experiment_name, plot_examples=True, seed=0):
    states = get_grid_states(option.overall_mdp)
    values = get_initiation_set_values(option)

    x = np.array([state[0] for state in states])
    y = np.array([state[1] for state in states])
    xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
    xx, yy = np.meshgrid(xi, yi)
    rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
    zz = rbf(xx, yy)
    plt.imshow(zz, vmin=min(values), vmax=max(values), extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", alpha=0.6, cmap=plt.cm.coolwarm)
    plt.colorbar()

    # Plot trajectories
    positive_examples = option.initiation_classifier.construct_feature_matrix(option.initiation_classifier.positive_examples)
    negative_examples = option.initiation_classifier.construct_feature_matrix(option.initiation_classifier.negative_examples)

    if positive_examples.shape[0] > 0 and plot_examples:
        plt.scatter(positive_examples[:, 0], positive_examples[:, 1], label="positive", c="black", alpha=0.3, s=10)

    if negative_examples.shape[0] > 0 and plot_examples:
        plt.scatter(negative_examples[:, 0], negative_examples[:, 1], label="negative", c="lime", alpha=1.0, s=10)

    if option.initiation_classifier.pessimistic_classifier is not None:
        plot_one_class_initiation_classifier(option)

    # background_image = imageio.imread("four_room_domain.png")
    # plt.imshow(background_image, zorder=0, alpha=0.5, extent=[-2.5, 10., -2.5, 10.])

    name = option.name if episode is None else option.name + f"_{experiment_name}_{episode}"
    plt.title(f"{option.name} Initiation Set")
    saving_path = os.path.join('~/scratch/results', experiment_name, 'initiation_set_plots', f'{name}_initiation_classifier_{seed}.png')
    plt.savefig(saving_path)
    plt.close()


def plot_importance_weights(episode, thresh, option, states, labels, weights):
        x_positions = states[:, 0].detach().cpu().numpy()
        y_positions = states[:, 1].detach().cpu().numpy()

        x_positions_positive = x_positions[labels == 1]
        y_positions_positive = y_positions[labels == 1]
        weights_positive = weights[labels == 1]

        x_positions_negative = x_positions[labels == 0]
        y_positions_negative = y_positions[labels == 0]
        weights_negative = weights[labels == 0]

        plt.figure(figsize=(16, 10))

        plt.subplot(1, 2, 1)
        plt.scatter(x_positions_positive, y_positions_positive, c=weights_positive, s=5, marker="P")
        plt.scatter(x_positions_negative, y_positions_negative, c=weights_negative, s=5, marker="X")
        plt.colorbar()
        plt.title(f"Importance Re-weighting Option: {option} Episode: {episode} Threshold: {thresh}")

        plt.suptitle("Sample Weights")
        plt.savefig(f"plots/initiation_weights/option_{option}_episode_{episode}.png")
        plt.close()


def plot_initiation_distribution(option, mdp, episode, experiment_name, chunk_size=10000):
    assert option.initiation_distribution is not None
    data = mdp.dataset[:, :2]

    num_chunks = int(np.ceil(data.shape[0] / chunk_size))
    if num_chunks == 0:
        return 0.

    state_chunks = np.array_split(data, num_chunks, axis=0)
    pvalues = np.zeros((data.shape[0],))
    current_idx = 0

    for chunk_number, state_chunk in tqdm(enumerate(state_chunks)):
        probabilities = np.exp(option.initiation_distribution.score_samples(state_chunk))
        pvalues[current_idx:current_idx + len(state_chunk)] = probabilities
        current_idx += len(state_chunk)

    plt.scatter(data[:, 0], data[:, 1], c=pvalues)
    plt.colorbar()
    plt.title("Density Estimator Fitted on Pessimistic Classifier")
    saving_path = os.path.join('~/scratch/results', experiment_name, 'initiation_set_plots', f'{option.name}_initiation_distribution_{episode}.png')
    plt.savefig(saving_path)
    plt.close()


def make_chunked_goal_conditioned_value_function_plot(solver, goal, episode, seed, experiment_name, chunk_size=1000, replay_buffer=None, option_idx=None):
    
    if ('RBF' in str(type(solver))):
        replay_buffer = solver.actor.buffer_object
        trans = replay_buffer.storage.get_all_transitions()
        states = trans['obs']
        states = [state[:-2] for state in states]
        actions = trans['act']
    else:
        replay_buffer = replay_buffer if replay_buffer is not None else solver.replay_buffer
        # Take out the original goal and append the new goal
        states = [exp[0] for exp in replay_buffer]
        states = [state[:-2] for state in states]
        actions = [exp[1] for exp in replay_buffer]

    goal = goal[:2]  # Extracting the position from the goal vector

    if len(states) > 100_000:
        print(f"Subsampling {len(states)} s-a pairs to 100,000")
        idx = np.random.randint(0, len(states), size=100_000)
        states = [states[i] for i in idx]
        actions = [actions[i] for i in idx]

    print(f"preparing {len(states)} states")
    states = np.array([np.concatenate((state, goal), axis=0) for state in states])

    print(f"preparing {len(actions)} actions")
    actions = np.array(actions)

    # Chunk up the inputs so as to conserve GPU memory
    num_chunks = int(np.ceil(states.shape[0] / chunk_size))

    if num_chunks == 0:
        return 0.

    print("chunking")
    state_chunks = np.array_split(states, num_chunks, axis=0)
    action_chunks = np.array_split(actions, num_chunks, axis=0)
    qvalues = np.zeros((states.shape[0],))
    current_idx = 0

    for chunk_number, (state_chunk, action_chunk) in tqdm(enumerate(zip(state_chunks, action_chunks)), desc="Making VF plot"):  # type: (int, np.ndarray)
        state_chunk = torch.from_numpy(state_chunk).float().to(solver.device)
        action_chunk = torch.from_numpy(action_chunk).float().to(solver.device)
        chunk_qvalues = solver.get_qvalues(state_chunk, action_chunk).cpu().numpy()
        chunk_qvalues = chunk_qvalues.sum(axis=1)
        if (len(chunk_qvalues.shape) > 1 and chunk_qvalues.shape[1] == 1):
            chunk_qvalues = chunk_qvalues.squeeze(1)
        current_chunk_size = len(state_chunk)
        qvalues[current_idx:current_idx + current_chunk_size] = chunk_qvalues
        current_idx += current_chunk_size

    print("plotting")
    plt.scatter(states[:, 0], states[:, 1], c=qvalues)
    plt.colorbar()

    if option_idx is None:
        file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}"
    else:
        file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}_option_{option_idx}"
    plt.title(f"VF Targeting {np.round(goal, 2)}")
    saving_path = os.path.join('~/scratch/results', experiment_name, 'value_function_plots', f'{file_name}.png')

    print("saving")
    plt.savefig(saving_path)
    plt.close()

    return qvalues.max()


def plot_value_distribution(solver, state, goal, episode, option_name, seed, experiment_name):
    def value2steps(value):
        """ Assuming -1 step reward, convert a value prediction to a n_step prediction. """
        def _clip(v):
            if isinstance(v, np.ndarray):
                v[v>0] = 0
                return v
            return v if v <= 0 else 0

        gamma = solver.gamma
        clipped_value = _clip(value)
        numerator = np.log(1 + ((1-gamma) * np.abs(clipped_value)))
        denominator = np.log(gamma)
        return np.abs(numerator / denominator)
    
    augmented_state = np.concatenate((state, goal))
    states = augmented_state[np.newaxis, :]
    states = torch.as_tensor(states).float().to(solver.device)
    values = solver.get_value_distribution(states)
    values = values.detach().cpu().numpy().squeeze()
    steps = value2steps(values)

    plt.hist(steps)
    plt.xlabel("Expected Number of Steps to Goal")
    plt.title(f"Episode {episode} Goal {np.round(goal, 2)}")
    fname = f"{option_name}_value_dist{seed}_episode_{episode}.png"
    saving_path = os.path.join('/users/mcorsaro/scratch/results', experiment_name, 'value_function_plots', f'{fname}')
    print("Saving in", saving_path)
    plt.savefig(saving_path)
    plt.close()


