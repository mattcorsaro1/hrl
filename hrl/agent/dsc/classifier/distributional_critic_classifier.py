import os
import ipdb
import scipy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools
from tqdm import tqdm
from hrl.utils import flatten
from sklearn.svm import OneClassSVM, SVC
from .flipping_classifier import FlippingClassifier
from .critic_classifier import CriticInitiationClassifier
from .position_classifier import PositionInitiationClassifier
import warnings
warnings.filterwarnings("ignore")

class DistributionalCriticClassifier(PositionInitiationClassifier):

    def __init__(
        self,
        agent,
        use_position,
        goal_sampler,
        augment_func,
        optimistic_threshold,
        pessimistic_threshold,
        option_name,
        maxlen=100,
        resample_goals=False,
        threshold=None,
        device=None
    ):
        self.agent = agent
        self.use_position = use_position
        self.device = device
        self.goal_sampler = goal_sampler

        self.critic_classifier = CriticInitiationClassifier(
            agent,
            goal_sampler,
            augment_func,
            optimistic_threshold,
            pessimistic_threshold
        )

        self.option_name = option_name
        self.resample_goals = resample_goals
        self.threshold = threshold

        self.optimistic_predict_thresh = 0.4
        self.pessimistic_predict_thresh = 0.6

        super().__init__(maxlen)
        
    @torch.no_grad()
    def get_weights(self, states, labels): 
        '''
        Given state, threshold, value function, compute the flipping prob for each state
        Return 1/flipping prob which is the weights
            The formula for weights is a little more complicated, see paper Akhil will send in 
            channel
        Args:
          states (np.ndarray): num states, state_dim
          labels (np.ndarray): num states, 

        '''
        states = states.to(self.device).to(torch.float32)
        best_actions = self.critic_classifier.agent.actor.get_best_qvalue_and_action(states.to(torch.float32))[1]
        best_actions = best_actions.to(self.device).to(torch.float32)
        value_distribution = self.critic_classifier.agent.actor.forward(states, best_actions).cpu().numpy()

        # We have to mmake sure that the distribution and threshold are in the same units
        step_distribution = self.value2steps(value_distribution)
        
        # Determine the threshold. It has units of # steps.
        self.threshold = np.median(step_distribution)  # TODO: This should be a percentile based on class ratios
        print(f"Set the threshold to {self.threshold}")

        probabilities = self._compute_weights_unbatched(states, labels, step_distribution)
        weights = 1. / (probabilities + 1e-1)
        return weights

    def _compute_weights_unbatched(self, states, labels, values):
        n_states = states.shape[0]
        weights = np.zeros((n_states,))
        for i in range(n_states):
            label = labels[i]
            state_value = values[i]
            if label == 1:  # These signs are assuming that we are thresholding *steps*, not values.
                flip_mass = state_value[state_value > self.threshold].sum()
            else:
                flip_mass = state_value[state_value < self.threshold].sum()
            weights[i] = flip_mass / state_value.sum()
        return weights

    def _compute_weights_batched(self, states, labels, values):  # TODO
        pass

    def add_positive_examples(self, states, infos):
        assert all(["value" in info for info in infos]), "need V(sg) for weights"
        assert all(["augmented_state" in info for info in infos]), "need sg to recompute V(sg)"
        return super().add_positive_examples(states, infos)
    
    def add_negative_examples(self, states, infos):
        assert all(["value" in info for info in infos]), "need V(s) for weights"
        assert all(["augmented_state" in info for info in infos]), "need sg to recompute V(sg)"
        return super().add_negative_examples(states, infos)

    @staticmethod
    def construct_feature_matrix(examples):
        examples = list(itertools.chain.from_iterable(examples))
        positions = [example.pos for example in examples]
        return np.array(positions)

    def get_sample_weights(self, plot=False):

        pos_egs = flatten(self.positive_examples)
        neg_egs = flatten(self.negative_examples)
        examples = pos_egs + neg_egs

        assigned_labels = np.concatenate((
            +1 * np.ones((len(pos_egs),)),
            -1 * np.ones((len(neg_egs),))
        ))

        # Extract what labels the current VF would have assigned
        augmented_states = np.array([eg.info["augmented_state"] for eg in examples])

        if self.resample_goals:
            observations = augmented_states[:, :-2]
            new_goal = self.critic_classifier.goal_sampler()[np.newaxis, ...]
            new_goals = np.repeat(new_goal, axis=0, repeats=observations.shape[0])
            augmented_states = np.concatenate((observations, new_goals), axis=1)

        # Compute the weights based on the probability that the samples will flip
        weights = self.get_weights(torch.from_numpy(augmented_states), assigned_labels)

        if plot:
            # ipdb.set_trace()
            x = [eg.info["player_x"] for eg in examples]
            y = [eg.info["player_y"] for eg in examples]
            c = assigned_labels.tolist()
            s = (1. * weights).tolist()
            plt.subplot(1, 2, 1)
            plt.scatter(x[:len(pos_egs)], y[:len(pos_egs)], c=s[:len(pos_egs)])
            plt.colorbar()
            plt.clim((0, 10))
            plt.subplot(1, 2, 2)
            plt.scatter(x[len(pos_egs):], y[len(pos_egs):], c=s[len(pos_egs):])
            plt.colorbar()
            plt.clim((0, 10))
            plt.savefig(f"results/weight_plots_{self.option_name}.png")
            plt.close()

        return weights

    @staticmethod
    def value2steps(value):
        """ Assuming -1 step reward, convert a value prediction to a n_step prediction. """
        def _clip(v):
            if isinstance(v, np.ndarray):
                v[v>0] = 0
                return v
            return v if v <= 0 else 0

        gamma = .99
        clipped_value = _clip(value)
        numerator = np.log(1 + ((1-gamma) * np.abs(clipped_value)))
        denominator = np.log(gamma)
        return np.abs(numerator / denominator)

    def plot_initiation_classifier(self, goal, option_name, episode, experiment_name, seed):
        print(f"Plotting Critic Initiation Set Classifier for {option_name}")

        chunk_size = 1000
        replay_buffer = self.agent.actor.buffer_object.storage

        # Take out the original goal
        trans = replay_buffer.get_all_transitions()
        states = trans['obs']
        states = [state[:-2] for state in states]

        if len(states) > 100_000:
            print(f"Subsampling {len(states)} s-a pairs to 100,000")
            idx = np.random.randint(0, len(states), size=100_000)
            states = [states[i] for i in idx]

        print(f"preparing {len(states)} states")
        states = np.array(states)

        # Chunk up the inputs so as to conserve GPU memory
        num_chunks = int(np.ceil(states.shape[0] / chunk_size))

        if num_chunks == 0:
            return 0.

        print("chunking")
        state_chunks = np.array_split(states, num_chunks, axis=0)
        steps = np.zeros((states.shape[0],))
        
        optimistic_predictions = np.zeros((states.shape[0],))
        pessimistic_predictions = np.zeros((states.shape[0],))

        current_idx = 0

        for state_chunk in tqdm(state_chunks, desc="Plotting Critic Init Classifier"):
            goal = np.repeat([goal], repeats=len(state_chunk), axis=0)
            state_chunk = state_chunk[:, :2]
            current_chunk_size = len(state_chunk)

            optimistic_predictions[current_idx:current_idx + current_chunk_size] = self.optimistic_classifier.predict(state_chunk)
            pessimistic_predictions[current_idx:current_idx + current_chunk_size] = self.pessimistic_classifier.predict(state_chunk)

            current_idx += current_chunk_size

        print("plotting")
        plt.figure(figsize=(20, 10))
        
        plt.subplot(1, 3, 1)
        plt.scatter(states[:, 0], states[:, 1], c=steps)
        plt.title(f"nSteps to termination region")
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.scatter(states[:, 0], states[:, 1], c=optimistic_predictions)
        plt.title(f"Optimistic Classifier")
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.scatter(states[:, 0], states[:, 1], c=pessimistic_predictions)
        plt.title(f"Pessimistic Classifier")
        plt.colorbar()

        plt.suptitle(f"{option_name}")
        file_name = f"{option_name}_critic_init_clf_{seed}_episode_{episode}"
        saving_path = os.path.join('results', experiment_name, 'initiation_set_plots', f'{file_name}.png')

        print("saving")
        plt.savefig(saving_path)
        plt.close()
