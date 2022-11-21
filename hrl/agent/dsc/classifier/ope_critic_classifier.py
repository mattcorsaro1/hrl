import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from hrl.agent.fqe.fqe import FQE
from hrl.agent.td3.TD3AgentClass import TD3
from ..datastructures import ValueThresholder
from .critic_classifier import CriticInitiationClassifier

RESULT_DIR = "/users/mcorsaro/scratch/"

class OPECriticInitiationClassifier(CriticInitiationClassifier):
    """ Initiation Classifier that thresholds V^pi. """

    def __init__(
        self,
        state_dim,
        action_dim,
        actor_critic,
        goal_sampler,
        augment_func,
        termination_clf,
        device,
        option_name,
        optimistic_threshold=0.6,
        pessimistic_threshold=0.8,
    ):
        assert isinstance(actor_critic, TD3), actor_critic
        
        fqe = FQE(
            state_dim=state_dim,
            action_dim=action_dim,
            pi_eval=actor_critic.actor,
            device="cuda",
            # option_name=option_name
        )

        super().__init__(fqe, goal_sampler, augment_func, optimistic_threshold, pessimistic_threshold)

        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.option_name = option_name
        self.actor_critic = actor_critic
        self.termination_clf = termination_clf
        self.optimistic_classifier = ValueThresholder(optimistic_threshold)
        self.pessimistic_classifier = ValueThresholder(pessimistic_threshold)
        
        self.is_fitted = False

    def is_initialized(self):
        return self.is_fitted

    def optimistic_predict(self, state):
        value = self.value_function(state)
        return self.optimistic_classifier(value)
    
    def pessimistic_predict(self, state):
        value = self.value_function(state)
        return self.pessimistic_classifier(value)

    def fit_initiation_classifier(self, n_iterations=5):
        n_iterations = 100 if not self.is_fitted else n_iterations
        self.agent.fit(
            self.actor_critic.replay_buffer.serialize(),
            self.termination_clf,
            n_iterations
        )
        self.is_fitted = True

    def plot_initiation_classifier(self, env, replay_buffer, option_name, episode, experiment_name, seed):
        print(f"Plotting OPE-Critic Initiation Set Classifier for {option_name}")

        chunk_size = 1000

        # Take out the original goal
        states = [exp[0] for exp in replay_buffer]
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
        values = np.zeros((states.shape[0],))
        current_idx = 0

        for state_chunk in tqdm(state_chunks, desc="Plotting Critic Init Classifier"):
            chunk_values = self.value_function(state_chunk)
            current_chunk_size = len(state_chunk)
            values[current_idx:current_idx + current_chunk_size] = chunk_values.squeeze()
            current_idx += current_chunk_size

        print("plotting")
        plt.scatter(states[:, 0], states[:, 1], c=values)
        plt.colorbar()

        file_name = f"{option_name}_ope_critic_init_clf_{seed}_episode_{episode}"
        plt.title(f"OPE VF for {option_name}")
        saving_path = os.path.join(RESULT_DIR, 'results', experiment_name, 'initiation_set_plots', f'{file_name}.png')

        print("saving")
        plt.savefig(saving_path)
        plt.close()
