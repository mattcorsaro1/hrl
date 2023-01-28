from copy import deepcopy

import numpy as np
import torch

from hrl.wrappers.gc_mdp_wrapper import GoalConditionedMDPWrapper


class D4RLGraspEnvWrapper(GoalConditionedMDPWrapper):
    def __init__(self, env, start_state, goal_state, init_truncate=True):
        self.env = env
        self.init_truncate = init_truncate
        self.norm_func = lambda x: np.linalg.norm(x, axis=-1) if isinstance(x, np.ndarray) else torch.norm(x, dim=-1)
        self.reward_func = self.sparse_gc_reward_func
        #self.observations = self.env.get_dataset()["observations"]
        super().__init__(env, start_state, goal_state)

    def state_space_size(self):
        return self.env.observation_space.shape[0]
    
    def action_space_size(self):
        return self.env.action_space.shape[0]
    
    def sparse_gc_reward_func(self, states, goals, batched=False):
        """
        overwritting sparse gc reward function for antmaze
        """
        # assert input is np array or torch tensor
        assert isinstance(states, (np.ndarray, torch.Tensor))
        assert isinstance(goals, (np.ndarray, torch.Tensor))

        print(";;;;;;;states", states)
        print(";;;;;;;goals", goals)
        sys.exit()

        if batched:
            current_positions = states[:,:2]
            goal_positions = goals[:,:2]
        else:
            current_positions = states[:2]
            goal_positions = goals[:2]
        distances = self.norm_func(current_positions-goal_positions)
        dones = distances <= self.goal_tolerance

        rewards = np.zeros_like(distances)
        rewards[dones==1] = 1.
        rewards[dones==0] = 0.

        return rewards, dones

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        reward, done = self.reward_func(next_state, self.get_current_goal())
        self.cur_state = next_state
        self.cur_done = done
        return next_state, reward, done, info

    def get_current_goal(self):
        return self.get_position(self.goal_state)

    def is_start_region(self, states):
        dist_to_start = self.norm_func(states - self.start_state)
        return dist_to_start <= self.goal_tolerance
    
    def is_goal_region(self, states):
        dist_to_goal = self.norm_func(states - self.goal_state)
        return dist_to_goal <= self.goal_tolerance
    
    def extract_features_for_initiation_classifier(self, states):
        """
        for antmaze, the features are the x, y coordinates (first 2 dimensions)
        """
        def _numpy_extract(states):
            if len(states.shape) == 1:
                return states[:2]
            assert len(states.shape) == 2, states.shape
            return states[:, :2]
        
        def _list_extract(states):
            return [state[:2] for state in states]
        
        if self.init_truncate:
            if isinstance(states, np.ndarray):
                return _numpy_extract(states)
            if isinstance(states, list):
                return _list_extract(states)
            raise ValueError(f"{states} of type {type(states)}")
        
        return states
    
    @staticmethod
    def get_position(state):
        """
        position in the antmaze is the x, y coordinates
        """
        return state[:2]
