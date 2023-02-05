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

    def get_door_position(self, state):
        door_state = state[-8:-6]
        # Door hinge is qpos index 12, latch is 13
        return door_state[0:1]

    def get_switch_position(self, state):
        return state[-7:-6]

    def state_space_size(self):
        return self.env.observation_space.shape[0]
    
    def action_space_size(self):
        return self.env.action_space.shape[0]
    
    def dense_gc_reward_func(self, states, goals, batched=False):
        raise NotImplementedError(f"Robot environment dense reward functions not implemented.")

    def sparse_gc_reward_func(self, states, goals, batched=False):
        """
        overwritting sparse gc reward function for antmaze
        """
        # assert input is np array or torch tensor
        assert isinstance(states, (np.ndarray, torch.Tensor))
        assert isinstance(goals, (np.ndarray, torch.Tensor))

        if batched:
            raise NotImplementedError(f'Implement batched sparse reward function.')
        else:
            current_positions = self.get_door_position(states) if self.env.obj == "door" else self.get_switch_position(states)
            goal_positions = goals
        dones = current_positions > goal_positions

        rewards = np.zeros_like(dones)
        rewards[dones==1] = 1.
        rewards[dones==0] = 0.

        breakpoint()

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
