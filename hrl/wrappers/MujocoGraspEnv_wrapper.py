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

        rewards = np.zeros(dones.shape)
        rewards[dones==1] = 1.
        rewards[dones==0] = 0.

        return rewards, dones

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        reward, done = self.reward_func(next_state, self.goal_state)
        self.cur_state = next_state
        self.cur_done = done
        return next_state, reward, done, info
