'''
Copyright, 2022, Matt Corsaro, matthew_corsaro@brown.edu
'''

# Originally derived from https://github.com/babbatem/motor_skills/blob/impedance/old/motor_skills/planner/mj_plan_test.py

import argparse
import copy
import numpy as np
import torch
import os
import hjson
import open3d as o3d
import random
import sys
import gym
import mj_control as mjc
from utils import mj_point_clouds as mjpc
from utils import file_io
import grasp_pose_generator as gpg

from mujoco_py import load_model_from_path, MjSim, MjViewer, load_model_from_mjb
from mujoco_py.builder import MujocoException
from gym.envs.mujoco import mujoco_env
from gym import spaces
from mujoco_py import GlfwContext
# from mujoco_py.builder import cymj
# from cymj import glfw
# import OpenGL.GLUT as gl

from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control.mujoco.engine import Physics

from arm_controller import PositionOrientationController
from scipy.special import softmax

# https://github.com/clemense/quaternion-conventions
# mujoco to pybullet
def wxyz2xyzw(wxyz):
    return wxyz[1:] + [wxyz[0]]
# pybullet to mujoco
def xyzw2wxyz(xyzw):
    return [xyzw[-1]] + list(xyzw[:-1])

# Class wrapping MuJoCo environment, point cloud, grasp pose estimation,
#     and grasp controller.
class MujocoGraspEnv(mujoco_env.MujocoEnv):
    def __init__(self, obj, visualize=True, reward_sparse=True, gravity=True, lock_fingers_closed=True, sample_method="random", state_space="friendly"):
        self.random_grasp_idx_chosen = 0
        self.sample_method = sample_method

        # When using classifier, initiatlly set weights to None so samples drawn at random
        self.classifier_probs = None

        # Load scene
        self.obj = obj
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        model_filepath = self.script_path + \
            '/../assets/kinova_j2s6s300/mj-j2s6s300_{}.xml'.format(self.obj)
        model = load_model_from_path(model_filepath)

        self.curr_step = 0
        self._max_episode_steps = 50
        self.reward_sparse = reward_sparse

        # the sampled grasp pose we chose x, y, z 
        # used in the reward function for distance in switch/door
        self.sampled_pose_chosen = None

        self.sim = MjSim(model)
        self.visualize = visualize
        if visualize:
            self.viewer = MjViewer(self.sim)
            GlfwContext(offscreen=True)
        else:
            self.viewer = None

        # Arm DoF
        self.aDOF=6
        # Gripper DoF
        self.gDOF = 6
        # Total robot dof
        self.tDOF = self.aDOF + self.gDOF
        self.num_links_per_finger = 2

        self.open_finger_state = None
        self.current_finger_state = None

        # current joint pos of the latch in door and switch 
        self.curr_joint_pos = None

        # Grasping parameters
        self.max_finger_delta = 1.3
        self.grasp_steps = 500
        self.grasp_steps_after_contact = 100
        self.finger_base_desired_torque = 2.5
        self.finger_tip_desired_torque = 2.5
        self.minimum_sensor_output_for_contact = 2.5

        self.filter_low_quality_grasps = True
        self.minimum_grasp_quality = 0.75

        if self.obj == "switch":
            # For the switch, bring the gripper closer to the switch body
            self.dist_from_point_to_ee_link = -0.033
        else:
            self.dist_from_point_to_ee_link = -0.02

        # Finger IDs
        self.finger_joint_idxs = []
        self.finger_base_idxs = []
        self.finger_tip_idxs = []
        for i in range(1,int(self.gDOF/self.num_links_per_finger) + 1):
            base_idx = self.sim.model.joint_name2id(\
                "j2s6s300_joint_finger_" + str(i))
            tip_idx = self.sim.model.joint_name2id(\
                "j2s6s300_joint_finger_tip_" + str(i))
            self.finger_joint_idxs.append(base_idx)
            self.finger_joint_idxs.append(tip_idx)
            self.finger_base_idxs.append(base_idx)
            self.finger_tip_idxs.append(tip_idx)

        # Boundaries used to crop point clouds and extract graspable regions
        point_cloud_cropping_boundaries = {}
        point_cloud_cropping_boundaries["door"] = \
            [(0.189, -2., 0.05), (2., 0.4, 2.)]
        # Not just the door handle, currently unused
        point_cloud_cropping_boundaries["full_door"] = \
            [(-2., -2., 0.05), (2., 2., 2.)]
        point_cloud_cropping_boundaries["pitcher"] = \
            [(-0.1, 0.1, 0.01), (1., 1., 1.)]
        point_cloud_cropping_boundaries["mug"] = \
            [(-1, 0.1, 0.01), (1., 1., 0.2)]
        point_cloud_cropping_boundaries["switch"] = \
            [(-1, 0.1, 0.01), (1., 1., 1.)]
        # currently unused
        point_cloud_cropping_boundaries["cylinder"] = \
            [(-1, 0.1, 0.01), (1., 1., 1.)]
        # currently unused
        point_cloud_cropping_boundaries["box"] = \
            [(-1, 0.1, 0.01), (1., 1., 1.)]
        crop_bounds = point_cloud_cropping_boundaries[self.obj]

        # Point cloud generator
        self.pc_gen = mjpc.PointCloudGenerator(self.sim, \
            min_bound=crop_bounds[0], max_bound=crop_bounds[1])

        # Joints in which objects are manipulable in Mujoco
        #     (may differ from pybullet equivalent)
        self.obj_dofs = None
        if self.obj == "door":
            self.obj_dofs = [self.sim.model.joint_name2id('door_hinge'), \
                self.sim.model.joint_name2id('latch')]
        elif self.obj == "pitcher":
            self.obj_dofs = [self.sim.model.joint_name2id('pitcher_free_joint')]
        elif self.obj == "mug":
            self.obj_dofs = [self.sim.model.joint_name2id('mug_free_joint')]
        elif self.obj == "switch":
            self.obj_dofs = [self.sim.model.joint_name2id('handle')]

        self.initial_obj_joint_values = self.getObjStatus()

        self.finger_joint_range = self.sim.model.jnt_range[:self.tDOF, ]

         # control
        self.cur_qpos = copy.deepcopy(self.sim.data.qpos[:self.tDOF])
        self.target_qpos = copy.deepcopy(self.sim.data.qpos[:self.tDOF])

        self.lock_fingers_closed = lock_fingers_closed

        controller_config_path = self.script_path + "/controller_config.hjson"

        use_rainbow = "rainbow" in os.getcwd()
        print(os.getcwd(), "rainbow" in os.getcwd())
        #if use_rainbow:
        #    controller_config_path = os.path.join(os.path.dirname(os.getcwd()), controller_config_path)

        # load OSC controller 
        with open(controller_config_path) as f:
            params = hjson.load(f)

        # initial impedance is specified in file above
        # fixed impedance for simplicity 
        params['position_orientation']["impedance_flag"]=False
        params['position_orientation']["interpolation"]=None
        control_freq = params['position_orientation']['control_freq']

        self.controller = PositionOrientationController(**params['position_orientation'])
        self.control_timestep = 1.0 / control_freq
        self.model_timestep = 0.01

        self.q_function = None
        
        self.state_space = state_space
        mujoco_env.MujocoEnv.__init__(self, model_filepath, 5)
        
        action_space_size = self.tDOF

        if self.lock_fingers_closed:
            action_space_size = self.tDOF - 6

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_space_size,))
        
        
        observation_shape = self._get_state().shape
        self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=observation_shape)


        # load in grasp cache
        cache_path_qpos = "cached_grasp_IK_task_"+self.obj + ".npy"
        cache_path_qvel = "cached_grasp_qvel_IK_task_"+self.obj + ".npy"
        cache_index_path = "cached_grasp_indices_task_"+self.obj + ".npy"

        if self.state_space == "friendly": 
            cache_state_path = "cached_grasp_observation_task_"+self.obj + "_stateType_friendly" + ".npy"
        else:
            cache_state_path = "cached_grasp_observation_task_"+self.obj + "_stateType_pure" + ".npy"
        
        cache_quality_path = "cached_grasp_quality_task_"+self.obj + ".npy"

        cache_oracle_path = "cached_grasp_oracle_indicies_"+self.obj + ".npy"

        #if "rainbow" in os.getcwd():
        #    cache_path_qpos = os.path.join(os.path.dirname(os.getcwd()), cache_path_qpos)
        #    cache_path_qvel = os.path.join(os.path.dirname(os.getcwd()), cache_path_qvel)
        #    cache_index_path = os.path.join(os.path.dirname(os.getcwd()), cache_index_path)
        #    cache_state_path = os.path.join(os.path.dirname(os.getcwd()), cache_state_path)
        #    cache_quality_path = os.path.join(os.path.dirname(os.getcwd()), cache_quality_path)
        cache_path_qpos = self.script_path + "/../" + cache_path_qpos
        cache_path_qvel = self.script_path + "/../" + cache_path_qvel 
        cache_index_path = self.script_path + "/../" + cache_index_path
        cache_state_path = self.script_path + "/../" + cache_state_path
        cache_quality_path = self.script_path + "/../" + cache_quality_path
        cache_oracle_path = self.script_path + "/../" + cache_oracle_path

        self.cache_grasps_for_task_data = np.load(cache_path_qpos)
        self.cache_grasps_for_task_qvel_data = np.load(cache_path_qvel)
        self.cache_grasp_indices_for_task = np.load(cache_index_path)
        # TODO: add in a conditional to load pure vs friendly state
        self.cache_grasp_state_for_task_data = np.load(cache_state_path)
        self.cache_grasp_quality_for_task_data = np.load(cache_quality_path)

        self.cache_oracle_grasp_indices = np.load(cache_oracle_path).tolist()

        self.cache_torch_state = torch.from_numpy(self.cache_grasp_state_for_task_data).float()

        # softmax the cache_grasp_quality scores
        
        self.cache_grasp_quality_for_task_data = softmax(self.cache_grasp_quality_for_task_data)

        if not gravity:
            self.model.opt.gravity[-1] = 0

        # Initialize the simulator
        self.setUpSim()

        # extra initialization now that we are using Mujoco IK
        self.setGeomIDs()


    def setUpSim(self):
        self.start_joints = [0, np.pi, np.pi, 0, np.pi, 0]
        self.sim.data.qpos[:self.aDOF] = self.start_joints
        self.sim.step()

        self.mj_render()

        self.open_finger_state = copy.deepcopy(self.sim.data.qpos[self.aDOF:self.tDOF])
        self.current_finger_state = copy.deepcopy(self.sim.data.qpos[self.aDOF:self.tDOF])

    def _get_state(self):
        ee_xyz = np.array(copy.deepcopy(self.sim.data.site_xpos[self.sim.model.site_name2id("end_effector")]))
        ee_orientation = np.array(copy.deepcopy(mjpc.mat2Quat(self.sim.data.site_xmat[self.sim.model.site_name2id("end_effector")].reshape(3, 3))))
        if self.state_space == "friendly": 
            return np.hstack((ee_xyz, ee_orientation, self.sim.data.qvel, self.sim.data.qpos, np.abs(self.sim.data.sensordata[:6])))
        elif self.state_space == "pure":
            return np.hstack((self.sim.data.qvel, self.sim.data.qpos, np.abs(self.sim.data.sensordata[:6])))

    def step(self, action):

        try: 
            self.step_env(action)
        except MujocoException: 
            # the simulation is unstable in this case, print a message and reset the env
            print("Simulation unstable, resetting the environment")
            self.reset()
        did_task_succeed = self.check_task_success()
        done = did_task_succeed or self.curr_step > self._max_episode_steps
        reward = self.get_task_reward(self.reward_sparse)
        obs = self._get_state()
        self.curr_step += 1
        if done:
            print(";;;Completed episode with reward", reward)

        if self.visualize:
            self.render()
            self.mj_render()

        info_dict = {}
        info_dict["success"] = float(did_task_succeed)
        info_dict["grasp_index"] = self.random_grasp_idx_chosen
        if self.obj == "door":
            door_hinge = self.sim.model.joint_name2id('door_hinge')
            door_latch = self.sim.model.joint_name2id('latch')
            door_hinge_state = copy.deepcopy(self.sim.data.qpos[door_hinge])
            door_latch_state = copy.deepcopy(self.sim.data.qpos[door_latch])
            info_dict["door_hinge_state"] = door_hinge_state
            info_dict["door_latch_state"] = door_latch_state
        elif self.obj == "switch":
            switch_handle = self.sim.model.joint_name2id('handle')
            switch_state = copy.deepcopy(self.sim.data.qpos[switch_handle])
            info_dict["switch_state"] = switch_state

        return obs, reward, done, info_dict

    def cache_grasps_for_task(self):

        # Generate a point cloud of the graspable scene
        self.generatePointCloud()
        # Sample a set of potential grasp poses using this cloud
        self.generateGraspPoses()
        # Load grasp quality file for predicted quality prior
        if self.filter_low_quality_grasps:
            self.loadGraspQuality()

        grasp_indices_used = []
        cached_results_for_grasp = []
        cached_results_qvel_for_grasp = []
        cached_observation_for_grasp = []

        # using Matt's TOG classifier to score the quality of each grasp
        cached_quality_for_grasp = []

        grasp_indices = list(range(len(self.grasp_poses)))
        for index in grasp_indices:
            self.resetObj()
            self.resetScene()
            self.curr_step = 0

            print("processing index", index, "of", len(grasp_indices))
            # Make sure this grasp has a high enough grasp quality score
            if self.filter_low_quality_grasps and \
                self.grasp_quality_scores[index] < self.minimum_grasp_quality:
                continue
            self.resetScene()

            sampled_pose = self.grasp_poses[index]
            
            grasp_pos, grasp_ori = self.sampledPoseToGraspPose(sampled_pose)

            self.sampled_pose_chosen = copy.deepcopy(grasp_pos)

            goal_joint_pos = self.IKMJ(grasp_pos, grasp_ori)
            validity_code = self.isInvalidMJ(goal_joint_pos)#self.planner.validityChecker.isInvalid(goal_joint_pos)
            ik_correct = self.isIKAtGoal(goal_joint_pos, grasp_pos, grasp_ori)
            
            try: 
                self.executeGraspAtGraspPose(grasp_pos, grasp_ori, goal_joints=goal_joint_pos)
            except: 
                # we failed to find a good grasp at this pose, move on to the next one
                continue
            if validity_code is 0 and ik_correct:
                cached_results_for_grasp.append(copy.deepcopy(self.sim.data.qpos))
                cached_results_qvel_for_grasp.append(copy.deepcopy(self.sim.data.qvel))
                cached_observation_for_grasp.append(copy.deepcopy(self._get_state()))
                grasp_indices_used.append(index)
                cached_quality_for_grasp.append(copy.deepcopy(self.grasp_quality_scores[index]))

        
        cached_results_for_grasp = np.array(cached_results_for_grasp)
        cached_results_qvel_for_grasp = np.array(cached_results_qvel_for_grasp)
        grasp_indices = np.array(grasp_indices)
        cached_observation_for_grasp = np.array(cached_observation_for_grasp)
        cached_quality_for_grasp = np.array(cached_quality_for_grasp)

        np.save("cached_grasp_IK_task_"+self.obj + ".npy", cached_results_for_grasp)
        np.save("cached_grasp_qvel_IK_task_"+self.obj + ".npy", cached_results_qvel_for_grasp)
        np.save("cached_grasp_indices_task_"+self.obj + ".npy", grasp_indices)
        np.save("cached_grasp_observation_task_"+self.obj + "_stateType_" + self.state_space + ".npy", cached_observation_for_grasp)
        np.save("cached_grasp_quality_task_"+self.obj+".npy", cached_quality_for_grasp)

    def set_q_function(self, q_object):
        self.q_function = q_object

    def update_q_values(self):

        # recompute V(s) for all start state grasps
        self.cache_torch_state = self.cache_torch_state.to(self.q_function.device)
        values, _, _ = self.q_function.get_best_qvalue_and_action(self.cache_torch_state)
        self.value_weights = softmax(values.detach().cpu().numpy())

    def reset(self):
        self.resetObj()
        self.resetScene()

        if self.sample_method == "random": 
            #sample a random grasp from our qpos cache
            random_grasp_idx = np.random.randint(low=0, high=len(self.cache_grasps_for_task_data))

        elif self.sample_method == "prior": 

            # sample a grasp weighted by the grasp quality scores that we have cached
            random_grasp_idx = np.random.choice(a=len(self.cache_grasps_for_task_data),
                                                p=self.cache_grasp_quality_for_task_data)
        
        elif self.sample_method == "value":
            # sample a grasp weighted by value function estimate 

            # q not yet provided, use prior, gets around reset bug 
            if self.q_function is None: 
                random_grasp_idx = np.random.choice(a=len(self.cache_grasps_for_task_data),
                                                p=self.cache_grasp_quality_for_task_data)

            else: 
                self.update_q_values()              
                random_grasp_idx = np.random.choice(a=len(self.cache_grasps_for_task_data),
                                                    p=self.value_weights)
        elif self.sample_method == "oracle":
            random_grasp_idx = random.choice(self.cache_oracle_grasp_indices)
        elif "classifier" in self.sample_method:
            # q not yet provided, use prior, gets around reset bug 
            if self.classifier_probs is None:
                random_grasp_idx = np.random.randint(low=0, high=len(self.cache_grasps_for_task_data))

            else:
                random_grasp_idx = np.random.choice(a=len(self.cache_grasps_for_task_data),
                                                    p=self.classifier_probs)
        else:
            print('sample_method provided to MujocoGraspEnv is not supported. choices are random, prior, value')
            sys.exit()

        print(";;;;Executing grasp at id", self.random_grasp_idx_chosen, self.cache_grasp_indices_for_task[self.random_grasp_idx_chosen])

        self.random_grasp_idx_chosen = random_grasp_idx
        #grasp_qpos = copy.deepcopy(self.cache_grasps_for_task[random_grasp_idx])
        # teleport arm + fingers to this qpos
        # self.teleportArmPlusFingersToJointState(grasp_qpos)

        # set sim state
        self.sim.data.qpos[:] = self.cache_grasps_for_task_data[random_grasp_idx]
        self.sim.data.qvel[:] = self.cache_grasps_for_task_qvel_data[random_grasp_idx]
        self.sim.forward()

        self.cur_qpos = copy.deepcopy(self.sim.data.qpos[:self.tDOF])
        self.target_qpos = copy.deepcopy(self.sim.data.qpos[:self.tDOF])

        self.controller.reset() 

        # simDataFile = filename = "cache_"+self.obj+ "/" + str(random_grasp_idx) +"_"+self.obj+".mjb"
        # model = load_model_from_mjb(simDataFile)
        # self.sim.set_state(model)


        # max_attempts = 100
        # attempts_made = 0
        # grasp_succeeded = False
        # while attempts_made < max_attempts and not grasp_succeeded:
        #     attempts_made += 1
        #     try:
        #         grasp_pos, grasp_ori, joint_state = self.sampleRandomValidGraspPose()
        #         self.executeGraspAtGraspPose(grasp_pos, grasp_ori, goal_joints=joint_state)
        #         grasp_succeeded = True
        #     except:
        #         print("Grasp attempt failed, trying again.")
        # if not(grasp_succeeded):
        #     print("Could not successfully complete a grasp in {} attempts.".format(max_attempts))
        #     sys.exit()
        
        self.curr_step = 0

        return self._get_state()

    def mj_render(self):
        if self.viewer is not None:
            self.viewer.render()
        else:
            self.sim.render(255, 255, camera_name="cam_0")

    def sampleRandomValidGraspPoseFromCache(self):
        random_grasp_cache = list(range(self.grasp_qpos_cache))

    def sampleRandomValidGraspPose(self):
        random_grasp_indices = list(range(len(self.grasp_poses)))
        random.shuffle(random_grasp_indices)
        for index in random_grasp_indices:
            # Make sure this grasp has a high enough grasp quality score
            if self.filter_low_quality_grasps and \
                self.grasp_quality_scores[index] < self.minimum_grasp_quality:
                continue
            self.resetScene()

            sampled_pose = self.grasp_poses[index]
            
            grasp_pos, grasp_ori = self.sampledPoseToGraspPose(sampled_pose)

            self.sampled_pose_chosen = copy.deepcopy(grasp_pos)

            goal_joint_pos = self.IKMJ(grasp_pos, grasp_ori)
            validity_code = self.isInvalidMJ(goal_joint_pos)#self.planner.validityChecker.isInvalid(goal_joint_pos)
            ik_correct = self.isIKAtGoal(goal_joint_pos, grasp_pos, grasp_ori)
            if validity_code is 0 and ik_correct:
                return (grasp_pos, grasp_ori, goal_joint_pos)
        sys.exit("All sampled grasp poses were invalid.")

    def getGraspPoseAtIndex(self, index):
        if index >= len(self.grasp_poses):
            sys.exit("Requested grasp pose at index", index, \
                "but only", len(self.grasp_poses), "were generated.")
        sampled_pose = self.grasp_poses[index]
        grasp_pos, grasp_ori = self.sampledPoseToGraspPose(sampled_pose)
        return (grasp_pos, grasp_ori)

    def teleportArmToJointState(self, joint_state):
        self.resetScene()
        self.sim.data.qpos[:self.aDOF] = joint_state
        self.sim.data.qvel[:self.aDOF] = [0]*self.aDOF
        self.sim.step()

    def teleportArmPlusFingersToJointState(self, joint_finger_state):
        self.resetScene()
        self.sim.data.qpos[:self.tDOF] = joint_finger_state
        # self.sim.data.qvel[:self.tDOF] = [0]*self.tDOF
        self.sim.step()

    def resetScene(self):
        # Reset simulator
        self.sim.reset()
        # Reset arm
        self.sim.data.qpos[:self.aDOF] = self.start_joints
        # Reset fingers
        self.sim.data.qpos[self.aDOF:self.tDOF] = self.open_finger_state

        #self.planner.validityChecker.updateFingerState(self.open_finger_state)
        self.current_finger_state = self.open_finger_state
        # set the fingers to have 0 velocity
        self.sim.data.qvel[:self.tDOF] = 0

        # Reset object
        self.resetObj()

        # self.planner.validityChecker.resetRobot(self.start_joints)
        # self.planner.validityChecker.resetScene()

        self.sim.step()

    def visualizePointCloud(self):
        world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([self.cloud_with_normals, world_axes])

    def getObjStatus(self):
        return copy.deepcopy(self.sim.data.qpos[self.tDOF:])

    def resetObj(self):
        for i, mj_id in enumerate(self.obj_dofs):
            self.sim.data.qpos[mj_id]=self.initial_obj_joint_values[i]

    def generatePointCloud(self):
        self.cloud_with_normals = self.pc_gen.generateCroppedPointCloud()
        return self.cloud_with_normals

    # Generates grasp poses at every point in the point cloud. When sampling
    #     orientation, optionally rotate each candidate by provided angles
    #     about the approach axis to generate additional candidates.
    def generateGraspPoses(self, rotation_values_about_approach=[0]):
        num_points = np.asarray(self.cloud_with_normals.points).shape[0]

        # Grasp pose generator
        pose_gen = gpg.GraspPoseGenerator(self.cloud_with_normals, \
            rotation_values_about_approach=rotation_values_about_approach)

        self.grasp_poses = []
        for i in range(num_points):
            self.grasp_poses += pose_gen.proposeGraspPosesAtCloudIndex(i)

        return self.grasp_poses

    # Loads stored predicted grasp quality labels, which are used to filter
    #     grasps that are unlikely to succeed. Checks that grasp poses on
    #     which labels were generated are the same as the ones generated.
    def loadGraspQuality(self):
        # Assumes data directory relative to this script is fixed in GitHub repo
        data_dir = os.path.dirname(os.path.realpath(__file__)) + '/../data/'

        loaded_grasp_poses = file_io.loadGraspFile(\
            "/{}_grasp_poses.txt".format(self.obj), data_dir)
        self.grasp_quality_scores = file_io.loadGraspQualityFile(\
            "/{}_grasp_quality.txt".format(self.obj), data_dir)

        if len(loaded_grasp_poses) != len(self.grasp_quality_scores):
            print("Loaded {} grasp poses but {} grasp quality scores.".format(\
                len(loaded_grasp_poses), len(self.grasp_quality_scores)))
            sys.exit()
        if len(loaded_grasp_poses) != len(self.grasp_poses):
            print("Generated {} grasp poses but loaded {} grasp poses when loading quality scores.".format(\
                len(self.grasp_poses), len(loaded_grasp_poses)))
            print("MAKE SURE THE ENVIRONMENT HASN'T CHANGED. Using loaded grasp poses instead.")
            loaded_grasp_positions = [pose[:3, 3] for pose in loaded_grasp_poses]
            num_grasp_positions_not_in_generated_pose_set = 0

            def vecsEqual(v1, v2, thresh=1e-4):
                if len(v1) != len(v2):
                    return False
                for i in range(len(v1)):
                    if np.abs(v1[i]-v2[i]) > thresh:
                        return False
                return True

            def vecInVecList(vec, vec_list):
                for possible_match in vec_list:
                    if vecsEqual(vec, possible_match):
                        return True
                return False

            for pose_index in range(len(self.grasp_poses)):
                grasp_pos = self.grasp_poses[pose_index][:3, 3]
                if not vecInVecList(grasp_pos, loaded_grasp_positions):
                    num_grasp_positions_not_in_generated_pose_set += 1
            self.grasp_poses = loaded_grasp_poses
            print(num_grasp_positions_not_in_generated_pose_set, \
                "generated grasps did not appear in loaded grasps.", \
                "Using loaded grasps instead, but be sure that the environment hasn't changed.")
        else:
            num_grasps_orientations_dont_match = 0
            for pose_index in range(len(self.grasp_poses)):
                max_difference_between_generated_and_loaded_grasp_pose_positions = \
                    np.max(np.abs(loaded_grasp_poses[pose_index][:3, 3]-self.grasp_poses[pose_index][:3, 3]))
                if max_difference_between_generated_and_loaded_grasp_pose_positions > 1e-4:
                    print("While loading grasp quality, found grasp positions that differ by {}.".format(\
                        max_difference_between_generated_and_loaded_grasp_pose_positions))
                    print(pose_index, loaded_grasp_poses[pose_index], self.grasp_poses[pose_index])
                    sys.exit()
                orientation_difference = mjpc.quatDiff(\
                    mjpc.mat2Quat(loaded_grasp_poses[pose_index][:3, :3]), \
                    mjpc.mat2Quat(self.grasp_poses[pose_index][:3, :3]))
                if orientation_difference > 1e-4:
                    print("While loading grasp quality, found grasp orientations that differ by {}.".format(\
                        orientation_difference))
                    print(pose_index, loaded_grasp_poses[pose_index], self.grasp_poses[pose_index])
                    print("Using saved grasp pose")
                    self.grasp_poses[pose_index] = loaded_grasp_poses[pose_index]
                    num_grasps_orientations_dont_match += 1
            print(num_grasps_orientations_dont_match, \
                "generated grasps did not match the orientations of loaded grasps.", \
                "Probably off by pi about approach axis. Using loaded poses here instead.")

        num_grasps_above_threshold = 0
        for pose_index in range(len(self.grasp_poses)):
            if self.grasp_quality_scores[pose_index] >= self.minimum_grasp_quality:
                    num_grasps_above_threshold += 1
        if len(self.grasp_quality_scores) != len(self.grasp_poses):
            print("Quality scores {} should correspond to grasp poses {}.".format(\
                len(self.grasp_quality_scores), len(self.grasp_poses)))
            sys.exit()
        print("{} of {} grasp poses are above quality threshold.".format(\
            num_grasps_above_threshold, len(self.grasp_quality_scores)))
        return self.grasp_quality_scores

    # Convert sampled point cloud point and grasp orientation to grasp
    #     position and orientation by translating along approach direction.
    def sampledPoseToGraspPose(self, sampled_pose):
        grasp_pose = copy.deepcopy(sampled_pose)
        grasp_pose = gpg.translateFrameNegativeZ(grasp_pose, self.dist_from_point_to_ee_link)
        grasp_position, grasp_orientation = mjpc.mat2PosQuat(grasp_pose)
        return grasp_position, grasp_orientation

    def executeGraspAtGraspPose(self, grasp_position, grasp_orientation, goal_joints=None):

        self.resetScene()

        goal_joint_pos = goal_joints if goal_joints is not None else self.IKMJ(grasp_position, grasp_orientation)
        grasp_invalid_code = self.isInvalidMJ(goal_joint_pos)#self.planner.validityChecker.isInvalid(goal_joint_pos)
        ik_correct = self.isIKAtGoal(goal_joint_pos, grasp_position, grasp_orientation)

        if grasp_invalid_code > 0 or not ik_correct:
            print("Cannot execute grasp at invalid pose:", grasp_position, grasp_orientation)
            sys.exit()

        # Since grasp goal contains position and velocity, assume 0 velocity
        #     and send only position
        self.teleportArmToJointState(goal_joint_pos[:self.aDOF])
        #print("\n;;;;; IK FK Test. Goal is", grasp_position, grasp_orientation)
        ee_site_id = self.sim.model.site_name2id("end_effector")
        #print(";;;;; Site id", ee_site_id)
        #print(";;;;; Site pose", self.sim.data.site_xpos[ee_site_id], mjpc.mat2Quat(self.sim.data.site_xmat[ee_site_id].reshape(3,3)))
        #print("\n")
        self.closeFingers()

        self.cur_qpos = copy.deepcopy(self.sim.data.qpos[:self.tDOF])
        self.target_qpos = copy.deepcopy(self.sim.data.qpos[:self.tDOF])
        return 

    # https://github.com/babbatem/motor_skills/blob/impedance/motor_skills/cip/MjGraspHead.py
    #     (link might be outdated)
    # Closes the gripper's fingers until contact is made with the object.
    #     If contact is made with the proximal link, distal link continues to
    #     close. The controller continues to close the fingers for a set
    #     number of steps after contact is first made. Contact is defined by a
    #     force threshold. An additional desired torque is added.
    def closeFingers(self):
        # % close fingers
        new_pos = copy.deepcopy(self.sim.data.qpos[:self.tDOF])
        step_where_contact_made = [[self.grasp_steps for link in range(self.num_links_per_finger)] for finger in range(int(self.gDOF/self.num_links_per_finger))]
        torque_goal = [0] * self.tDOF
        for t in range(self.grasp_steps):
            # % see which sensors are reporting force
            touched = np.where(np.abs(self.sim.data.sensordata[:6]) > self.minimum_sensor_output_for_contact)[0]

            # check each finger to determine if links are in contact

            fingers_stopped = [False]*int(self.gDOF/self.num_links_per_finger)
            touched_finger_idxs = [touched_link + self.aDOF for touched_link in touched]
            for finger_i in range(int(self.gDOF/self.num_links_per_finger)):
                base_idx = self.finger_base_idxs[finger_i]
                tip_idx = self.finger_tip_idxs[finger_i]
                if tip_idx in touched_finger_idxs:
                    if step_where_contact_made[finger_i][1] == self.grasp_steps:
                        # This is the first step where contact is made. Don't stop distal link yet, stop proximal link.
                        step_where_contact_made[finger_i][1] = t
                        new_pos[tip_idx] += self.max_finger_delta/self.grasp_steps
                    elif t > step_where_contact_made[finger_i][1] + self.grasp_steps_after_contact:
                        # grasp_steps_after_contact steps have been taken after first contact and we're still in contact. Stop the finger.
                        fingers_stopped[finger_i] = True
                    else:
                        # First contact had been made but we've not yet passed grasp_steps_after_contact additional steps
                        new_pos[tip_idx] += self.max_finger_delta/self.grasp_steps
                elif base_idx in touched_finger_idxs:
                    if step_where_contact_made[finger_i][0] == self.grasp_steps:
                        # This is the first step where contact is made. Don't stop yet.
                        step_where_contact_made[finger_i][0] = t
                        new_pos[tip_idx] += self.max_finger_delta/self.grasp_steps
                        new_pos[base_idx] += self.max_finger_delta/self.grasp_steps
                    elif t > step_where_contact_made[finger_i][0] + self.grasp_steps_after_contact:
                        # grasp_steps_after_contact steps have been taken after first contact and we're still in contact. Stop the finger.
                        fingers_stopped[finger_i] = True
                    else:
                        # First contact had been made but we've not yet passed grasp_steps_after_contact additional steps
                        new_pos[tip_idx] += self.max_finger_delta/self.grasp_steps
                        new_pos[base_idx] += self.max_finger_delta/self.grasp_steps
                else:
                    new_pos[base_idx] += self.max_finger_delta/self.grasp_steps
                    new_pos[tip_idx] += self.max_finger_delta/self.grasp_steps
                # In any case, assign a goal baseline torque
                torque_goal[base_idx] = self.finger_base_desired_torque
                torque_goal[tip_idx] = self.finger_tip_desired_torque

            # Once all fingers have stopped, break
            if len(fingers_stopped) == sum(fingers_stopped):
                break

            # Make sure goals are within limits
            for f_idx in self.finger_joint_idxs:
                new_pos[f_idx] = max(self.finger_joint_range[f_idx, 0], new_pos[f_idx])
                new_pos[f_idx] = min(self.finger_joint_range[f_idx, 1], new_pos[f_idx])

            # % compute torque and step
            self.sim.data.ctrl[:] = mjc.pd(torque_goal, [0] * self.tDOF, new_pos, self.sim, ndof=self.tDOF, kp=np.eye(self.tDOF)*300)
            self.sim.forward()
            self.sim.step()

            self.mj_render()

        # Make sure goals are within limits
        for f_idx in self.finger_joint_idxs:
            new_pos[f_idx] = max(self.finger_joint_range[f_idx, 0], new_pos[f_idx])
            new_pos[f_idx] = min(self.finger_joint_range[f_idx, 1], new_pos[f_idx])

        # Make sure it reaches the goal
        for t in range(200):
            self.sim.data.ctrl[:] = mjc.pd(torque_goal, [0] * self.tDOF, new_pos, self.sim, ndof=self.tDOF, kp=np.eye(self.tDOF)*300)
            self.sim.step()
            self.mj_render()

    '''
    Step the environment forward with an action vector of joint torques
    '''
    def step_env(self, action):

        # action is interpreted as (delta position, delta orientation)
        # (orientation as euler angles)

        # gripper info 
        self.grp_target = self.target_qpos
        self.grp_idx = np.arange(self.aDOF, self.tDOF)

        policy_step = True
        for t in range(int(self.control_timestep / self.model_timestep)):

            # update model
            # HACK: try/except gets around initial reset bug. 
            # issue is singular mass matrix at the initial joint config. 
            try:
                self.controller.update_model(self.sim,
                                         id_name='j2s6s300_link_6',
                                         joint_index=np.arange(6))
            except Exception as e:
                print(e)
                return 
            
            # compute arm, gripper torques
            torques = self.controller.action_to_torques(action,
                                                        policy_step)
            torques += self.sim.data.qfrc_bias[:self.aDOF]
            gripper_torques = mjc.pd(None,
                                 np.zeros(len(self.sim.data.ctrl)),
                                 self.grp_target,
                                 self.sim,
                                 kp=np.eye(12)*1500)

            gripper_torques=gripper_torques[self.grp_idx]
            all_torques = np.concatenate((torques, gripper_torques))
            self.sim.data.ctrl[:] = all_torques
            self.sim.forward()
            self.sim.step()

            policy_step = False

            if self.visualize and self.viewer is not None:
                self.viewer.render()

    def get_task_reward(self, reward_sparse):
        if self.obj == "door":
            if reward_sparse:
                return self.check_task_success()*1
            else:

                wrist_ft_xyz = copy.deepcopy(self.sim.data.site_xpos[self.sim.model.site_name2id("end_effector")])
                if self.sampled_pose_chosen is None:
                    print("Should only happen ONCE!")
                    self.sampled_pose_chosen = copy.deepcopy(self.sim.data.site_xpos[self.sim.model.site_name2id("S_handle")])
                dist = np.linalg.norm(self.sampled_pose_chosen - wrist_ft_xyz)
                reward = 0

                if not self.lock_fingers_closed: 
                    reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
                    reward += reaching_reward
                
                # Add rotating component if we're using a locked door
                door_hinge_idx = self.sim.model.joint_name2id('door_hinge')
                door_latch_idx = self.sim.model.joint_name2id('latch')
                
                handle_qpos = copy.deepcopy(self.sim.data.qpos[door_latch_idx])
                reward += np.clip(0.25 * np.abs(handle_qpos / (0.5 * np.pi)), -0.25, 0.25)

                # add hinge qpos component 
                hinge_qpos = copy.deepcopy(self.sim.data.qpos[door_hinge_idx])
                reward += np.clip(hinge_qpos, 0, 0.5)

                return reward
        elif self.obj == "switch":
            if reward_sparse:
                return self.check_task_success()*1
            else:
                switch_handle = self.sim.model.joint_name2id('handle')
                wrist_ft_xyz = copy.deepcopy(self.sim.data.site_xpos[self.sim.model.site_name2id("end_effector")])
                if self.sampled_pose_chosen is None:
                    print("Should only happen ONCE!")
                    self.sampled_pose_chosen = copy.deepcopy(self.sim.data.body_xpos[self.sim.model.body_name2id("handle")])
                dist = np.linalg.norm(self.sampled_pose_chosen - wrist_ft_xyz)
                reward = 0
                
                if not self.lock_fingers_closed: 
                    reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
                    reward += reaching_reward
                
                # add hinge qpos component 
                hinge_qpos = copy.deepcopy(self.sim.data.qpos[switch_handle])
                reward += np.clip(hinge_qpos, -1.0, 1.0)
                return reward


        elif self.obj == "pitcher":
            reward = 0.

            pitcher_position = copy.deepcopy(self.sim.data.get_body_xpos("pitcher"))
            pitcher_site = copy.deepcopy(self.sim.data.get_site_xpos("pitcher_site"))
            pitcher_xmat = copy.deepcopy(self.sim.data.get_body_xmat("pitcher"))
            pitcher_rotation_down = np.dot(pitcher_xmat[:, 2], np.array([0, 0, 1]))
            
            MAX_DISTANCE = 0.316 # distance at the start of simulation
            DISTANCE_THRESHOLD = 0.1
            d = np.linalg.norm(pitcher_site[0:2] - pitcher_position[0:2])     
            reaching_reward = 0.5 * (1 - np.tanh(10.0 * d))
            reward += reaching_reward 

            if d < DISTANCE_THRESHOLD:        
                # check that the pitcher is facing down.
                pitcher_rot_z = copy.deepcopy(self.sim.data.get_body_xmat("pitcher")[:, 2])
                world_z = np.array([0, 0, 1])
                score = np.dot(pitcher_rot_z, world_z)

                # dot product == 1 --> pitcher is upright 
                # dot product == -1 --> pitcher is upside down
                # thus our reward should be max at -1 and min at 1
                rot_bonus = -0.25 * np.tanh(2*score) + 0.25
                reward += np.clip(rot_bonus, 0., 0.5)

            return reward

    def check_task_success(self):
        easy_or_hard = "harderer"

        if self.obj == "door":
            # check to see if the door hinge is open
            door_hinge = self.sim.model.joint_name2id('door_hinge')
            door_latch = self.sim.model.joint_name2id('latch')
            door_state, door_thresh = None, None
            if easy_or_hard == "easy":
                door_state = copy.deepcopy(self.sim.data.qpos[door_latch])
                door_thresh = 0.1
            elif easy_or_hard == "hard":
                door_state = copy.deepcopy(self.sim.data.qpos[door_hinge])
                door_thresh = 0.25
            elif easy_or_hard == "harderer":
                door_state = copy.deepcopy(self.sim.data.qpos[door_hinge])
                door_thresh = 0.5
            print("&&&Current door state:", door_state)
            if easy_or_hard == "easy":
                door_state = np.abs(door_state)
            return door_state > door_thresh

        elif self.obj == "switch":
            # check to see if the switch has been flipped up
            # switch starts off at 0, and fully up is pi radians, check to make sure we are 90% flipped
            switch_handle = self.sim.model.joint_name2id('handle')
            switch_state = copy.deepcopy(self.sim.data.qpos[switch_handle])
            switch_thresh = None
            if easy_or_hard == "easy":
                switch_thresh = 0.1
            elif easy_or_hard == "hard":
                switch_thresh = 3
            elif easy_or_hard == "harderer":
                switch_thresh = 3
            elif easy_or_hard == "hardererer":
                switch_thresh = 3.5
            elif easy_or_hard == "harderererer":
                switch_thresh = 4
            print("&&&Current switch state:", switch_state)
            return switch_state > switch_thresh
        elif self.obj == "pitcher":
            pitcher_position = copy.deepcopy(self.sim.data.get_body_xpos("pitcher"))
            pitcher_site = copy.deepcopy(self.sim.data.get_site_xpos("pitcher_site"))
            pitcher_near_pitcher_site = (np.linalg.norm(pitcher_site[0:2] - pitcher_position[0:2]) < 0.02)

            # check that the pitcher is facing down.
            pitcher_facing_down = (np.sign(np.dot(self.sim.data.get_body_xmat("pitcher")[:, 2], np.array([0, 0, 1]))) == -1)
            #pitcher_facing_down_score = (np.dot(self.sim.data.get_body_xmat("pitcher")[:, 2], np.array([0, 0, 1])))

            #print("pitcher facing down score:", pitcher_facing_down_score)

            return pitcher_near_pitcher_site and pitcher_facing_down
        elif self.obj == "mug":
            mug_pos = copy.deepcopy(self.sim.data.get_body_xpos("mug"))
            mug_vel = copy.deepcopy(self.sim.data.get_body_xvelp("mug"))
            post_position = copy.deepcopy(self.sim.data.get_body_xpos("post"))
            current_joint_vals = copy.deepcopy(self.sim.data.qpos[:self.aDOF])
            #current_end_effector_pos, current_quat_xyzw = self.planner.calculateForwardKinematics(0, self.aDOF, current_joint_vals.tolist())

            # check distance between mug and current_pos of end effector is > than 0.3
            #arm_far_from_mug = np.linalg.norm(current_end_effector_pos-mug_pos) > 3

            # ensure the mug has not fallen back on to the table
            mug_above_ground = mug_pos[2] > 0.075

            # also check the velocity of the mug is low so it isn't like flying through the air
            mug_velocity_low = (np.linalg.norm(mug_vel) < 0.03)

            # check the distance to the post is small
            mug_near_post = (np.linalg.norm(mug_pos - post_position) < 0.1)
            return False
            #return arm_far_from_mug and mug_above_ground and mug_velocity_low and mug_near_post

    # Collision checking and IK functions in Mujoco ############################

    def setGeomIDs(self):
        self.ground_geom_id = None
        self.robot_geom_ids = []
        self.obj_geom_ids = []
        for n in range(self.sim.model.ngeom):
            body = self.sim.model.geom_bodyid[n]
            body_name = self.sim.model.body_id2name(body)
            geom_name = self.sim.model.geom_id2name(n)
            #print(n, body, geom_name, body_name)
            if geom_name == "ground" and body_name == "world":
                self.ground_geom_id = n
            elif "j2s6s300_link_" in body_name:
                self.robot_geom_ids.append(n)
            elif body_name != "world":
                self.obj_geom_ids.append(n)

    def contactBetweenRobotAndObj(self, contact):
        if contact.geom1 in self.robot_geom_ids and contact.geom2 in self.obj_geom_ids:
            return True
        if contact.geom2 in self.robot_geom_ids and contact.geom1 in self.obj_geom_ids:
            return True
        return False

    def contactBetweenRobotAndFloor(self, contact):
        if contact.geom1 == self.ground_geom_id and contact.geom2 in self.robot_geom_ids:
            return True
        if contact.geom2 == self.ground_geom_id and contact.geom1 in self.robot_geom_ids:
            return True
        return False

    def jointsViolateLimits(self, joint_vals):
        for i, joint_val in enumerate(joint_vals):
            if joint_val < self.sim.model.jnt_range[i][0] or joint_val > self.sim.model.jnt_range[i][1]:
                return True
        return False

    # Returns 0 if the given joint pose is a valid, collision-free joint config.
    #     Returns 1, 2, or 3 if arm collides with object, ground plane, or
    #     exceeds joint limits.
    def isInvalidMJ(self, joint_pos):
        self.resetScene()
        self.teleportArmToJointState(joint_pos[:self.aDOF])

        if self.jointsViolateLimits(joint_pos[:self.aDOF]):
            return 3
        # Note that the contact array has more than `ncon` entries,
        # so be careful to only read the valid entries.
        for contact_index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[contact_index]
            if self.contactBetweenRobotAndObj(contact):
                return 1
            elif self.contactBetweenRobotAndFloor(contact):
                return 2
        return 0

    def isIKAtGoal(self, joints, goal_pos, goal_quat, pos_thresh=1e-3, quat_thresh=1e-3):
        # Check to make sure the returned results acutually reach the goal in our simulation.
        self.teleportArmToJointState(joints[:self.aDOF])
        ee_site_id = self.sim.model.site_name2id("end_effector")
        actual_pos = self.sim.data.site_xpos[ee_site_id]
        actual_quat = mjpc.mat2Quat(self.sim.data.site_xmat[ee_site_id].reshape(3,3))

        max_directional_pos_diff = np.max(np.abs(np.array(actual_pos) - np.array(goal_pos)))
        quat_diff = mjpc.quatDiff(goal_quat, actual_quat)

        ik_is_at_goal = max_directional_pos_diff < pos_thresh and quat_diff < quat_thresh
        #if not ik_is_at_goal:
        #print("IK is not at goal: position is off by {:0.6} in one direction, quaternion off by {:0.6}".format(max_directional_pos_diff, quat_diff))
        return ik_is_at_goal

    def IKMJ(self, grasp_position, grasp_orientation, target_site="end_effector"):
        self.resetScene()

        if self.obj == "door" or self.obj == "pitcher" or self.obj == "switch":
            physics = Physics.from_xml_path(self.script_path + "/../assets/kinova_j2s6s300/mujoco-arm-ik_{}.xml".format(self.obj))
        else:
            physics = Physics.from_xml_path(self.script_path + "/../assets/kinova_j2s6s300/mujoco-arm-ik.xml")

        # set joint_names to None to use all available joints during IK optimization
        ik_results = qpos_from_site_pose(physics=physics, site_name=target_site, target_pos=grasp_position, target_quat=grasp_orientation, joint_names = None, )
        return ik_results.qpos

    ############################################################################

            
def test():
    parser = argparse.ArgumentParser(description='Generate binary task label dataset from hard-coded policy.')
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.add_argument('--headless', dest='visualize', action='store_false')
    parser.set_defaults(visualize=True)
    parser.add_argument('--grasp_index', type=int, default=-1, \
        help='Index of grasp to execute. If -1, select one at random.')
    parser.add_argument('--obj', type=str, default="door")
    args = parser.parse_args()

    from mujoco_py import GlfwContext
    GlfwContext(offscreen=True)  # Create a window to init GLFW.

    mge = MujocoGraspEnv("door", visualize=False, lock_fingers_closed=True, state_space="friendly")
    mge.cache_grasps_for_task()
    sys.exit()
    # # #mge.visualizePointCloud()

    # mge.check_task_success()

    # grasp_pos, grasp_ori = None, None
    # if args.grasp_index == -1:
    #     grasp_pos, grasp_ori, joint_state = mge.sampleRandomValidGraspPose()
    # else:
    #     grasp_pos, grasp_ori = mge.getGraspPoseAtIndex(args.grasp_index)

    # mge.executeGraspAtGraspPose(grasp_pos, grasp_ori)

    mge.reset()
    while True:
        # a = np.random.normal(loc=0., scale=1.0, size=mge.action_space.shape)
        # a = mge.action_space.sample()
        a = [0]*6
        a[0] = -1
        # a[1] = -0.1
        a[2] = -1
        a[3] = 0.1
        # a = [0]*12
        # a[1]=-0.01
        mge.step(a)

if __name__ == '__main__':
    test()
