from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.backend.observation import Observation
from rlbench.tasks import *
from typing import List
from gym import spaces
import numpy as np

# list of state types
state_types = ['left_shoulder_rgb',
               'left_shoulder_depth',
               'left_shoulder_mask',
               'right_shoulder_rgb',
               'right_shoulder_depth',
               'right_shoulder_mask',
               'wrist_rgb',
               'wrist_depth',
               'wrist_mask',
               'joint_velocities',
               'joint_velocities_noise',
               'joint_positions',
               'joint_positions_noise',
               'joint_forces',
               'joint_forces_noise',
               'gripper_pose',
               'gripper_touch_forces',
               'task_low_dim_state']

image_types = ['left_shoulder_rgb',
               'left_shoulder_depth',
               'left_shoulder_mask',
               'right_shoulder_rgb',
               'right_shoulder_depth',
               'right_shoulder_mask',
               'wrist_rgb',
               'wrist_depth',
               'wrist_mask', ]


class SimulationEnvironment():
    """
    This can be a parent class from which we can have multiple child classes that 
    can diversify for different tasks and deeper functions within the tasks.
    """

    def __init__(self,
                 task_name,
                 state_type_list=['left_shoulder_rgb'],
                 dataset_root='',
                 observation_mode='state',
                 headless=True):
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
        self._observation_mode = observation_mode
        self.env = Environment(
            action_mode, dataset_root, obs_config=obs_config, headless=headless)
        # Dont need to call launch as task.get_task can launch env.
        self.env.launch()
        self.task = self.env.get_task(task_name)
        _, obs = self.task.reset()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.action_size,), dtype=np.float32)
#         self.logger = logger.create_logger(__class__.__name__)
#         self.logger.propagate = 0
        if len(state_type_list) > 0:
            self.observation_space = []
            # for state_type in state_type_list:
            #     state = getattr(obs, state_type)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape)      
        else:
            raise ValueError('No State Type!')

        self.state_type_list = state_type_list

    def _get_state(self, obs):
        if len(self.state_type_list) > 0:
            if self._observation_mode == 'state':
                return self.get_low_dim_data(obs)
            elif self._observation_mode == 'vision':
                return None

    def get_low_dim_data(self, obs) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional obseervations.
        :return: 1D array of observations.
        """
        low_dim_data = [] if obs.gripper_open is None else [[obs.gripper_open]]
        for data in [obs.joint_velocities, obs.joint_positions,
                     obs.joint_forces,
                     obs.gripper_pose, obs.gripper_joint_positions,
                     obs.gripper_touch_forces, obs.task_low_dim_state]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data)


    def reset(self):
        descriptions, obs = self.task.reset()
        return self._get_state(obs)

    def step(self, action):
        # reward in original rlbench is binary for success or not
        obs_, reward, terminate = self.task.step(action)
        return self._get_state(obs_), reward, terminate, None

    def shutdown(self):
        #         self.logger.info("Environment Shutdown! Create New Instance If u want to start again")
        #         self.logger.handlers.pop()
        self.env.shutdown()
