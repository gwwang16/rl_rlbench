# -*- coding: utf-8 -*-

import os
import numpy as np
import parl
from parl import layers
# from parl.utils import logger

from parl.utils import action_mapping  # 将神经网络输出映射到对应的 实际动作取值范围内
from parl.utils import ReplayMemory  # 经验回放

from td3_model import RLBenchModel
from td3_agent import RLBenchAgent
from parl.algorithms import TD3

import gym
import rlbench.gym
from custom_logger import CustomLogger

MAX_EPISODES = 20000
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99
TAU = 0.005
MEMORY_SIZE = int(1e6)
WARMUP_SIZE = 1e3
BATCH_SIZE = 256
EXPL_NOISE = 0.1  # Std of Gaussian exploration noise
EPISODE_LENGTH = 200  # max steps in each episode
TEST_EVERY_STEPS = 200  # e2  # 每个N步评估一下算法效果，每次评估5个episode求平均reward
REWARD_SCALE = 1


def distance_cal(obs):
    '''calculate distance between end effector and target'''
    ee_pose = np.array(obs[22:25])
    target_pose = np.array(obs[-3:])

    distance = np.sqrt((target_pose[0]-ee_pose[0])**2 +
                       (target_pose[1]-ee_pose[1])**2+(target_pose[2]-ee_pose[2])**2)

    # reward = np.tanh(1/(distance+1e-5))
    reward = np.exp(-1*distance)

    return distance, reward


def run_train_episode(env, agent, rpm):
    obs = env.reset()
    total_reward = 0
    steps = 0
    # Use joint positions only
    target_pose = np.expand_dims(obs[-3:], axis=0)
    max_action = float(env.action_space.high[0])

    for steps in range(EPISODE_LENGTH):

        batch_obs = np.expand_dims(obs[8:15], axis=0)
        batch_obs_full = np.concatenate((batch_obs, target_pose), axis=1)

        if rpm.size() < WARMUP_SIZE:
            action = env.action_space.sample()

        else:
            action = agent.predict(batch_obs_full.astype('float32'))
            # Add gripper action here
            action = np.append(action, 0)
            action = np.squeeze(action)

            # Add exploration noise, and clip to [-max_action, max_action]
            action = np.clip(
                np.random.normal(action, EXPL_NOISE * max_action), -max_action,
                max_action)

        next_obs, reward, done, info = env.step(action)

        # Use joint positions and target position only
        obs_full = np.concatenate((obs[8:15], obs[-3:]))
        next_obs_full = np.concatenate((next_obs[8:15], next_obs[-3:]))

        # feed first 7 action into rpm here
        rpm.append(obs_full, action[0:7], reward, next_obs_full, done)

        if rpm.size() > WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

        obs = next_obs
        total_reward += reward

        distance, _ = distance_cal(obs)

        if done:
            break

    return total_reward, distance


def evaluate(env, agent, render=False):

    obs = env.reset()
    total_reward = 0
    target_pose = np.expand_dims(obs[-3:], axis=0)

    for i in range(EPISODE_LENGTH):
        batch_obs = np.expand_dims(obs[8:15], axis=0)
        batch_obs_full = np.concatenate((batch_obs, target_pose), axis=1)
        action = agent.predict(batch_obs_full.astype('float32'))

        # Add gripper action again
        action = np.append(action, 1)
        
        action = np.squeeze(action)
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        next_obs, reward, done, info = env.step(action)

        obs = next_obs
        total_reward += reward

        if render:
            env.render()

        if done:
            break

    return total_reward


# logger = CustomLogger('train_log/train_gym.txt')
# Create rlbench gym env
env = gym.make('reach_target-state-v0', render_mode='human')

env.reset()
obs_dim = 7 + 3  # 7 joint positions plus 3 target poses
# drop gripper action to speed up training
act_dim = env.action_space.shape[0]-1

max_action = float(env.action_space.high[0])

model = RLBenchModel(act_dim, max_action)
algorithm = TD3(model, max_action=max_action,
                gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR,
                critic_lr=CRITIC_LR)
agent = RLBenchAgent(algorithm, obs_dim, act_dim)
# rpm = ReplayMemory(MEMORY_SIZE, obs_dim, act_dim)
# load model
if os.path.exists('model_dir/gym_actor_10000.ckpt'):
    agent.restore_actor('model_dir/gym_actor_10000.ckpt')
    agent.restore_critic('model_dir/gym_critic_10000.ckpt')
    print('model loaded')

test_flag = 0
total_steps = 0
while total_steps < 10:
    total_steps += 1
    
    evaluate_reward = evaluate(env, agent)
    print('Steps {}, Evaluate reward: {}'.format(
        total_steps, evaluate_reward))


print('Done')
env.close()