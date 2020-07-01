# -*- coding: utf-8 -*-

import os
import numpy as np
import parl
from parl import layers
# from parl.utils import logger
import argparse
from parl.utils import action_mapping  # 将神经网络输出映射到对应的 实际动作取值范围内
from parl.utils import ReplayMemory  # 经验回放

from td3_model import RLBenchModel
from td3_agent import RLBenchAgent
from parl.algorithms import TD3

import gym
import rlbench.gym
from utils import CustomLogger, distance_cal

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


def run_train_episode(env, agent, rpm):
    obs = env.reset()
    total_reward = 0
    # Use joint positions only
    target_pose = np.expand_dims(obs[-3:], axis=0)
    max_action = float(env.action_space.high[0])

    for steps in range(EPISODE_LENGTH):

        batch_obs = np.expand_dims(obs[8:15], axis=0)
        batch_obs_full = np.concatenate((batch_obs, target_pose), axis=1)

        if rpm.size() < WARMUP_SIZE:
            action = env.action_space.sample()
            action[-1] = 0 # set gripper state as close

        else:
            action = agent.predict(batch_obs_full.astype('float32'))
            # Add gripper action here, 0: close, 1: open
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

        # Feed first 7 action into rpm here, gripper state is dropped here
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


# evaluate agent, calculate reward mean of 5 episodes
def evaluate_episode(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward = 0
        target_pose = np.expand_dims(obs[-3:], axis=0)

        for i in range(EPISODE_LENGTH):
            batch_obs = np.expand_dims(obs[8:15], axis=0)
            batch_obs_full = np.concatenate((batch_obs, target_pose), axis=1)
            action = agent.predict(batch_obs_full.astype('float32'))

            # Add gripper action again
            action = np.append(action, 0)

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
        eval_reward.append(total_reward)
    return np.mean(eval_reward)


def test_episode(env, agent, render=False):
    obs = env.reset()
    total_reward = 0
    target_pose = np.expand_dims(obs[-3:], axis=0)

    for i in range(EPISODE_LENGTH):
        batch_obs = np.expand_dims(obs[8:15], axis=0)
        batch_obs_full = np.concatenate((batch_obs, target_pose), axis=1)
        action = agent.predict(batch_obs_full.astype('float32'))

        # Add gripper action again
        action = np.append(action, 0)

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


def main(args):

    # Create rlbench gym env
    env = gym.make('reach_target-state-v0', render_mode=args.mode)

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

    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, act_dim)
    # load model
    if os.path.exists('model_dir/gym_actor_steps_20000.ckpt'):
        agent.restore_actor('model_dir/gym_actor_steps_20000.ckpt')
        agent.restore_critic('model_dir/gym_critic_steps_20000.ckpt')
        print('model loaded')

    test_flag = 0
    total_steps = 0
    if args.train:
        logger = CustomLogger('train_log/train_gym.txt')
        while total_steps < MAX_EPISODES:
            train_reward, distance = run_train_episode(env, agent, rpm)
            total_steps += 1
            logger.info('Steps: {}, Distance: {:.4f}, Reward: {}'.format(
                total_steps, distance, train_reward))

            if total_steps // TEST_EVERY_STEPS >= test_flag:
                while total_steps // TEST_EVERY_STEPS >= test_flag:
                    test_flag += 1

                evaluate_reward = evaluate_episode(env, agent)
                logger.info('Steps {}, Evaluate reward: {}'.format(
                    total_steps, evaluate_reward))

                # 保存模型
                actor_ckpt = 'model_dir/gym_actor_steps_{}.ckpt'.format(
                    total_steps)
                critic_ckpt = 'model_dir/gym_critic_steps_{}.ckpt'.format(
                    total_steps)
                agent.save_actor(actor_ckpt)
                agent.save_critic(critic_ckpt)

    if args.test:
        for i in range(10):
            test_reward = test_episode(env, agent)
            print('Steps {}, Test reward: {}'.format(
                i, test_reward))

    print('Done')
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test mode.')
    parser.add_argument('--train', dest='train',
                        action='store_true', default=False)
    parser.add_argument('--test', dest='test',
                        action='store_true', default=False)
    parser.add_argument('--mode', help='render mode name', default='None')
    args = parser.parse_args()

    main(args)
