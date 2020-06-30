import logging
import numpy as np


class CustomLogger(object):
    def __init__(self, logfile, level=logging.DEBUG):
        self.logger = logging.getLogger()
        self.logger.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.ch1 = logging.FileHandler(logfile, mode='a')
        self.ch1.setLevel(level)
        self.ch1.setFormatter(formatter)
        self.logger.addHandler(self.ch1)

    def info(self, string_msg):
        self.logger.info(string_msg)


def distance_cal(obs):
    '''calculate distance between end effector and target'''
    ee_pose = np.array(obs[22:25])
    target_pose = np.array(obs[-3:])

    distance = np.sqrt((target_pose[0]-ee_pose[0])**2 +
                       (target_pose[1]-ee_pose[1])**2+(target_pose[2]-ee_pose[2])**2)

    # reward = np.tanh(1/(distance+1e-5))
    reward = np.exp(-1*distance)

    return distance, reward
