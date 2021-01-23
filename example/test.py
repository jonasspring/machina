"""
An example of QT-Opt.
"""

import argparse
import copy
import json
import os
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import gym.wrappers

import machina as mc
from machina.pols import ArgmaxQfPol
from machina.noise import OUActionNoise
from machina.algos import qtopt
from machina.vfuncs import DeterministicSAVfunc, CEMDeterministicSAVfunc
from machina.envs import GymEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import set_device, measure

from simple_net import QNet

import panda_gym_env

import rospy
import time
from multiprocessing import Pool
import multiprocessing as mp
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage',
                    help='Directory name of log.')
parser.add_argument('--env_name', type=str,
                    default='panda_gym_env-v0', help='Name of environment.')
parser.add_argument('--record', action='store_true',
                    default=False, help='If True, movie is saved.')
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_epis', type=int,
                    default=1000, help='Number of episodes to run.')
parser.add_argument('--max_steps_off', type=int,
                    default=1000000000000, help='Number of episodes stored in off traj.')
parser.add_argument('--num_parallel', type=int, default=1,
                    help='Number of processes to sample.')
parser.add_argument('--cuda', type=int, default=-1, help='cuda device number.')

parser.add_argument('--max_steps_per_iter', type=int, default=40,
                    help='Number of steps to use in an iteration.')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--pol_lr', type=float, default=1e-1,
                    help='Policy learning rate.')
parser.add_argument('--qf_lr', type=float, default=5e-1,
                    help='Q function learning rate.')
parser.add_argument('--h1', type=int, default=100,
                    help='hidden size of layer1.')
parser.add_argument('--h2', type=int, default=100,
                    help='hidden size of layer2.')
parser.add_argument('--tau', type=float, default=0.001,
                    help='Coefficient of target function.')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='Discount factor.')

parser.add_argument('--lag', type=int, default=100,
                    help='Lag of gradient steps of target function2.')
parser.add_argument('--num_iter', type=int, default=2,
                    help='Number of iteration of CEM.')
parser.add_argument('--num_sampling', type=int, default=60,
                    help='Number of samples sampled from Gaussian in CEM.')
parser.add_argument('--num_best_sampling', type=int, default=6,
                    help='Number of best samples used for fitting Gaussian in CEM.')
parser.add_argument('--multivari', action='store_true',
                    help='If true, Gaussian with diagonal covarince instead of Multivariate Gaussian matrix is used in CEM.')
parser.add_argument('--eps', type=float, default=0.3,
                    help='Probability of random action in epsilon-greedy policy.')
parser.add_argument('--loss_type', type=str,
                    choices=['mse', 'bce'], default='mse',
                    help='Choice for type of belleman loss.')
parser.add_argument('--save_memory', action='store_true',
                    help='If true, save memory while need more computation time by for-sentence.')
args = parser.parse_args()


def f(port):
   os.environ['ROS_MASTER_URI'] = "http://localhost:" + str(port) + '/'
   print(os.environ['ROS_MASTER_URI'])
   rospy.init_node("test", anonymous=True)
   #env1 = GymEnv(args.env_name, log_dir=os.path.join(args.log, 'movie'), record_video=args.record)
   #env1.env.seed(args.seed)
   while not rospy.is_shutdown():
        time.sleep(0.2)

env1 = GymEnv(args.env_name, log_dir=os.path.join(args.log, 'movie'), record_video=args.record)
env1.env.seed(args.seed)
env2 = GymEnv(args.env_name, log_dir=os.path.join(args.log, 'movie'), record_video=args.record)
env2.env.seed(args.seed)

jobs = []
#with Pool(2) as p:
     #envs = p.map(f, [env1.env.port, env2.env.port])
     #print(envs)
ports = [env1.env.port, env2.env.port]
for i in range(2):
  p = mp.Process(target=f, args=(ports[i],))
  jobs.append(copy.deepcopy(p))
  p.start()
p1 = mp.Process(target=f, args=(ports[i],))
jobs.append(copy.deepcopy(p))
  p.start()
env1.env.init()
env2.env.init()
time.sleep(3.0)
os.environ['ROS_MASTER_URI'] = "http://localhost:" + str(env1.env.port) + '/'
env1.reset()
os.environ['ROS_MASTER_URI'] = "http://localhost:" + str(env2.env.port) + '/'
env2.reset()
#time.sleep(10.0)

while not rospy.is_shutdown():
  time.sleep(0.1)
