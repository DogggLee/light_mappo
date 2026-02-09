
import socket
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])

from envs.uav_pursuit_env import MultiUavPursuitEnv


