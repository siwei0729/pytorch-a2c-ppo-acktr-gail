import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

sys.path.append('a2c_ppo_acktr')
seed = 1
log_interval = 10
env_name = "PongNoFrameskip-v4"
load_dir = "./trained_models/ppo/"
det = True

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = torch.load(os.path.join(load_dir, env_name + ".pt"))

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)


def expert_forward(obs):
    global recurrent_hidden_states
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = \
            actor_critic.act(obs, recurrent_hidden_states, masks, deterministic=det)

    return action

