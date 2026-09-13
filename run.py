import os
import time
import random; random.seed(0)
import numpy as np; np.random.seed(0)
import math

import matplotlib.pyplot as plt
from PIL import Image

import ai2thor
import ai2thor_colab
from ai2thor_colab import plot_frames
from ai2thor.controller import Controller

from ai2thor.platform import CloudRendering
controller = Controller()

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.distributions import Categorical

import clip

# Set random seeds for maintaining reproducibility
os.environ['PYTHONHASHSEED'] = str(0)
torch.backends.cudnn.deterministic = True

from ppo import PPOAgent

from train import (
    process_inputs, random_policy, train_ppo, manual_control_policy
)

import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')

floor_index = random.randint(0, 30)
floor_index = 11

controller = Controller(
    agentMode="arm",
    massThreshold=None,
    visibilityDistance=0.50,
    scene=f"FloorPlan{floor_index}",

    # step sizes
    gridSize=0.25,
    
    # image modalities
    renderInstanceSegmentation=False,
    renderDepthImage=False,
    renderSemanticSegmentation=False,
    renderNormalsImage=False,
    
    # camera properties
    width = 600,
    height = 420,
    fieldOfView = 120,
    
    # set seed for reproducibility
    seed=90,
)

# plot_frames(controller.last_event)
controller.reset(f"FloorPlan{floor_index}");

is_cuda = torch.cuda.is_available()

if is_cuda: device = torch.device('cuda')
else: device = torch.device('cpu')

STATE_DIM = SCREEN_WIDTH = SCREEN_HEIGHT = 224 # Resnet's input size

action_space = ["MoveAgent", "RotateAgent", "MoveArm", "MoveArmBase", "LookUp", "LookDown"]

# instruction = "Pick up the red tomato on the table"
instruction = "tomato"

# Not implemented yet as the policy
ppo_agent = PPOAgent(state_dim=STATE_DIM, action_dim = len(action_space))
# state_dim = 224*224*3
manual_control_policy(controller, action_space, instruction)
# random_policy(controller, action_space, instruction, num_steps=50)
# train_ppo(controller, ppo_agent, action_space, state_dim, num_episodes=10, max_timesteps=200)

controller.stop()