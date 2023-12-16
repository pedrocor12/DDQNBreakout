# """
# no changes made, author of the code:
# @misc{youtube_dzskvsszgjs,
#     author = "{Robert Cowher - DevOps, Python, AI}",
#     title = "{Playing Breakout with Deep Reinforcement Learning}",
#     year = {2023},
#     howpublished = "\url{https://www.youtube.com/watch?v=DzSKVsSzGjs}"

import os
from breakout import *
from model import AtariNet
from agent import Agent

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# render_mode = 'human' to see the env training
environment = DQNBreakout(device=device)

model = AtariNet(nb_actions=4)

model.to(device)

model.load_the_model()

agent = Agent(model=model,
              device=device,
              epsilon=1.0,
              nb_warmup=5000,
              nb_actions=4,
              learning_rate=0.007133149062266376,
              memory_capacity=1000000,
              batch_size=256) #should be 64

agent.train(env=environment, epochs=200000)
