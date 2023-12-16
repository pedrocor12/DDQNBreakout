"""
No changes made the code from this sections belongs to:
@misc{youtube_dzskvsszgjs,
    author = "{Robert Cowher - DevOps, Python, AI}",
    title = "{Playing Breakout with Deep Reinforcement Learning}",
    year = {2023},
    howpublished = "\url{https://www.youtube.com/watch?v=DzSKVsSzGjs}",
}
"""
import os
from breakout import *
from model import AtariNet
from agent import Agent

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# To check if it's training
print(device)

# render_mode = 'human' to see the env training
environment = DQNBreakout(device=device, render_mode="human")



model = AtariNet(nb_actions=4)

model.to(device)

model.load_the_model()

agent = Agent(model=model,
              device=device,
              epsilon=0.5,
              nb_warmup=5000,
              nb_actions=4,
              learning_rate=0.007133149062266376,
              memory_capacity=1000000,
              batch_size=54)

agent.test(env=environment)
