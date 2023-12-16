# """
# Some changes were made to the code but the majority of the implementation is from:
# @misc{youtube_dzskvsszgjs,
#     author = "{Robert Cowher - DevOps, Python, AI}",
#     title = "{Playing Breakout with Deep Reinforcement Learning}",
#     year = {2023},
#     howpublished = "\url{https://www.youtube.com/watch?v=DzSKVsSzGjs}",
# }
# """
import copy
import random
import torch
from torch import optim
import torch.nn.functional as F
import time
from plot import LivePlot
import numpy as np



class ReplayMemory:

    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device
        self.memory_max_report = 0

    def insert(self, transition):
        # Apply .to('cpu') only to PyTorch tensors
        transition = [item.to('cpu') if isinstance(item, torch.Tensor) else item for item in transition]

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory.pop(0)  # More efficient than remove(self.memory[0])
            self.memory.append(transition)

    def sample(self, batch_size=32):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)

        sampled_data = []
        for items in batch:
            # If the first item is a tensor, concatenate all items
            if isinstance(items[0], torch.Tensor):
                sampled_data.append(torch.cat(items).to(self.device))
            else:
                # Directly convert the list of non-tensor items into a tensor
                sampled_data.append(torch.tensor(items, device=self.device))

        return sampled_data

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10

    def __len__(self):
        return len(self.memory)


class Agent:
    #To run the genetic algorithm nb=actions=4 because it needs to take the possible actions, and to run main nb_actions = None as main has a place to set the parameter
    #Additionally to run the algorithm epsilong_decay=None needs to be added to the hyperparameters
    def __init__(self, model, device='cpu', epsilon=1.0, min_epsilon=0.1,epsilon_decay=None, nb_warmup=10000, nb_actions=4,
                 memory_capacity=10000,
                 batch_size=32, learning_rate=0.00025):

        self.memory = ReplayMemory(device=device, capacity=memory_capacity)
        self.model = model
        self.target_model = copy.deepcopy(model).eval()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        #ADDED  put back to normal line after if
        if epsilon_decay is None:
            self.epsilon_decay = 1 - (((epsilon - min_epsilon) / nb_warmup) * 2)
        else:
            self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.model.to(device)
        self.target_model.to(device)
        self.gamma = 0.99
        self.nb_actions = nb_actions

        # Check if this is workign later
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        # To see the decay process if needed
        # print(f"Starting Epsilon: {self.epsilon}")
        # print(f"Epsilon decay: {self.epsilon_decay}")

    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.nb_actions, (1, 1,))
        else:
            av = self.model(state).detach()
            return torch.argmax(av, dim=1, keepdim=True)

    def train(self, env, epochs):

        stats = {'Returns': [], 'AvgReturns': [], 'EpsilonCheckpoint': []}

        plotter = LivePlot()

        for epoch in range(1, epochs + 1):
            state = env.reset()
            done = False
            ep_return = 0

            while not done:
                action = self.get_action(state)

                next_state, reward, done, info = env.step(action)

                self.memory.insert([state, action, reward, done, next_state])

                if self.memory.can_sample(self.batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = self.memory.sample(self.batch_size)
                    qsa_b = self.model(state_b).gather(1, action_b)
                    next_qsa_b = self.target_model(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]
                    # reward_b = reward_b.unsqueeze(1)  # Reshape reward_b to [64, 1]
                    target_b = reward_b + ~done_b * self.gamma * next_qsa_b
                    loss = F.mse_loss(qsa_b, target_b)
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                state = next_state
                ep_return += reward


            stats['Returns'].append(ep_return)

            if self.epsilon > self.min_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay

            if epoch % 10 == 0:

                self.model.save_the_model()
                print(" ")

                average_returns = np.mean(stats['Returns'][-100:])

                stats['AvgReturns'].append(average_returns)
                stats['EpsilonCheckpoint'].append(self.epsilon)

                # print(f"Epoch: {epoch}")
                # print(f"Average Return (last 10 epochs): {average_returns}")
                # print(f"Epsilon: {self.epsilon}")
                # print(f"Total Average Returns: {stats['AvgReturns']}")
                # print(f"Total Epsilon Checkpoints: {stats['EpsilonCheckpoint']}")

                if (len(stats['Returns'])) > 100:
                    print(
                        F"Epoch: {epoch} - Average Return: {np.mean(stats['Returns'][-100:])} - Epsilon: {self.epsilon}")
                else:
                    print(
                        F"Epoch: {epoch} - Episode Return: {np.mean(stats['Returns'][-1:])} - Epsilon: {self.epsilon}")

            # Should be 100
            if epoch % 100 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                plotter.update_plot(stats)

            if epoch % 1000 == 0:
                self.model.save_the_model(f"models/model_iter_{epoch}.pt")







        return stats

    def test(self, env):

        for epoch in range(1, 3):
            state = env.reset()

            done = False

            for _ in range(1000):
                time.sleep(0.01)
                action = self.get_action(state)
                state, reward, done, info = env.step(action)
                if done:
                    break

