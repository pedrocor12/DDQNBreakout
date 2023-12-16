# """
# Some changes were made to the code but the majority of the implementation is from:
# @misc{youtube_dzskvsszgjs,
#     author = "{Robert Cowher - DevOps, Python, AI}",
#     title = "{Playing Breakout with Deep Reinforcement Learning}",
#     year = {2023},
#     howpublished = "\url{https://www.youtube.com/watch?v=DzSKVsSzGjs}",

import os
import torch
import torch.nn as nn

# Define a neural network class for Atari games
class AtariNet(nn.Module):
    def __init__(self, nb_actions=4):
        super(AtariNet, self).__init__()

        # Define ReLU activation function
        self.relu = nn.ReLU()

        # Define convolutional layers with specified kernel sizes and strides
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        # Define a layer to flatten the output of the convolutions
        self.flatten = nn.Flatten()

        # Define dropout layer for regularization
        self.dropout = nn.Dropout(p=0.2)

        # Size of flattened layer's output
        flattened_size = 3136

        # Define linear layers for action value estimation
        self.action_value1 = nn.Linear(flattened_size, 1024)
        self.action_value2 = nn.Linear(1024, 1024)
        self.action_value3 = nn.Linear(1024, nb_actions)

        # Define linear layers for state value estimation
        self.state_value1 = nn.Linear(flattened_size, 1024)
        self.state_value2 = nn.Linear(1024, 1024)
        self.state_value3 = nn.Linear(1024, 1)

    # Define the forward pass of the network
    def forward(self, x):
        # Apply the convolutional and ReLU layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Flatten the output for the linear layers
        x = self.flatten(x)

        # Calculate the state value
        state_value = self.relu(self.state_value1(x))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value3(state_value))

        # Calculate the action value
        action_value = self.relu(self.action_value1(x))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.dropout(action_value)
        action_value = self.action_value3(action_value)

        # Combine state and action values to get the final output
        output = state_value + (action_value - action_value.mean())
        return output

    # Function to save the model weights
    def save_the_model(self, weights_filename='models/latest.pt'):
        # Create a directory for models if it doesn't exist
        if not os.path.exists("models"):
            os.makedirs("models")
        # Save the model state
        torch.save(self.state_dict(), weights_filename)

    # Function to load the model weights
    def load_the_model(self, weights_filename='models/latest.pt'):
        try:
            # Load the model state
            self.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights from {weights_filename}")
        except:
            print(f"No weights file available at {weights_filename}")



