# Some changes made to the file to accomodate but the code in this file belongs to:
# @misc{youtube_dzskvsszgjs,
#     author = "{Robert Cowher - DevOps, Python, AI}",
#     title = "{Playing Breakout with Deep Reinforcement Learning}",
#     year = {2023},
#     howpublished = "\url{https://www.youtube.com/watch?v=DzSKVsSzGjs}",
#

import collections
import cv2
import gym
import numpy as np
from PIL import Image
import torch
import ale_py as ale

# Define the DQNBreakout class as a wrapper around the Gym environment
class DQNBreakout(gym.Wrapper):
    def __init__(self, render_mode='rgb_array', repeat=4, device='cpu'):
        # Create the Breakout environment from OpenAI Gym with specified render mode
        env = gym.make('BreakoutNoFrameskip-v4', render_mode=render_mode)

        # Initialize the gym.Wrapper with the created environment
        super(DQNBreakout, self).__init__(env)

        # Define image shape for processing observations and other relevant attributes
        self.image_shape = (84, 84)  # Size of the processed image
        self.repeat = repeat  # Number of times an action is repeated
        self.lives = env.ale.lives()  # Track the number of lives in the game
        self.frame_buffer = []  # Buffer to store frames for processing
        self.device = device  # Device (CPU or GPU) where computations are performed

    # Override the step method for custom behavior
    def step(self, action):
        total_reward = 0  # Initialize total reward for this step
        done = False  # Flag to indicate if the episode is done

        # Repeat the action for a specified number of times
        for i in range(self.repeat):
            observation, reward, done, truncated, info = self.env.step(action)

            # Convert action to a Python int if it's a numpy array or tensor
            if isinstance(action, np.ndarray) or torch.is_tensor(action):
                action = action.item()

            # Accumulate reward
            total_reward += reward

            # Check for life loss and adjust reward accordingly
            current_lives = info['lives']
            if current_lives < self.lives:
                total_reward += total_reward - 1
                self.lives = current_lives

            # Store the current frame
            self.frame_buffer.append(observation)

            # Break the loop if the episode is done
            if done:
                break

        # Process the observation by taking the maximum over the last two frames
        max_frame = np.max(self.frame_buffer[-2:], axis=0)
        max_frame = self.process_observation(max_frame)
        max_frame = max_frame.to(self.device)

        # Convert total_reward and done to tensors and move to the specified device
        total_reward = torch.tensor(total_reward).view(1, -1).float()
        total_reward = total_reward.to(self.device)

        done = torch.tensor(done).view(1, -1)
        done = done.to(self.device)

        return max_frame, total_reward, done, info

    # Override the reset method for custom behavior
    def reset(self):
        # Clear the frame buffer
        self.frame_buffer = []

        # Reset the environment and capture the initial observation
        observation, _ = self.env.reset()

        # Update lives based on the reset environment
        self.lives = self.env.ale.lives()

        # Process the initial observation
        observation = self.process_observation(observation)

        return observation

    # Define a method to process the observation images
    def process_observation(self, observation):
        # Convert the observation to an image, resize, and convert to grayscale
        img = Image.fromarray(observation)
        img = img.resize(self.image_shape)
        img = img.convert('L')
        img = np.array(img)

        # Convert the image to a PyTorch tensor, normalize, and add batch and channel dimensions
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)  # Add channel dimension
        img = img.unsqueeze(0)  # Add batch dimension
        img = img / 255.0       # Normalize pixel values

        # Move the processed image to the specified device
        img = img.to(self.device)

        return img



