import gym



# Create the environment
env = gym.make("ALE/SpaceInvaders-v5")

# Initialize the environment
initial_state = env.reset()

# Interact with the environment
done = False
while not done:
    action = env.action_space.sample()  # Random action
    state, reward, done, *_ = env.step(action)
    # Add your processing logic here

# Close the environment
env.close()
