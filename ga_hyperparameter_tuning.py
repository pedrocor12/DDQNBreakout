# Import necessary libraries
"""Fully Original code:
Author: Pedro Pinto
Univeristy of Greenwich student"""

from deap import base, creator, tools, algorithms
import random
import numpy as np
from agent import Agent
from breakout import DQNBreakout
from model import AtariNet
import os
import matplotlib.pyplot as plt

# Function to ensure offspring are within specified bounds
def checkBounds(min_values, max_values):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            # Clamp each child's values within the min and max bounds
            for child in offspring:
                for i, val in enumerate(child):
                    child[i] = min(max(val, min_values[i]), max_values[i])
            return offspring
        return wrapper
    return decorator

# Evaluate an individual's fitness by running the agent in the environment
def evalAgent(individual):
    learning_rate, epsilon, batch_size = individual  # Unpack the individual's attributes

    # Calculate epsilon decay based on the given epsilon
    min_epsilon = 0.1  # Fixed minimum epsilon
    nb_warmup = 10000  # Fixed number of warmup steps
    epsilon_decay = 1 - (((epsilon - min_epsilon) / nb_warmup) * 2)

    # Initialize the environment and the agent with the model
    environment = DQNBreakout()
    model = AtariNet(nb_actions=4)
    agent = Agent(model=model, learning_rate=learning_rate, epsilon=epsilon,
                  epsilon_decay=epsilon_decay, batch_size=int(batch_size))

    # Run the agent for a fixed number of episodes and calculate average reward
    total_reward = 0
    num_episodes = 100  # Number of episodes for testing
    for _ in range(num_episodes):
        state = environment.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = environment.step(action)
            state = next_state
            total_reward += reward
    average_reward = total_reward / num_episodes
    return average_reward,

# Define bounds for the hyperparameters
LR_MIN, LR_MAX = 0.0001, 0.01
EPSILON_DECAY_MIN, EPSILON_DECAY_MAX = 0.1, 1.0
BATCH_SIZE_MIN, BATCH_SIZE_MAX = 32, 128

# Setup DEAP framework for genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Register functions to randomly initialize hyperparameters
toolbox.register("attr_lr", random.uniform, LR_MIN, LR_MAX)
toolbox.register("attr_epsilon_decay", random.uniform, EPSILON_DECAY_MIN, EPSILON_DECAY_MAX)
toolbox.register("attr_batch_size", random.randint, BATCH_SIZE_MIN, BATCH_SIZE_MAX)

# Create individual and population
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_lr, toolbox.attr_epsilon_decay, toolbox.attr_batch_size), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Apply checkBounds decorator to genetic operators
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.decorate("mate", checkBounds([LR_MIN, EPSILON_DECAY_MIN, BATCH_SIZE_MIN],
                                     [LR_MAX, EPSILON_DECAY_MAX, BATCH_SIZE_MAX]))
toolbox.decorate("mutate", checkBounds([LR_MIN, EPSILON_DECAY_MIN, BATCH_SIZE_MIN],
                                       [LR_MAX, EPSILON_DECAY_MAX, BATCH_SIZE_MAX]))

# Register selection and evaluation functions
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evalAgent)

# Main function to run the genetic algorithm
def main():
    population = toolbox.population(n=50)  # Create a population of 50 individuals
    ngen = 1000  # Number of generations

    # Run the genetic algorithm
    for gen in range(ngen):
        print(f"Generation {gen + 1} / {ngen}...")
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    # After all generations, select the best individual
    best_ind = tools.selBest(population, k=1)[0]
    print("Best Hyperparameters:", best_ind)

# Run the main function if this script
if __name__ == "__main__":
    main()



