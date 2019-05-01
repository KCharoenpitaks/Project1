from Agent import Agent, QNetwork, Agent_subgoal, ActorCritic
from Monitor import interact
import gym
import numpy as np
from taxi import TaxiEnv
import tensorflow as tf
from memory import Memory
from frozen_lake import FrozenLakeEnv



env = FrozenLakeEnv()
#env = TaxiEnv()


agent1 = Agent()
agent2 = Agent()
agent3 = Agent()
agent4 = Agent()
#agent = Agent_subgoal()
#agent = QNetwork()
#agent = ActorCritic()

avg_rewards, best_avg_reward = interact(env, agent1,agent2,agent3,agent4, num_episodes=100000)