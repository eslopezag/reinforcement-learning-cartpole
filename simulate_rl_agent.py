from time import sleep
import sys

import gym

from rl_agents.agents import load_agent
from env_wrappers import RBFObservations

agent_name = sys.argv[1]
agent_folder = f'./saved_agents/{agent_name}'
agent = load_agent(agent_folder)
agent.set_mode('inference')

env = gym.make('CartPole-v1')

if agent_name == 'rbf_linear':
    env = RBFObservations(env)

for i_episode in range(1):
    observation = env.reset()
    for t in range(600):
        sleep(0.2)
        env.render(mode='human')
        print(observation)

        # Choose the first action of the best action trajectory:
        action = agent.get_action(observation)

        print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
