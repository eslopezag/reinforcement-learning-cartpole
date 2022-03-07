import gym
from time import sleep

env = gym.make('CartPole-v1')
for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        sleep(0.2)
        env.render(mode='human')
        print(observation)
        action = 1 if observation[2] >= 0 else 0
        print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
