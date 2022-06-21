import sys

from PIL import Image
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
    frames = []
    observation = env.reset()
    done = False
    terminate = False
    for t in range(600):
        render = env.render(mode='rgb_array')
        image = Image.fromarray(render)
        # image.save(f'gifs/{agent_name}_img_{i:0>3}.png')
        frames.append(image)

        if not done:
            action = agent.get_action(observation)
            observation, reward, done, info = env.step(action)
        else:
            terminate = True

        if terminate:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

frames[0].save(
    f'gifs/{agent_name}.gif',
    format='GIF',
    append_images=frames[1:],
    save_all=True,
    duration=50,  # Each frame will be shown for 200 milliseconds
    #  Not specifying the `loop` parameter means the GIF will only play once
)
