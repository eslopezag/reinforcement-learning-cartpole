import numpy as np
import tensorflow as tf
import gym

from rl_agents.q_approximators import (
    GradQApproximator, AvgRewardGradQApproximator
)
from rl_agents.policies import EpsGreedyPolicy
from rl_agents.schedulers import cosine_decay_scheduler
from rl_agents.agents import (
    SemiGradQLearningAgent,
    AvgRewardSemiGradSarsaAgent,
    AvgRewardSemiGradExpectedSarsaAgent,
    SemiGradExpectedSarsaAgent
)
from custom_agent_history import CustomAgentHistory


def get_cartpole_agent(
    name: str,
    training_steps: int = 100000,
):
    """
    Returns the specified agent with the given characteristics.
    """
    env = gym.make('CartPole-v1')

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, input_shape=(4,), activation='elu'),
        tf.keras.layers.Dense(64, activation='elu'),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dense(2),
    ])

    def is_terminal(state):
        return (np.abs(state[0]) > 2.4 or np.abs(state[2]) > 12 * np.pi / 180)

    if name in ('q_learning', 'discounted_expected_sarsa'):
        Q = GradQApproximator(
            model=model,
            optimizer=tf.keras.optimizers.Adam(0.0005),
            target_delay_steps=128,
            batch_size=2048,
            state_dim=4,
            buffer_size=16384,
            batches_per_step=1,
            start_training_buffer_size=1,
            scheduler=cosine_decay_scheduler(
                0.0005, 0.00005, 3 * training_steps // 4
            ),
            is_terminal_fn=is_terminal,
            env=env,
        )

    elif name in ('sarsa', 'avg_reward_expected_sarsa'):
        Q = AvgRewardGradQApproximator(
            model=model,
            optimizer=tf.keras.optimizers.Adam(0.001),
            target_delay_steps=128,
            batch_size=2048,
            state_dim=4,
            avg_reward_step_size=0.1,
            initial_avg_reward=1.,
            buffer_size=16384,
            batches_per_step=1,
            start_training_buffer_size=1,
            step_size_scheduler=cosine_decay_scheduler(
                0.001, 0.00001, 3 * training_steps // 4
            ),
            is_terminal_fn=is_terminal,
            env=env,
        )

    if name == 'sarsa':
        agent = AvgRewardSemiGradSarsaAgent(
            env=env,
            Q=Q,
            history=CustomAgentHistory(),
            target_policy=EpsGreedyPolicy(
                scheduler=cosine_decay_scheduler(0.3, 0.05, training_steps),
            ),
        )

    elif name == 'q_learning':
        agent = SemiGradQLearningAgent(
            env=env,
            Q=Q,
            history=CustomAgentHistory(),
            exploration_policy=EpsGreedyPolicy(
                scheduler=cosine_decay_scheduler(
                    0.5, 0., 3 * training_steps // 4
                )
            ),
            discount=0.99,
        )

    elif name == 'avg_reward_expected_sarsa':
        agent = AvgRewardSemiGradExpectedSarsaAgent(
            env=env,
            Q=Q,
            history=CustomAgentHistory(),
            target_policy=EpsGreedyPolicy(
                scheduler=cosine_decay_scheduler(0.3, 0.05, training_steps),
            ),
        )

    elif name == 'discounted_expected_sarsa':
        agent = SemiGradExpectedSarsaAgent(
            env=env,
            Q=Q,
            history=CustomAgentHistory(),
            target_policy=EpsGreedyPolicy(
                scheduler=cosine_decay_scheduler(
                    0.3, 1e-5, 3 * training_steps // 4
                ),
            ),
            exploration_policy=EpsGreedyPolicy(
                scheduler=cosine_decay_scheduler(
                    0.5, 0., 3 * training_steps // 4
                )
            ),
            discount=0.99,
        )

    else:
        raise NotImplementedError(
            'The specified agent is not among the ones that have been defined.'
        )

    return agent


if __name__ == '__main__':
    import sys

    name, training_steps = sys.argv[1:]
    training_steps = int(training_steps)

    agent = get_cartpole_agent(name, training_steps)

    agent.train(training_steps)
    agent.save(f'saved_agents/{name}')
    agent.show_training_results(10)
