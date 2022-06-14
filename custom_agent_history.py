from typing import Tuple
from itertools import chain

import numpy as np
from matplotlib import pyplot as plt

from rl_agents.agent_history import TabularAgentHistory


class CustomAgentHistory(TabularAgentHistory):
    def show_training_results(
        self,
        episode_window: int,
        figsize: Tuple[int] = (10, 12)
    ) -> None:
        fig, ax = plt.subplots(2, figsize=figsize)

        ax[0].set_xlabel('Episode')
        ax[0].set_ylabel(
            'Running average of episode reward (window size = '
            f'{episode_window})'
        )
        episode_rewards = [
            sum(r for r in self.reward_history[i + 1: j + 1] if r is not None)
            for i, j in zip(
                chain([-1], self.episode_ends),
                self.episode_ends,
            )
        ]
        cumsum = np.cumsum(np.insert(episode_rewards, 0, 0))
        running_avg = (
            cumsum[episode_window:] - cumsum[:-episode_window]
        ) / episode_window
        ax[0].plot(
            range(episode_window, len(running_avg) + episode_window),
            running_avg
        )

        ax[1].set_xlabel('Episode')
        ax[1].set_ylabel('Episode length')

        ax[1].plot(np.diff(self.episode_ends))

        plt.show(block=True)
