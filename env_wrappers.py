import numpy as np
import gym


class RBFObservations(gym.ObservationWrapper):
    """
    Observation transformation wrapper for the Cartpole environment that adds
    features extracted from radial basis functions with centers lying on a
    rectangular grid to the original environment's state.
    """
    proposed_min_values = np.array([-2.5, -4, -0.24, -2])
    proposed_max_values = np.array([2.5, 4, 0.24, 2])
    centers_per_dimension = np.array([7, 7, 9, 9])
    size = np.prod(centers_per_dimension) + 4

    linspaces = []
    for centers in centers_per_dimension:
        linspaces.append(np.linspace(0, 1, centers))

    grid = np.meshgrid(*linspaces, indexing='ij')

    scaling_factor = 32.

    def __init__(self, env):
        super().__init__(env)
        self._observation_space = gym.spaces.Box(
            shape=(self.size,), low=0, high=1)

    def _normalize(self, obs):
        """
        Normalizes all observations to lie between 0 and 1.
        """
        # Scale the observations:
        scaled_obs = (
            (obs - self.proposed_min_values) /
            (self.proposed_max_values - self.proposed_min_values)
        )

        # Clip the scaled observations:
        clipped_scaled_obs = np.clip(scaled_obs, 0, 1)

        return clipped_scaled_obs

    def observation(self, obs):
        state = np.empty((self.size,))

        state[:4] = obs

        state[4:] = np.power(
            2,
            -self.scaling_factor * sum(
                np.square(o - g)
                for o, g in zip(self._normalize(obs), self.grid)
            )
        ).flatten()

        return state
