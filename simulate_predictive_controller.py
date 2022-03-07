from time import sleep
from itertools import product
from functools import partial
from typing import Callable, Iterable

import gym
import numpy as np


PREDICTION_HORIZON = 4

env = gym.make('CartPole-v1')


class Predictor:
    def __init__(self) -> None:
        # Parameters taken from
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py:
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

    def predict_step(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Predict the state one step ahead after applying an action. Adapated
        from
        https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        """
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (
                4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass
            )
        )
        xacc = (
            temp - self.polemass_length * thetaacc * costheta / self.total_mass
        )

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        return np.array([x, x_dot, theta, theta_dot])

    def predict(
        self, initial_state: np.ndarray, action_traj: Iterable
    ) -> np.ndarray:
        """
        Predicts the state trajectory when applying the control action
        trajectory `action_traj`.
        """
        state_traj = [initial_state]
        for action in action_traj:
            state_traj.append(self.predict_step(state_traj[-1], action))

        return np.array(state_traj)

    def find_error(self, state_traj: np.ndarray):
        return np.sum(state_traj[1:]**2)

    def predict_error(
        self, initial_state: np.ndarray, action_traj: Iterable
    ) -> float:
        return self.find_error(self.predict(initial_state, action_traj))

    def error_predictor_function(self, initial_state: np.ndarray) -> Callable:
        return partial(self.predict_error, initial_state)


predictor = Predictor()

for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        sleep(0.2)
        env.render(mode='human')
        print(observation)

        # Choose the first action of the best action trajectory:
        action = min(
            product(range(env.action_space.n), repeat=PREDICTION_HORIZON),
            key=predictor.error_predictor_function(observation),
        )[0]

        print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
