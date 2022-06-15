# Reinforcement Learning Educational Project: Cart Pole

Ths is an educational project consisting in applying Reinforcement Learning to OpenAI Gym's [Cart-Pole environment](https://gym.openai.com/envs/CartPole-v1/).

To simulate a naïve controller that decides what action to apply based only on the current angular position, run one of the following

```shell
make simulate_naive
```

```shell
python simulate_naive_controller.py
```

To simulate a receding horizon, model-based, predictive controller, run one of the following:

```shell
make simulate_predictive
```

```shell
python simulate_predictive_controller.py
```

To train an Reinforcement Learning agent, run one of the following:

```shell
make train algorithm=<ALGORITHM>
```

```shell
python cartpole_agents.py <ALGORITHM>
```

Where `<ALGORITHM>` is one of:

- `rfb_linear` (linear function approximation Q Learning with features based on radial basis functions)
- `sarsa` (average reward SARSA)
- `q_learning`
- `avg_reward_expected_sarsa`
- `discounted_expected_sarsa`

To train an Reinforcement Learning agent that was already trained, run one of the following:

```shell
make simulate algorithm=<ALGORITHM>
```

```shell
python simulate_rl_agent.py <ALGORITHM>
```
