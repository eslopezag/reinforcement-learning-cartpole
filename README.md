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
