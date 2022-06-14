train:
	python cartpole_agents.py $(algorithm)
simulate:
	python simulate_rl_agent.py $(algorithm)
simulate_naive:
	python simulate_naive_controller.py
simulate_predictive:
	python simulate_predictive_controller.py
