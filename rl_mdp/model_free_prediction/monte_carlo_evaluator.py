from collections import defaultdict
from typing import List, Tuple
import numpy as np
from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.model_free_prediction.abstract_evaluator import AbstractEvaluator
from rl_mdp.policy.abstract_policy import AbstractPolicy


class MCEvaluator(AbstractEvaluator):
    def __init__(self, env: AbstractMDP):
        """
        Initializes the Monte Carlo Evaluator.

        :param env: An environment object.
        """
        self.env = env
        self.value_fun = np.zeros(self.env.num_states)    # Estimate of state-value function.
        self.returns = defaultdict(list)  # Stores returns for each state

    def evaluate(self, policy: AbstractPolicy, num_episodes: int) -> np.ndarray:
        """
        Perform the Monte Carlo prediction algorithm.

        :param policy: A policy object that provides action probabilities for each state.
        :param num_episodes: Number of episodes to run for estimating V(s).
        :return: The state-value function V(s) for the associated policy.
        """
        self.value_fun.fill(0)  # Reset value function.
        self.returns.clear()

        for _ in range(num_episodes):
            episode = self._generate_episode(policy)
            self._update_value_function(episode)
        return self.value_fun.copy()

    def _generate_episode(self, policy: AbstractPolicy) -> List[Tuple[int, int, float]]:
        """
        Generate an episode following the policy.

        :return: A list of (state, action, reward) tuples representing the episode.
        """
        episode = []
        state = self.env.reset()
        done = False

        while not done:
            action = policy.sample_action(state)
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state

        return episode

    def _update_value_function(self, episode: List[Tuple[int, int, float]]) -> None:
        """
        Update the value function using the Monte Carlo method.

        :param episode: A list of (state, action, reward) tuples.
        """
        # episode[n][0] = state, [1] = action, [2] = reward
        returns = [0 for step in episode] 
        states = [None for step in episode]
        returns[-1] = episode[-1][2]
        states[-1] = episode[-1][0]

        for step in range(len(episode) - 2, -1, -1):
            returns[step] = (self.env.discount_factor * returns[step + 1] + episode[step][2]
            states[step] = episode[step][0]

        for state in set(states):
            index = states.index(state)
            self.returns[state].append(returns[index])
            self.value_fun[state] = np.mean(self.returns[state])
