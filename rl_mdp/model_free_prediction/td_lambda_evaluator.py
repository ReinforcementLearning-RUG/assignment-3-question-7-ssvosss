import numpy as np

from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.model_free_prediction.abstract_evaluator import AbstractEvaluator
from rl_mdp.policy.abstract_policy import AbstractPolicy


class TDLambdaEvaluator(AbstractEvaluator):
    def __init__(self,
                 env: AbstractMDP,
                 alpha: float,
                 lambd: float):
        """
        Initializes the TD(λ) Evaluator.

        :param env: A mdp object.
        :param alpha: The step size.
        :param lambd: The trace decay parameter (λ).
        """
        self.env = env
        self.alpha = alpha
        self.lambd = lambd
        self.value_fun = np.zeros(self.env.num_states)    # Estimate of state-value function.
        self.eligibility_traces = np.zeros(self.env.num_states)

    def evaluate(self, policy: AbstractPolicy, num_episodes: int) -> np.ndarray:
        """
        Perform the TD prediction algorithm.

        :param policy: A policy object that provides action probabilities for each state.
        :param num_episodes: Number of episodes to run for estimating V(s).
        :return: The state-value function V(s) for the associated policy.
        """
        self.value_fun.fill(0)              # Reset value function.

        for _ in range(num_episodes):
            self._update_value_function(policy)

        return self.value_fun.copy()

    def _update_value_function(self, policy: AbstractPolicy) -> None:
        """
        Runs a single episode using the TD(λ) method to update the value function.

        :param policy: A policy object that provides action probabilities for each state.
        """
        current_state = self.env.current_state

        while True:
            action = policy.sample_action(current_state)
            new_state, reward, done = self.env.step(action)
            error = reward + self.env.discount_factor * self.value_fun[new_state] - self.value_fun[current_state]
            self.eligibility_traces[current_state] = self.eligibility_traces[current_state] + 1

            for state in self.env.states:

                self.value_fun[state] = self.value_fun[state] + self.alpha*error*self.eligibility_traces[state]
                self.eligibility_traces[state] = self.env.discount_factor*self.lambd*self.eligibility_traces[state]

            current_state = new_state
            if done:
                self.env.reset()
                break

