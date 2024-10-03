import util

from rl_mdp.model_free_prediction.monte_carlo_evaluator import MCEvaluator
from rl_mdp.model_free_prediction.td_evaluator import TDEvaluator
from rl_mdp.model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator


def main() -> None:
    """
    Starting point of the program, you can instantiate any classes, run methods/functions here as needed.
    """

    mdp = util.create_mdp()
    policy_1 = util.create_policy_1()
    policy_2 = util.create_policy_2()


    mc_evaluator = MCEvaluator(mdp)
    mc_evaluator.evaluate(policy_1, 1000)
    print(f'Monte Carlo policy 1: {mc_evaluator.value_fun}')
    mc_evaluator.evaluate(policy_2, 1000)
    print(f'Monte Carlo policy 2: {mc_evaluator.value_fun}')

    td_evaluator = TDEvaluator(mdp, 0.1)
    td_evaluator.evaluate(policy_1, 1000)
    print(f'TD policy 1: {td_evaluator.value_fun}')
    td_evaluator.evaluate(policy_2, 1000)
    print(f'TD policy 2: {td_evaluator.value_fun}')

    tdl_evaluator = TDLambdaEvaluator(mdp, alpha = 0.1, lambd = 0.5)
    tdl_evaluator.evaluate(policy_1, 1000)
    print(f'TD lambda policy 1: {tdl_evaluator.value_fun}')
    tdl_evaluator.evaluate(policy_2, 1000)
    print(f'TD lambda policy 2: {tdl_evaluator.value_fun}')


if __name__ == "__main__":
    main()
