import util
import model_free_prediction.monte_carlo_evaluator as mce


def main() -> None:
    """
    Starting point of the program, you can instantiate any classes, run methods/functions here as needed.
    """
    mdp = util.create_mdp()
    policy_1 = util.create_policy_1()
    policy_2 = util.create_policy_2()

    mc_evaluator = MCEvaluator(mdp)
    mc_evaluator.evaluate(policy_1, 5)



if __name__ == "__main__":
    main()
