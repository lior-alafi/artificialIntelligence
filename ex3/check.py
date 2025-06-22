import base64
import numpy as np
import ex3
import dill
from pprint import pprint
import time


def generate_policy(policy_file):
    # TODO: you will be able to make limited changes to this function in order to train your policy
    ex3.Policy(policy_file=policy_file, train=True)


def load_policy_and_solve(policy_file, gameClass, test_games, debug=False):
    policy = ex3.Policy(policy_file=policy_file, train=False)
    t1 = time.time()
    print("The average score for the problem is:",
          gameClass.evaluate_policy(policy, test_games, visualize=debug))
    t2 = time.time()
    print("Policy evaluation took: ", t2 - t1, " seconds")


problem1 = (
    (99,99,99,99,99,99),
    (99,2 ,40,98,98,99),
    (99,99,99,10,98,99),
    (99,98,98,98,98,99),
    (99,20,98,98,1 ,99),
    (99,99,99,99,99,99),
)

problem2 = (
    (99,99,99,99,99,99,99,99,99,99,99,99,99,99,99),
    (99,98,98,98,99,99,99,99,99,99,99,99,99,99,99),
    (99,98,99,98,99,99,99,99,99,99,98,98,99,99,99),
    (99,98,99,98,98,99,25,98,99,99,98,98,98,99,99),
    (99,98,99,98,2 ,45,98,98,98,98,98,98,98,99,99),
    (99,98,99,99,99,99,98,98,99,99,99,42,99,99,99),
    (99,98,98,98,98,99,99,99,99,99,22,98,98,99,99),
    (99,99,99,99,98,99,98,98,98,99,98,98,98,99,99),
    (99,98,98,98,98,99,12,98,98,99,98,98,98,99,99),
    (99,98,99,99,23,98,98,15,98,99,99,41,99,99,99),
    (99,98,99,99,98,98,98,98,98,99,20,98,98,98,99),
    (99,98,99,99,98,98,99,98,98,99,98,98,10,98,99),
    (99,98,99,99,98,13,98,98,98,40,11,98,98,98,99),
    (99,98,43,98,98,98,98,98,98,99,21,98,98,1 ,99),
    (99,99,99,99,99,99,99,99,99,99,99,99,99,99,99),)


def main():
    debug_mode = True
    with open('pressure_plate.encoded', "rb") as f:
        encoded = f.read()
    gameClass = dill.loads(base64.b64decode(encoded))
    # game = gameClass.generate_new_game(100, problem1, 32, debug_mode)
    # TODO: it is suggested to create more problems for your policy to train/test on, and play with the seeds.
    test_games = [(100, problem1, 32), (200, problem2, 42)]
    generate_policy("my_policy_file")
    load_policy_and_solve("my_policy_file", gameClass, test_games, debug=debug_mode)


if __name__ == "__main__":
    main()
