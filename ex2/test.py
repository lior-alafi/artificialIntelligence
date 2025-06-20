import unittest

import pressure_plate
from check_for_tests import example2
from ex2 import Controller
example = {'chosen_action_prob': {'U': [0.9, 0.05, 0.05, 0], 'L': [0.1, 0.8, 0.075, 0.025],
                                  'R': [0.05, 0.05, 0.85, 0.05], 'D': [0.05, 0.1, 0.15, 0.7]},
           'finished_reward': 350,
           'opening_door_reward': {},
           'step_punishment': -2,
           'seed': 42}

class MyTestCase(unittest.TestCase):
    def test_something(self):
        right_corner = (
    (2,98,99),
    (98,1 ,99),
    (99,99,99),
)

        game = pressure_plate.create_pressure_plate_game((100, right_corner, example, True))
        ctrler = Controller(game)
        succ = ctrler.successor(((1,1),(),right_corner))
        self.assertEqual(len(succ), 4)  # add assertion here
        graph = ctrler.value_iteration(((1,1),(),right_corner),3)
        print(graph)


if __name__ == '__main__':
    unittest.main()

