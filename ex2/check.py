import numpy as np

import pressure_plate
import ex2
import sim_data
from problems import stuff


def solve(game: pressure_plate.Game,hyper_params=None):
    policy = ex2.Controller(game,hyper_parameters=hyper_params,debug=True)
    for i in range(game.get_max_steps()):
        game.submit_next_action(chosen_action=policy.choose_next_action(game.get_current_state()))
        if game.get_current_state()[3]:
            break
    print('Game result:\n\tMap state ->\n', game.get_current_state()[0], '\n\tFinished in', game.get_current_state()[2],
          'Steps.\n\tReward result->',game.get_current_reward())
    print("Game finished ", "" if game.get_current_state()[-1] else "un", "successfully.", sep='')
    game.show_history()
    return game.get_current_reward()


# ["U", "L", "R", "D"]
example = {'chosen_action_prob': {'U': [0.9, 0.05, 0.05, 0], 'L': [0.1, 0.8, 0.075, 0.025],
                                  'R': [0.05, 0.05, 0.85, 0.05], 'D': [0.05, 0.1, 0.15, 0.7]},
           'finished_reward': 350,
           'opening_door_reward': {10: 5, 11: 7, 12: 9, 13: 11, 14: 13, 15: 15, 16: 17, 17: 19, 18: 21, 19: 23},
           'step_punishment': -2,
           'seed': 42}

example2 = {'chosen_action_prob': {'U': [0.6, 0.05, 0.05, 0.3], 'L': [0, 0.9, 0.075, 0.025],
                                   'R': [0.15, 0.02, 0.7, 0.13], 'D': [0.05, 0.13, 0.15, 0.67]},
            'finished_reward': 200,
            'opening_door_reward': {10: -3, 11: 2, 12: 15, 13: -6, 14: 3, 15: -10, 16: 17, 17: 0, 18: 1, 19: -2},
            'step_punishment': -2,
            'seed': 42}
# experiment with the seed as well
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

seed_array = [42,13,131,146]#,142,652,34,5,2,57,4,345,1,2,7,77,777,565,4,344,3424,636,666,8,6,89,9,76,24,36]
gamma = np.arange(start=0.05,stop=1.05,step=0.05)
horizon = [x for x in range(1,20)]
queue = [x for x in range(100,10000,1000)]


def main():
    import datetime
    debug_mode = False



    # rewards = []
    # start = datetime.datetime.now()
    # params = {'gamma':np.random.choice(gamma),'horizon':np.random.choice(horizon),'max_queue':np.random.choice(queue)}
    # for seed in seed_array:
    #             # print(params)
    #             example2['seed'] = seed
    #             start = datetime.datetime.now()
    #             game = pressure_plate.create_pressure_plate_game((100, problem1, example2, debug_mode))
    #             r = solve(game)
    #             rewards.append(r)
    # print(f'{datetime.datetime.now()-start} avg: {np.mean(rewards)} max: {np.max(rewards)} rewards: {rewards}')


    for prob in sim_data.problems+sim_data.tests:
        rewards = []
        print(prob['name'])
        start = datetime.datetime.now()
        for seed in seed_array:
                # print(params)
                print(prob['name'],seed)
                example['seed']= seed
                game2 = pressure_plate.create_pressure_plate_game((200, prob['board'], example, False))
                r= solve(game2)
                rewards.append(r)
        print(f'{prob['name']} {datetime.datetime.now()-start} avg: {np.mean(rewards)} max: {np.max(rewards)} rewards: {rewards}')
        print("#"*10)

if __name__ == "__main__":
    main()
