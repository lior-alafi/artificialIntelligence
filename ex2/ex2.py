
import pressure_plate
import numpy as np



id = ["000000000"]


""" Rules """
BLANK = 0
WALL = 99
FLOOR = 98
AGENT = 1
GOAL = 2
AGENT_ON_GOAL = 3
PRESSURE_PLATES = set(range(20, 30))
LOCKED_DOORS = set(range(40, 50))
PRESSED_PLATES = set(range(30, 40))
KEY_BLOCKS = set(range(10, 20))
#



class Controller:
    """This class is a controller for a pressure plate game."""

    def __init__(self, game: pressure_plate.Game):
        """Initialize controller for given game model.
        """
        self.doors = set()
        self.valid_keys ={}
        self.original_game = game
        self.MAX_STEPS = game.get_max_steps()
        state = self.original_game.get_current_state()
        self.init_agent_pos = state[1]
        self.actions = {
            "R": (0, 1),
            "L": (0, -1),
            "U": (1, 0), #down irl
            "D": (-1, 0), #up irl
        }

        self.forbidden_tiles = list(PRESSURE_PLATES.union(PRESSED_PLATES.union(LOCKED_DOORS) ) ) + [WALL]
        self.initial_state = game.get_current_state()
        self.slip_indx2pos ={0:"U",1:"L",2:"R",3:"D"}
        self.slip_pos2indx ={"U":0,"L":1,"R":2,"D":3}
        self.__make_key_dict(state[0])
        self.goals = self.__find_item_pos_by_tile(state[0],GOAL)
        model = game.get_model()
        self.action_prob = model['chosen_action_prob']
        self.R = {"goal": model['finished_reward'], "step": model['step_punishment']}
        for door,reward in model[ 'opening_door_reward'].items():
            self.R[str(door)]= reward

        self.board_shape = state[0].shape
        self.door_pos2Type = {}
        for door in self.doors:
            tile = state[0][door[0],door[1]]
            self.door_pos2Type[door] = tile

        #value iteration hyper-params
        self.gamma = 0.8
        self.epsilon = 0.01

        self.previous_state = None
        self.remaining_steps = self.MAX_STEPS
        self.tree = None

        self.graph = {}
        self.policies = {}




    def goal_test(self, state):
        board = state[2]
        for goal_pos in self.goals:
            if board[goal_pos[0]][goal_pos[1]] == AGENT_ON_GOAL:
                return True
        return  False

    def __find_item_pos_by_tile(self,board,tile_type):
        return list(map(tuple, np.argwhere(board == tile_type)))

    def __make_key_dict(self,board):

        for key_id in KEY_BLOCKS:

            keys_pos = self.__find_item_pos_by_tile(board,key_id)
            if len(keys_pos) == 0:
                continue

            p_plates_pos = self.__find_item_pos_by_tile(board,key_id + 10)
            locked_doors = self.__find_item_pos_by_tile(board,key_id + 30)

            self.doors = self.doors | set(locked_doors)

            self.valid_keys[key_id] = {
                'KEY_POS': keys_pos,
                'KEY_PLATES_POS': p_plates_pos,
                'KEY_DOORS_POS': locked_doors,
            }
    def manhattan_distance(self,lhs,rhs):
        return abs(lhs[0]-rhs[0])+abs(lhs[1]-rhs[1])

    def in_bound(self, pos):
        return 0 <= pos[0] < self.board_shape[0] and 0 <= pos[1] < self.board_shape[1]

    def reward_aug(self,current_state,next_state):
        pos = current_state[0]
        board = current_state[2]
        next_pos = next_state[0]
        next_board = next_state[2]
        penalties_factor = -1
        plates_factor = 1
        opened_doors_factor = 1
        near_goal_factor = 1

        opened_doors = 0
        pressed_pplates = 0
        goal = 0

        for key_id,data in self.valid_keys.items():
            for pos in  data['KEY_PLATES_POS']:
               if board[pos[0]][pos[1]] != next_board[pos[0]][pos[1]]:
                   pressed_pplates += 1
            for pos in data['KEY_DOORS_POS']:
                if board[pos[0]][pos[1]] != next_board[pos[0]][pos[1]]:
                    opened_doors += 1

        penalties = 0
        if next_pos[0]== pos[0] and next_pos[1]== pos[1]:
            penalties += 1

        if board[next_pos[0]][next_pos[1]] in {WALL} | PRESSED_PLATES | PRESSURE_PLATES | LOCKED_DOORS:
            penalties += 5

        if board[next_pos[0]][next_pos[1]] == GOAL:
            goal = 1
        return penalties*penalties_factor + opened_doors_factor*opened_doors+plates_factor*pressed_pplates+near_goal_factor*goal

    def reward_state(self,agent_pos,board,is_s_tag=False):
        goal_factor = 10
        if agent_pos in self.goals:
            return self.R['goal']



        if agent_pos in self.doors and board[agent_pos[0]][agent_pos[1]] in [BLANK,FLOOR]:
            #door was opened
            door_type = str(self.door_pos2Type[agent_pos])
            return self.R[door_type]
        return self.R["step"] if not is_s_tag else 0.0


    def successor(self, state):
        agent_pos = state[0]
        keys = state[1]
        board = state[2]
        successors = []

        if agent_pos in self.goals:
            return [('U', state)],True

        for action, pos in self.actions.items():
            new_agent_pos = (agent_pos[0] + pos[0], agent_pos[1] + pos[1])
            if not self.in_bound(new_agent_pos):
                successors.append((action, state))
                continue
            if new_agent_pos in self.goals:
                new_board = np.array(board)
                new_board[new_agent_pos[0], new_agent_pos[1]] = AGENT_ON_GOAL
                new_board[agent_pos[0], agent_pos[1]] = FLOOR
                successors.append((action, (new_agent_pos, keys, tuple(map(tuple, new_board)))))

            tile = board[new_agent_pos[0]][new_agent_pos[1]]
            if tile in  PRESSURE_PLATES | PRESSED_PLATES | LOCKED_DOORS | {WALL}:
                successors.append((action, state))
                continue

            if tile in [FLOOR, BLANK]:
                new_board = np.array(board)
                new_board[new_agent_pos[0], new_agent_pos[1]] = AGENT
                new_board[agent_pos[0], agent_pos[1]] = FLOOR
                new_board = tuple(map(tuple, new_board))
                successors.append((action, (new_agent_pos, keys, new_board)))
                continue

            if tile in KEY_BLOCKS:
                key_new_pos = (new_agent_pos[0] + pos[0], new_agent_pos[1] + pos[1])
                if not self.in_bound(key_new_pos):
                    successors.append((action, state))
                    continue
                key_new_tile = board[key_new_pos[0]][key_new_pos[1]]
                if key_new_tile not in [FLOOR, BLANK, tile + 10]:
                    successors.append((action, state))
                    continue

                new_board = np.array(board)
                new_board[new_agent_pos[0], new_agent_pos[1]] = AGENT
                new_board[agent_pos[0], agent_pos[1]] = FLOOR

                keys_dict = {}
                for key_positions in keys:
                    keys_dict[key_positions[0]] = list(key_positions[1])

                if tile in keys_dict and new_agent_pos in keys_dict[tile]:
                    keys_dict[tile].remove(new_agent_pos)
                else:
                    # when we reach this?
                    successors.append((action, state))
                    continue

                if key_new_tile != tile + 10:
                    keys_dict[tile].append(key_new_pos)

                if key_new_tile == tile + 10:
                    new_board[key_new_pos[0]][key_new_pos[1]] = tile + 20
                    key_info = self.valid_keys[tile]
                    doors_unlocked = True
                    for plate_pos in key_info['KEY_PLATES_POS']:
                        if new_board[plate_pos[0]][plate_pos[1]] != tile + 20:
                            doors_unlocked = False
                            break
                    if doors_unlocked:
                        for door_pos in key_info['KEY_DOORS_POS']:
                            new_board[door_pos[0]][door_pos[1]] = FLOOR
                else:
                    new_board[key_new_pos[0]][key_new_pos[1]] = tile
                new_board = tuple(map(tuple, new_board))
                successors.append(
                    (action, (new_agent_pos, self.keys_dict_2_tuples_keys(keys_dict), new_board)))

        return successors,False


    def optimal_policy(self,horizon,graph):
        policies = {}
        visited = set()
        for _,data in graph.items():
            state = data['STATE']
            if hash(state) in visited:
                continue
            visited.add(hash(state))
            action_2_states = {}
            current_V_s_tag = np.zeros(4)
            action_vec = np.zeros(4)
            agent_pos, keys_poses, board = state
            succ,finish= self.successor(state)
            m = {}
            for i, n_state in enumerate(succ):
                action = n_state[0]
                action_2_states[action] = n_state[1]
                agent_new_pos, new_key_pos, new_board = n_state[1]
                data_s_tag = {
                    "CURRENT_VALUE": self.reward_state(agent_new_pos, new_board, True)+self.reward_aug(state,n_state[1]),
                    "next_value": 0.0,
                    "current_action:": "U"
                }
                if hash(n_state[1]) in graph:
                    data_s_tag = graph[hash(n_state[1])]

                if hash(n_state[1]) == hash(state):
                    data_s_tag['CURRENT_VALUE'] -= 10
                current_V_s_tag[self.slip_pos2indx[action]] = data_s_tag['CURRENT_VALUE']


                # calc for each action
            action_vec[self.slip_pos2indx['U']] = np.sum(self.action_prob['U'] * np.array(
                [self.reward_state(agent_pos, board, False) + self.gamma * current_V_s_tag]))
            action_vec[self.slip_pos2indx['D']] = np.sum(self.action_prob['D'] * np.array(
                [self.reward_state(agent_pos, board, False) + self.gamma * current_V_s_tag]))
            action_vec[self.slip_pos2indx['R']] = np.sum(self.action_prob['R'] * np.array(
                [self.reward_state(agent_pos, board, False) + self.gamma * current_V_s_tag]))
            action_vec[self.slip_pos2indx['L']] = np.sum(self.action_prob['L'] * np.array(
                [self.reward_state(agent_pos, board, False) + self.gamma * current_V_s_tag]))
            # pi*(s)=argmax_a(Pr(s'|s,a)(R(s) + gamma * max_a(V*(s')))
            next_action = self.slip_indx2pos[int(np.argmax(action_vec))]
            next_action = str(next_action)
            if next_action in action_2_states:
                policies[hash(state)] = {'action': next_action, 'value': np.max(action_vec),'state':state,"next_board": action_2_states[next_action]} #= {'action': next_action, 'value': np.max(action_vec),'board':board, "next_board": action_2_states[next_action]}
            else:
                break

        return policies
    def value_iteration(self, init_state,horizon):
        delta = np.infty


        while delta > self.epsilon:

            max_horizon = 0
            queue = [init_state]
            delta = 0
            visited = set()
            while len(queue) != 0 and len(queue) < 3000 and max_horizon < horizon + 1:

                state = queue[0]
                queue = queue[1:]
                data_s = {
                    "STATE": state,
                    'DEPTH': 0,
                    'PARENT': None,
                    'CURRENT_VALUE': 0.0,

                }
                if hash(state) in self.graph:
                    data_s = self.graph[hash(state)]
                if hash(state) in visited:
                    continue

                visited.add(hash(state))
                agent_pos, keys_poses, board = state
                action_vec = np.zeros(4)
                current_V_s_tag = np.zeros(4)
                plates = 0
                doors = 0
                if 'PRESSURE_PLATE' not in data_s:
                    for k,data in self.valid_keys.items():
                       plates += len([x for x in data['KEY_PLATES_POS'] if board[x[0]][x[1]] == k + 20])
                       doors += len([x for x in data['KEY_DOORS_POS'] if board[x[0]][x[1]] == k + 30])
                    data_s['PRESSURE_PLATE'] = plates
                    data_s['KEY_DOORS_POS'] = doors


                max_horizon = max(max_horizon,data_s['DEPTH'])
                V_s_prev = data_s["CURRENT_VALUE"]
                # populate current value for each succsessive state(s')
                succ,finish = self.successor(state)
                unseen_actions = {'U','D','L','R'}
                for i, n_state in enumerate(succ):
                        action = n_state[0]
                        unseen_actions.remove(action)

                        agent_new_pos, new_key_pos, new_board = n_state[1]
                        data_s_tag = {
                            "STATE": n_state[1],
                            'DEPTH': data_s['DEPTH'] + 1,
                            'PARENT': hash(state),
                            'CURRENT_VALUE': self.reward_state(agent_new_pos, new_board, True),
                        }
                        if hash(n_state[1]) in self.graph:
                            data_s_tag = self.graph[hash(n_state[1])]
                        plates = 0
                        doors = 0
                        if 'PRESSURE_PLATE' not in data_s_tag:
                            for k, data in self.valid_keys.items():
                                plates += len([x for x in data['KEY_PLATES_POS'] if new_board[x[0]][ x[1]] == k + 20])
                                doors += len([x for x in data['KEY_DOORS_POS'] if new_board[x[0]][ x[1]] == k + 30])
                            data_s_tag['PRESSURE_PLATE'] = plates
                            data_s_tag['KEY_DOORS_POS'] = doors

                        if data_s_tag['DEPTH'] > data_s['DEPTH'] + 1:
                            data_s_tag['DEPTH'] = data_s['DEPTH'] + 1
                            data_s_tag['PARENT'] = hash(state)


                        if data_s_tag['DEPTH'] > horizon:
                            continue

                        queue.append(n_state[1])
                        current_V_s_tag[self.slip_pos2indx[action]] = data_s_tag['CURRENT_VALUE'] -self.h(n_state[1])
                        self.graph[hash(n_state[1])] = data_s_tag
                for action in unseen_actions:
                    current_V_s_tag[self.slip_pos2indx[action]] = -100
                # calc for each action
                action_vec[self.slip_pos2indx['U']] = np.sum(self.action_prob['U'] * (current_V_s_tag))
                action_vec[self.slip_pos2indx['D']] = np.sum(self.action_prob['D'] * current_V_s_tag)
                action_vec[self.slip_pos2indx['L']] = np.sum(self.action_prob['L'] * current_V_s_tag)
                action_vec[self.slip_pos2indx['R']] = np.sum(self.action_prob['R'] * current_V_s_tag)
                # V*(s)=R(s) + gamma * max_a(Pr(s'|s,a)V*(s'))
                data_s['CURRENT_VALUE'] = self.reward_state(agent_pos, board, False) + self.gamma * np.max(action_vec)
                delta = np.max([abs(data_s['CURRENT_VALUE'] - V_s_prev), delta])
                self.graph[hash(state)] = data_s

                # process = psutil.Process(os.getpid())
                # memory_info = process.memory_info()
                # print(
                #     f"{(dt.datetime.now() - start).seconds}s iteration: {i} Memory Usage (RSS): {memory_info.rss / (1024 * 1024):.2f} MB  delta: {delta} |states| = {len(self.states)}")  # Resident Set Size, memory allocated to the process
                # print(
                #     f"{i} Memory Usage (VMS): {memory_info.vms / (1024 * 1024):.2f} MB  delta: {delta} |states| = {len(self.states)}")  # Virtual Memory Size, memory allocated to the process
            # print(f'delta {delta}')

        return self.graph

    def h(self,state):
        agent_pos = state[0]
        board = np.array(state[2])
        keys = state[1]
        board = np.array(board)

        agent_2_goal = np.min([self.manhattan_distance(pos,agent_pos) for pos in self.goals])
        agent_2_keys = 0
        keys_2_plates = 0
        for key_data in keys:
            key_type = int(key_data[0])
            key_poses = key_data[1]
            plates = self.__find_item_pos_by_tile(board,key_type+10)
            if len(key_poses) == len(plates):
                for i in range(len(key_poses)):
                    keys_2_plates += self.manhattan_distance(key_poses[i],plates[i])

            agent_2_keys += np.sum([self.manhattan_distance(pos,agent_pos) for pos in key_poses])
        return (agent_2_keys+keys_2_plates+agent_2_goal)/self.board_shape[0]

        # return (5*agent_2_keys+4*keys_2_plates+6*agent_2_goal)/10
        # return min(agent_2_keys+keys_2_plates,agent_2_goal)

    def keys_dict_2_tuples_keys(self,key_dict):
        return tuple((k, tuple(v)) for k, v in key_dict.items())

    def find_keys_pos(self,board):
        key_poses = []
        for key_id in KEY_BLOCKS:

            keys_pos = self.__find_item_pos_by_tile(board,key_id)
            if len(keys_pos) == 0:
                continue

            key_poses.append((key_id, tuple(keys_pos)))

        return key_poses

    def open_doors(self,map,key_type):
        if len(self.__find_item_pos_by_tile(map,key_type)) != 0:
            return False
        key_data = self.valid_keys[key_type]
        for plate_pos in key_data['KEY_PLATES_POS']:
            if map[plate_pos[0],plate_pos[1]]  != key_type + 20:
                return False
        for door_pos in key_data['KEY_DOORS_POS']:
            map[door_pos[0],door_pos[1]] = FLOOR
        return True


    def ndarray2tuple(self,ndarr):
        return tuple(map(tuple, ndarr))
    def choose_next_action(self, state):
        """Choose next action for a pressure plate game given the current state of the game.
        """
        board = state[0]
        agent_pos = state[1]
        steps = state[2]
        done = state[3]
        successful = state[4]

        current_state = (agent_pos, tuple(self.find_keys_pos(state[0])), self.ndarray2tuple(state[0]))
        if self.previous_state is None:
            self.previous_state = current_state

        horizon = 10

        s = self.graph.get(hash(current_state))
        if s is None or s['DEPTH'] == (horizon+ self.MAX_STEPS-self.remaining_steps)//2-1:
            self.graph = self.value_iteration(current_state,horizon+(self.MAX_STEPS - self.remaining_steps))
            self.policies = self.optimal_policy(self.MAX_STEPS,self.graph )

        data = self.policies.get(hash(current_state))

        for h in list(self.graph.keys()):
            if self.graph[h]['DEPTH'] < (self.MAX_STEPS - self.remaining_steps):
                del self.graph[h]

        self.remaining_steps -= 1
        self.previous_state = current_state
        if data is None:
            return 'U'
        return data['action']

