import datetime
from collections import OrderedDict

import pressure_plate
import numpy as np
import operator, math, random, copy, sys, os.path, bisect, inspect
from abc import ABC, abstractmethod
import bisect


id = ["000000000"]


class Queue(ABC):
    """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
        PriorityQueue(lt): Queue where items are sorted by lt, (default <).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
    Note that isinstance(Stack(), Queue) is false, because we implement stacks
    as lists.  If Python ever gets interfaces, Queue will be an interface."""

    def update(x, **entries):
        """Update a dict; or an object with slots; according to entries.
        # >>> update({'a': 1}, a=10, b=20)
        # {'a': 10, 'b': 20}
        # >>> update(Struct(a=1), a=10, b=20)
        Struct(a=10, b=20)
        """
        if isinstance(x, dict):
            x.update(entries)
        else:
            x.__dict__.update(entries)
        return x

    @abstractmethod
    def __init__(self):
        pass


    def extend(self, items):
        for item in items: self.append(item)

class PriorityQueue(Queue):
    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with minimum f(x) is
    returned first; if order is max, then it is the item with maximum f(x)."""

    def __init__(self, order=min, f=lambda x: x):
        self.update( A=[], order=order, f=f)

    def append(self, item):

        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        self.update(self, state=state, parent=parent, action=action,
               path_cost=path_cost, depth=0)
        if parent:
            self.depth = parent.depth + 1
        self.state = state
        self.action = action
        self.path_cost = path_cost


    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1


    def update(self,x, **entries):
        """Update a dict; or an object with slots; according to entries.
        update({'a': 1}, a=10, b=20)
        {'a': 10, 'b': 20}
         update(Struct(a=1), a=10, b=20)
        Struct(a=10, b=20)
        """
        if isinstance(x, dict):
            x.update(entries)
        else:
            x.__dict__.update(entries)
        return x
    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def path(self):
        "Create a list of nodes from the root to this node."
        x, result = self, [self]
        while x.parent:
            result.append(x.parent)
            x = x.parent
        return result

    def expand(self, problem):
        "Return a list of nodes reachable from this node. [Fig. 3.8]"
        l =  [Node(next, self, act,
                     problem.path_cost(self.path_cost, self.state, act, next))
                for (act, next) in problem.successor(self.state)]
        return  l

    def __eq__(self, other):
        return (self.f == other.f)

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return (self.f < other.f)

    def __gt__(self, other):
        return (self.f > other.f)

    def __le__(self, other):
        return (self < other) or (self == other)

    def __ge__(self, other):
        return (self > other) or (self == other)


class AStar:
    def __init__(self,problem,f,max_traj_time=None):
        self.fringe = PriorityQueue(min, f)
        self.max_trajectory_calc = max_traj_time #in seconds

    def graph_search(self,problem, fringe,initial = None):
        """Search through the successors of a problem to find a goal.
        The argument fringe should be an empty queue.
        If two paths reach a state, only use the best one. [Fig. 3.18]"""
        closed = {}
        expanded = 0
        if initial:
            fringe.append((Node(initial)))
        else:
            fringe.append(Node(problem.initial))
        start = datetime.datetime.now()
        while fringe :
            node = fringe.pop()
            if problem.goal_test(node.state):
                return node, expanded
            if self.max_trajectory_calc is not None and self.max_trajectory_calc <= (datetime.datetime.now() - start).seconds:
                break
            if node.state not in closed:
                closed[node.state] = True
                fringe.extend(node.expand(problem))
                expanded += 1
        return None

    def search(self,problem, h=None,initial_state=None):
        """A* search is best-first graph search with f(n) = g(n)+h(n).
        You need to specify theh function when you call astar_search.
        Uses the pathmax trick: f(n) = max(f(n), g(n)+h(n))."""
        h = h or problem.h

        def f(n):

            return max(getattr(n, 'f', -np.infty), n.path_cost + h(n))

        f = self.memoize(f, 'f')
        return self.graph_search(problem, PriorityQueue(min, f),initial_state)

    def memoize(self,fn, slot=None):
        """Memoize fn: make it remember the computed value for any argument list.
        If slot is specified, store result in that slot of first argument.
        If slot is false, store results in a dictionary."""
        if slot:
            def memoized_fn(obj, *args):
                if hasattr(obj, slot):
                    return getattr(obj, slot)
                else:
                    val = fn(obj, *args)
                    setattr(obj, slot, val)
                    return val
        else:
            def memoized_fn(*args):
                if not memoized_fn.cache.has_key(args):
                    memoized_fn.cache[args] = fn(*args)
                return memoized_fn.cache[args]

            memoized_fn.cache = {}
        return memoized_fn


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

    def __init__(self, game: pressure_plate.Game,hyper_parameters=None,debug=False):
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
        self.horizon = 2
        self.max_queue = 9000
        if hyper_parameters is not None:
            self.gamma = hyper_parameters['gamma']
            self.horizon = hyper_parameters['horizon']
            self.max_queue = hyper_parameters['max_queue']

        self.previous_state = None
        self.remaining_steps = self.MAX_STEPS
        self.tree = None
        self.initial = (self.init_agent_pos,tuple(self.find_keys_pos(state[0])),self.ndarray2tuple(state[0]))

        self.ignore_trajectory = False # incase trajectory is none
        self.max_traj_calc_time =30#25
        self.count_stray_path = 0
        self.trajectory_dict = OrderedDict()
        self.astar = AStar(self,self.h,self.max_traj_calc_time)
        self.a_star(self.initial)

        self.graph = {}
        self.policies = {}


    def a_star(self,state,debug=False):
        start = datetime.datetime.now()
        astar_sol = self.astar.search(self, self.h,state)
        self.trajectory_dict = OrderedDict()
        if astar_sol:
            solve = astar_sol[0].path()[::-1]
            for i, pi in enumerate(solve):
                if i == len(solve) - 1:
                    break
                p_curr = solve[i + 1].state[0]
                p_next = solve[i].state[0]
                d = (p_curr[0] - p_next[0], p_curr[1] - p_next[1])
                for a, dir in self.actions.items():
                    if dir == d:
                        pi.action = a
                        break
                self.trajectory_dict[pi.state] = {"action": pi.action, "CURRENT_VALUE":  10, 'agent': pi.state[0]}
                if debug:
                    print(self.trajectory_dict[pi.state])
        else:
            self.ignore_trajectory = True
        print(f'A* time = {datetime.datetime.now()-start}')
    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1
    def dead_end(self,board,pos,tile):
        dead_ends = [["L","R","U"],
                         ["L","R","D"],
                         ["L","U","D"],
                         ["R","U","D"],
                         ]

        DEAD_END = {WALL, None} | PRESSED_PLATES
        if tile in KEY_BLOCKS:
            # these are not necessarily dead ends:
            #(
            # (99,99,99,99,99),
            # (99,1,98,98,99),
            # (99,11,10,98,40),
            # (99,98,98 ,20,2),
            # (99,99,99,99,99),)
            # this board is valid and solvable,even if we add 21 and 41
            #  somewhere it may still be valid and necessary to put 11 in dead end just to free
            #  10, a look ahead might help but this situation prove this is a delicate matter
            #
            # dead_ends = dead_ends + [["LU","L","LD","D","RD"],# |_
            #                          ["LU", "L", "LD", "U", "RU"], #|-
            #                          ["LU", "U", "RU", "R", "RD"], #-|
            #                          ["LD", "D", "RU", "R", "RD"], #_|
            #                          ]
            DEAD_END = DEAD_END | (PRESSURE_PLATES - {tile + 10})

        neighbours = self.get_8_neighbours(board,pos)
        for dead_end in dead_ends:
            is_dead_end = True
            for dir in dead_end:
                if neighbours[dir]['tile'] not in DEAD_END:
                    is_dead_end = False
                    break

            if is_dead_end:
                return True

        return False


    def get_8_neighbours(self,board,pos):
        additional_actions = {
            "LU": (1,-1),
            "RU": (1,1),
            "LD": (-1, -1),
            "RD": (-1, 1),
        }
        neighbours = {}
        for act,direction in self.actions.items():
            new_pos = (pos[0] + direction[0],pos[1] + direction[1])
            neighbours[act] = {'tile':None,"pos":None}
            if self.in_bound(new_pos):
                neighbours[act] = {'tile': board[new_pos[0]][new_pos[1]], "pos": new_pos}

        for act,direction in additional_actions.items():
            new_pos = (pos[0] + direction[0], pos[1] + direction[1])
            neighbours[act] = {'tile': None, "pos": None}
            if self.in_bound(new_pos):
                neighbours[act] = {'tile': board[new_pos[0]][new_pos[1]], "pos": new_pos}

        return neighbours



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
                'KEY_VALID': len(p_plates_pos) ==len(keys_pos)
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
        plates_factor = 20
        opened_doors_factor = 20

        opened_doors = 0
        pressed_pplates = 0

        for key_id,data in self.valid_keys.items():
            for pos in  data['KEY_PLATES_POS']:
               if board[pos[0]][pos[1]] != next_board[pos[0]][pos[1]]:
                   pressed_pplates += 1
            for pos in data['KEY_DOORS_POS']:
                if board[pos[0]][pos[1]] != next_board[pos[0]][pos[1]]:
                    opened_doors += 1

        return opened_doors_factor*opened_doors+plates_factor*pressed_pplates

    def reward_state(self,agent_pos,board,is_s_tag=False):
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
            return []

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

                # if self.dead_end(board,new_agent_pos,key_new_tile):
                #     successors.append((action, state))
                #     continue

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

        return successors


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
            succ= self.successor(state)
            m = {}
            for i, n_state in enumerate(succ):
                action = n_state[0]
                action_2_states[action] = n_state[1]
                agent_new_pos, new_key_pos, new_board = n_state[1]
                traj_data = self.trajectory_dict.get(n_state[1])
                data_s_tag = {
                    "CURRENT_VALUE": self.reward_state(agent_new_pos, new_board, True),# + (traj_data['CURRENT_VALUE'] if traj_data is not None else 0.0),
                    "next_value": 0.0,
                    "current_action:": "U"
                }
                if hash(n_state[1]) in graph:
                    data_s_tag = graph[hash(n_state[1])]

                # if hash(n_state[1]) == hash(state):
                #     data_s_tag['CURRENT_VALUE'] -= 10
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
                policies[hash(state)] = {'action': next_action, 'value': np.max(action_vec)}#,'state':state,"next_board": action_2_states[next_action]} #= {'action': next_action, 'value': np.max(action_vec),'board':board, "next_board": action_2_states[next_action]}
            else:
                break

        return policies
    def value_iteration(self, init_state,horizon):
        delta = np.infty

        graph = {}
        while delta > self.epsilon:

            max_horizon = 0
            queue = [init_state]
            delta = 0
            visited = set()
            while len(queue) != 0 and len(queue) < 9000 and max_horizon < horizon + 1:
                state = queue[0]
                queue = queue[1:]

                curr_value = 0.0
                traj_data = self.trajectory_dict.get(state)

                data_s = {
                    "STATE": state,
                    'DEPTH': 0,
                    'PARENT': None,
                    'CURRENT_VALUE': 0 if traj_data is None else traj_data['CURRENT_VALUE'],

                }
                if hash(state) in graph:#self.graph:
                    data_s = graph[hash(state)] #self.graph[hash(state)]
                if hash(state) in visited:
                    continue

                visited.add(hash(state))
                agent_pos, keys_poses, board = state
                action_vec = np.zeros(4)
                current_V_s_tag = np.zeros(4)



                max_horizon = max(max_horizon,data_s['DEPTH'])
                V_s_prev = data_s["CURRENT_VALUE"]
                # populate current value for each succsessive state(s')
                succ = self.successor(state)
                # unseen_actions = {'U','D','L','R'}
                for i, n_state in enumerate(succ):
                        action = n_state[0]
                        # unseen_actions.remove(action)

                        agent_new_pos, new_key_pos, new_board = n_state[1]
                        traj_data = self.trajectory_dict.get(n_state[1])
                        data_s_tag = {
                            "STATE": n_state[1],
                            'DEPTH': data_s['DEPTH'] + 1,
                            'PARENT': state,
                            'CURRENT_VALUE': self.reward_state(agent_new_pos, new_board, True) if traj_data is None else traj_data['CURRENT_VALUE']
                            # 'CURRENT_VALUE':(self.reward_aug(state,n_state[1])+self.reward_state(agent_new_pos, new_board, True)) if traj_data is None else traj_data['CURRENT_VALUE']
                        }
                        if hash(n_state[1]) in graph:
                            data_s_tag = graph[hash(n_state[1])]


                        # if data_s_tag['DEPTH'] > data_s['DEPTH'] + 1:
                        #     data_s_tag['DEPTH'] = data_s['DEPTH'] + 1
                        #     data_s_tag['PARENT'] = hash(state)
                        #

                        if data_s_tag['DEPTH'] > horizon:
                            continue

                        queue.append(n_state[1])
                        current_V_s_tag[self.slip_pos2indx[action]] = data_s_tag['CURRENT_VALUE']
                        graph[hash(n_state[1])] = data_s_tag
                # for action in unseen_actions:
                #     current_V_s_tag[self.slip_pos2indx[action]] = -100
                # calc for each action
                action_vec[self.slip_pos2indx['U']] = np.sum(self.action_prob['U'] * current_V_s_tag)
                action_vec[self.slip_pos2indx['D']] = np.sum(self.action_prob['D'] * current_V_s_tag)
                action_vec[self.slip_pos2indx['L']] = np.sum(self.action_prob['L'] * current_V_s_tag)
                action_vec[self.slip_pos2indx['R']] = np.sum(self.action_prob['R'] * current_V_s_tag)
                # V*(s)=R(s) + gamma * max_a(Pr(s'|s,a)V*(s'))
                data_s['CURRENT_VALUE'] = self.reward_state(agent_pos, board, False) + self.gamma * np.max(action_vec)
                delta = np.max([abs(data_s['CURRENT_VALUE'] - V_s_prev), delta])
                graph[hash(state)] = data_s

                # process = psutil.Process(os.getpid())
                # memory_info = process.memory_info()
                # print(
                #     f"{(dt.datetime.now() - start).seconds}s iteration: {i} Memory Usage (RSS): {memory_info.rss / (1024 * 1024):.2f} MB  delta: {delta} |states| = {len(self.states)}")  # Resident Set Size, memory allocated to the process
                # print(
                #     f"{i} Memory Usage (VMS): {memory_info.vms / (1024 * 1024):.2f} MB  delta: {delta} |states| = {len(self.states)}")  # Virtual Memory Size, memory allocated to the process
            # print(f'delta {delta}')

        return graph
    def h(self, node):
        agent_pos = node.state[0]
        if node.state[2][agent_pos[0]][agent_pos[1]] == AGENT_ON_GOAL:
            return 0

        mh_key_2_plate_sum = 0
        door_2_goal_sum = 0

        min_agent_to_goal_distance = min(self.manhattan_distance(goal,agent_pos) for goal in self.goals)


        unnecessary_keys_movements = 0
        agent_2_key_pos = 0
        for key_grp in node.state[1]:
            key_tile = key_grp[0]

            if 'KEY_VALID' in self.valid_keys[key_tile] and  self.valid_keys[key_tile]['KEY_VALID'] and len(self.valid_keys[key_tile]['KEY_DOORS_POS']) > 0:
                # since all doors unlocked together we can just take the first one

                door_pos = self.valid_keys[key_tile]['KEY_DOORS_POS'][0]
                if node.state[2][door_pos[0]][door_pos[1]] not in [BLANK,FLOOR,AGENT]:

                    key_positions = np.array(key_grp[1])
                    remaining_plate_pos = np.array([x for x in self.valid_keys[key_tile]['KEY_PLATES_POS'] if node.state[2][x[0]][x[1]] == key_tile + 10])
                    if len(remaining_plate_pos) > 0:
                        agent_2_key_pos += np.sum((np.abs(key_positions-agent_pos)))
                        distances = np.sum(np.abs(key_positions-remaining_plate_pos),axis=1)
                        mh_key_2_plate_sum += sum(distances)

                for door_pos in self.valid_keys[key_tile]['KEY_DOORS_POS']:
                    if  node.state[2][door_pos[0]][door_pos[1]] not in [BLANK, FLOOR]:
                        goal_door_distance = np.average([self.manhattan_distance(door_pos,goal) for goal in self.goals])
                        # since likelyhood of needing to pass through such door is high
                        if goal_door_distance < 3:
                            door_2_goal_sum += 1
            else:
                unnecessary_keys_movements +=1

        # return max(mh_key_2_plate_sum + agent_2_key_pos + door_2_goal_sum,min_agent_to_goal_distance )
        return 2 *min_agent_to_goal_distance +  4 *mh_key_2_plate_sum \
              + 4*door_2_goal_sum +  5* unnecessary_keys_movements

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


    def ndarray2tuple(self,ndarr):
        return tuple(map(tuple, ndarr))
    def choose_next_action(self, state):
        """Choose next action for a pressure plate game given the current state of the game.
        """
        board = state[0]
        agent_pos = state[1]

        current_state = (agent_pos, tuple(self.find_keys_pos(state[0])), self.ndarray2tuple(state[0]))


        traj_data = self.trajectory_dict.get(current_state)
        if traj_data:
            self.count_stray_path = 0
            return traj_data['action']

        if not self.ignore_trajectory and self.count_stray_path == self.horizon:
            self.trajectory_dict = OrderedDict()
            self.a_star(current_state)
            traj_data = self.trajectory_dict.get(current_state)
            if traj_data:
                self.count_stray_path = 0
                return traj_data['action']

        # s = self.graph.get(hash(current_state))
        # if s is None:# or s['DEPTH'] == (self.horizon+ self.MAX_STEPS-self.remaining_steps)//2-1:

        self.graph = {}
        self.graph = self.value_iteration(current_state,self.horizon)
        self.policies = self.optimal_policy(self.MAX_STEPS,self.graph )
        self.count_stray_path += 1
        data = self.policies.get(hash(current_state))



        self.remaining_steps -= 1
        if data is None:
            return 'U'
        return data['action']

