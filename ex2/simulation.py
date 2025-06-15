import pygame
import time
import numpy as np
from sim_data import tests,stuff,problems
import hashlib
# צבעים
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
BLUE = (100, 100, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
YELLOW = (255, 255, 100)

# חוקים
BLANK = 0
WALL = 99
FLOOR = 98
AGENT = 1
GOAL = 2
AGENT_ON_GOAL = 3

KEY_BLOCKS = list(range(10, 20))
PRESSURE_PLATES = list(range(20, 30))
PRESSED_PLATES = list(range(30, 40))
LOCKED_DOORS = list(range(40, 50))

CELL_SIZE = 60//2
# Define the class PressurePlateProblem
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
KEY_VALID = 'valid'
KEY_POS = 'positions'
KEY_PLATES_POS = 'pressure_plates'
KEY_DOORS_POS =  'doors'
class PressurePlateProblem:
    """This class implements a pressure plate problem"""

    def __init__(self, initial):
        self.agent_2_goal_penalty = 2
        self.goal_2_doors_penalty = 4
        self.keys_2_plates_penalty = 4
        self.invalid_keys_penalty = 5

        self.actions = {
            "R": (0, 1),
            "L": (0, -1),
            "U": (1, 0),
            "D": (-1, 0),
        }


        self.valid_keys = {}
        self.visited = set()
        board = np.array(initial)
        self.board_shape = board.shape
        agent_pos = tuple(np.argwhere(board == AGENT)[0])
        self.walls = set(map(tuple, np.argwhere(board == WALL)))
        self.floors = set(map(tuple, np.argwhere(board == FLOOR)))
        self.goals = set(map(tuple, np.argwhere(board == GOAL)))
        self.plates = set(map(tuple, np.argwhere(np.isin(board, list(PRESSURE_PLATES)))))
        key_poses = []
        for key_id in KEY_BLOCKS:

            keys_pos = list(map(tuple, np.argwhere(board == key_id)))
            if len(keys_pos) == 0:
                continue

            p_plates_pos = list(map(tuple, np.argwhere(board == key_id + 10)))
            locked_doors = list(map(tuple, np.argwhere(board == key_id + 30)))

            key_poses.append((key_id, tuple(keys_pos)))
            self.valid_keys[key_id] = {
                    'KEY_POS': keys_pos,
                    'KEY_PLATES_POS': p_plates_pos,
                    'KEY_DOORS_POS': locked_doors,
                    'KEY_VALID': len(keys_pos) == len(p_plates_pos) and len(locked_doors) > 0
                }

        self.forbidden_tiles = list(PRESSURE_PLATES.union(PRESSED_PLATES.union(LOCKED_DOORS) ) ) + [WALL]
        self.initial_state = (agent_pos, tuple(key_poses), initial)
    def in_bound(self, pos):
        return (0 <= pos[0] < self.board_shape[0]) and (0 <= pos[1] < self.board_shape[1])
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
    # def successor(self, state):
    #     agent_pos = state[0]
    #     keys = state[1]
    #     board = state[2]
    #     board_hash =  hashlib.sha1(repr(board).encode()).hexdigest()
    #     if board_hash not in self.visited:
    #         self.visited.add(board_hash)
    #     else:
    #         return []
    #     if agent_pos in self.goals:
    #         return []
    #
    #     successors = []
    #
    #     for action,pos in self.actions.items():
    #         new_agent_pos = (agent_pos[0] + pos[0],agent_pos[1] + pos[1])
    #         if not self.in_bound(new_agent_pos):
    #             continue
    #         if new_agent_pos in self.goals:
    #             new_board = np.array(board)
    #             new_board[new_agent_pos[0],new_agent_pos[1]] = AGENT_ON_GOAL
    #             new_board[agent_pos[0], agent_pos[1]] = FLOOR
    #             return  [(action,(new_agent_pos,keys,tuple(map(tuple, new_board))))]
    #
    #         tile =  board[new_agent_pos[0]][new_agent_pos[1]]
    #         if tile in self.forbidden_tiles:
    #             continue
    #
    #         if tile in [FLOOR,BLANK]:
    #             new_board = np.array(board)
    #             new_board[new_agent_pos[0], new_agent_pos[1]] = AGENT
    #             new_board[agent_pos[0],agent_pos[1]] = FLOOR
    #             new_board = tuple(map(tuple, new_board))
    #             if hash(new_board)  not in self.visited:
    #                 successors.append((action,(new_agent_pos,keys,new_board)))
    #             continue
    #
    #         if tile in KEY_BLOCKS:
    #             key_new_pos = (new_agent_pos[0] + pos[0], new_agent_pos[1] + pos[1])
    #             if not self.in_bound(key_new_pos):
    #                 continue
    #             key_new_tile = board[key_new_pos[0]][key_new_pos[1]]
    #             if key_new_tile not in [FLOOR, BLANK, tile + 10]:
    #                 continue
    #
    #             new_board = np.array(board)
    #             new_board[new_agent_pos[0], new_agent_pos[1]] = AGENT
    #             new_board[agent_pos[0], agent_pos[1]] = FLOOR
    #
    #             keys_dict = {}
    #             for key_positions in keys:
    #                 keys_dict[key_positions[0]] = list(key_positions[1])
    #
    #             if tile in keys_dict and new_agent_pos in keys_dict[tile]:
    #                 keys_dict[tile].remove(new_agent_pos)
    #             else:
    #                 continue
    #
    #
    #
    #             if key_new_tile != tile + 10:
    #                 keys_dict[tile].append(key_new_pos)
    #
    #
    #             if key_new_tile == tile + 10:
    #                 new_board[key_new_pos[0]][key_new_pos[1]] = tile + 20
    #                 key_info = self.valid_keys[tile]
    #                 doors_unlocked = True
    #                 for plate_pos in key_info[KEY_PLATES_POS]:
    #                     if new_board[plate_pos[0]][plate_pos[1]] != tile + 20:
    #                         doors_unlocked = False
    #                         break
    #                 if doors_unlocked:
    #                     for door_pos in key_info[KEY_DOORS_POS]:
    #                         new_board[door_pos[0]][door_pos[1]] = BLANK
    #             else:
    #
    #                 new_board[key_new_pos[0]][key_new_pos[1]] = tile
    #             new_board = tuple(map(tuple, new_board))
    #             successors.append(
    #                     (action, (new_agent_pos, self.keys_dict_2_tuples_keys(keys_dict), new_board)))
    #
    #     return successors
    def keys_dict_2_tuples_keys(self,key_dict):
        return tuple((k, tuple(v)) for k, v in key_dict.items())


    def goal_test(self, state):
        board = state[2]
        for goal_pos in self.goals:
            if board[goal_pos[0]][goal_pos[1]] == AGENT_ON_GOAL:
                return True
        return  False


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

            if KEY_VALID in self.valid_keys[key_tile] and  self.valid_keys[key_tile][KEY_VALID]:
                # since all doors unlocked together we can just take the first one
                door_pos = self.valid_keys[key_tile][KEY_DOORS_POS][0]
                if node.state[2][door_pos[0]][door_pos[1]] not in [BLANK,FLOOR,AGENT]:

                    key_positions = np.array(key_grp[1])
                    remaining_plate_pos = np.array([x for x in self.valid_keys[key_tile][KEY_PLATES_POS] if node.state[2][x[0]][x[1]] == key_tile + 10])
                    if len(remaining_plate_pos) > 0:
                        agent_2_key_pos += np.sum((np.abs(key_positions-agent_pos)))
                        distances = np.sum(np.abs(key_positions-remaining_plate_pos),axis=1)
                        mh_key_2_plate_sum += sum(distances)

                for door_pos in self.valid_keys[key_tile][KEY_DOORS_POS]:
                    if  node.state[2][door_pos[0]][door_pos[1]] not in [BLANK, FLOOR]:
                        goal_door_distance = np.average([self.manhattan_distance(door_pos,goal) for goal in self.goals])
                        # since likelyhood of needing to pass through such door is high
                        if goal_door_distance < 3:
                            door_2_goal_sum += 1
            else:
                unnecessary_keys_movements +=1

        # return max(mh_key_2_plate_sum + agent_2_key_pos + door_2_goal_sum,min_agent_to_goal_distance )
        return self.agent_2_goal_penalty *min_agent_to_goal_distance +  self.keys_2_plates_penalty *mh_key_2_plate_sum \
              + self.goal_2_doors_penalty*door_2_goal_sum + self.invalid_keys_penalty * unnecessary_keys_movements*5

    # def h(self, node):
    #     agent_pos = node.state[0]
    #     if node.state[2][agent_pos[0]][agent_pos[1]] == AGENT_ON_GOAL:
    #         return 0
    #     keys2plate_distances = []
    #     sums = 0
    #     door_2_goal_sum = 0
    #
    #     min_agent_to_goal_distance = min(self.manhattan_distance(goal,agent_pos) for goal in self.goals)
    #     unnecessary_keys_movements = 0
    #     for key_grp in node.state[1]:
    #         key_tile = key_grp[0]
    #         #since all doors unlocked together
    #         if KEY_VALID in self.valid_keys[key_tile] and  self.valid_keys[key_tile][KEY_VALID]:
    #             door_pos = self.valid_keys[key_tile][KEY_DOORS_POS][0]
    #             if node.state[2][door_pos[0]][door_pos[1]] not in [BLANK,FLOOR]:
    #
    #                 key_positions = np.array(key_grp[1])
    #                 remaining_plate_pos = np.array([x for x in self.valid_keys[key_tile][KEY_PLATES_POS] if node.state[2][x[0]][x[1]] == key_tile + 10])
    #                 if len(remaining_plate_pos) > 0:
    #                     distances = np.sum(np.abs(key_positions-remaining_plate_pos),axis=1)
    #                     keys2plate_distances.append(distances)
    #                     # sums += min(distances)
    #                     sums += sum(distances)
    #
    #             for door_pos in self.valid_keys[key_tile][KEY_DOORS_POS]:
    #                 if  node.state[2][door_pos[0]][door_pos[1]] not in [BLANK, FLOOR]:
    #                     goal_door_distance = np.average([self.manhattan_distance(door_pos,goal) for goal in self.goals])
    #                     if goal_door_distance < 3:
    #                         door_2_goal_sum += 1
    #         else:
    #             unnecessary_keys_movements +=1
    #     return 2*min_agent_to_goal_distance + 4*sums + 4*door_2_goal_sum + unnecessary_keys_movements*5


    def manhattan_distance(self,lhs,rhs):
        return abs(lhs[0]-rhs[0])+abs(lhs[1]-rhs[1])




def draw_board(screen, grid, font):
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            val = grid[row][col]

            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            if val == WALL:
                color = BLACK
            elif val == FLOOR:
                color = WHITE
            elif val == AGENT or val == AGENT_ON_GOAL:
                color = BLUE
            elif val == GOAL:
                color = GREEN
            elif val in KEY_BLOCKS:
                color = RED
            elif val in PRESSURE_PLATES:
                color = YELLOW
            elif val in PRESSED_PLATES:
                color = (255, 165, 0)  # כתום
            elif val in LOCKED_DOORS:
                color = GRAY
            else:
                color = WHITE

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRAY, rect, 1)

            # כתיבת הערך
            text = font.render(str(val), True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)

def run_simulation(board,actions):
    # Initial state setup (map)
    # הלוח

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((len(board[0]) * (CELL_SIZE), len(board) * (CELL_SIZE)))
    pygame.display.set_caption("Pressure Plate Problem")


    clock = pygame.time.Clock()

    problem = PressurePlateProblem(board)
    state = problem.initial_state

    # Run the simulation
    for action in actions:
        succ,finish = problem.successor(state)
        for a, n_state in succ:
            if a == action:
                state = n_state
                board = state[2]
                break

        # Clear the screen and redraw the state
        screen.fill((255, 255, 255))  # White background
        draw_board(screen, board, pygame.font.SysFont(None, 24))
        pygame.display.flip()

        time.sleep(0.5)  # Slow down the movement to see it
        clock.tick(60)

        # Event handling to quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

    pygame.quit()


# Run the simulation
for x in  problems:
    print(x['name'])
    run_simulation(x['board'],x['astar_actions'])
    run_simulation(x['board'],x['bfs_actions'])
