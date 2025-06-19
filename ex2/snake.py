import pygame
import random
import numpy as np
import pickle
from tabulate import tabulate

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 640, 480
CELL_SIZE = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Directions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Q-learning parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.5  # Exploration rate
EPISODES = 20000  # Number of episodes for training

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Snake Game with Q-learning')

# Initialize font
font = pygame.font.SysFont(None, 36)

class SnakeGame:
    """
    A class to represent the Snake game.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the game to the initial state.
        """
        self.snake = [(WIDTH // 2, HEIGHT // 2)]  # Initial snake position
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])  # Initial direction
        self.food = self.place_food()  # Place the first food
        self.score = 0  # Initial score
        self.done = False  # Game over flag

    def place_food(self):
        """
        Place food at a random position not occupied by the snake.
        """
        while True:
            x = random.randint(0, (WIDTH // CELL_SIZE) - 1) * CELL_SIZE
            y = random.randint(0, (HEIGHT // CELL_SIZE) - 1) * CELL_SIZE
            if (x, y) not in self.snake:
                return (x, y)

    def step(self, action):
        """
        Take a step in the game based on the action.
        """
        # Update direction based on action
        if action == UP and self.direction != DOWN:
            self.direction = UP
        elif action == DOWN and self.direction != UP:
            self.direction = DOWN
        elif action == LEFT and self.direction != RIGHT:
            self.direction = LEFT
        elif action == RIGHT and self.direction != LEFT:
            self.direction = RIGHT

        # Calculate new head position
        head_x, head_y = self.snake[0]
        if self.direction == UP:
            head_y -= CELL_SIZE
        elif self.direction == DOWN:
            head_y += CELL_SIZE
        elif self.direction == LEFT:
            head_x -= CELL_SIZE
        elif self.direction == RIGHT:
            head_x += CELL_SIZE

        new_head = (head_x, head_y)

        # Check for collisions with walls or itself
        if (head_x < 0 or head_x >= WIDTH or head_y < 0 or head_y >= HEIGHT or new_head in self.snake):
            self.done = True
            return -1  # Negative reward for collision

        self.snake.insert(0, new_head)

        # Check if food is eaten
        if new_head == self.food:
            self.food = self.place_food()
            self.score += 1
            return 1  # Positive reward for eating food
        else:
            self.snake.pop()
            return 0  # Neutral reward for normal move

    def get_state(self):
        """
        Get the current state of the game.
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        state = [
            head_x < food_x,  # Food is to the right
            head_x > food_x,  # Food is to the left
            head_y < food_y,  # Food is below
            head_y > food_y,  # Food is above
            self.direction == UP,
            self.direction == DOWN,
            self.direction == LEFT,
            self.direction == RIGHT,
            (head_x, head_y - CELL_SIZE) in self.snake,  # Danger up
            (head_x, head_y + CELL_SIZE) in self.snake,  # Danger down
            (head_x - CELL_SIZE, head_y) in self.snake,  # Danger left
            (head_x + CELL_SIZE, head_y) in self.snake,  # Danger right
        ]

        return tuple(state)


def train_q_learning():
    """
    Train the Snake game using Q-learning.
    """
    # Load the existing Q-table if it exists
    try:
        with open('q_table.pkl', 'rb') as f:
            q_table = pickle.load(f)
        print("Loaded existing Q-table.")
    except FileNotFoundError:
        q_table = {}
        print("No existing Q-table found. Starting fresh.")

    game = SnakeGame()

    for episode in range(EPISODES):
        game.reset()
        state = game.get_state()

        while not game.done:
            # Choose action based on epsilon-greedy policy
            if random.uniform(0, 1) < EPSILON:
                action = random.choice([UP, DOWN, LEFT, RIGHT])
            else:
                action = max(q_table.get(state, {a: 0 for a in [UP, DOWN, LEFT, RIGHT]}), key=q_table.get(
                    state, {a: 0 for a in [UP, DOWN, LEFT, RIGHT]}).get)

            reward = game.step(action)
            next_state = game.get_state()

            if state not in q_table:
                q_table[state] = {a: 0 for a in [UP, DOWN, LEFT, RIGHT]}

            if next_state not in q_table:
                q_table[next_state] = {a: 0 for a in [UP, DOWN, LEFT, RIGHT]}

            # Update Q-value
            q_table[state][action] = q_table[state][action] + ALPHA * \
                (reward + GAMMA *
                 max(q_table[next_state].values()) - q_table[state][action])

            state = next_state

        # Provide feedback on the progress
        if episode % 4000 == 0:
            print(
                f"Episode {episode}/{EPISODES} completed. Current score: {game.score}")

    # Save the updated Q-table to a file
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(q_table, f)

    # Export the Q-table to a text file using tabulate
    state_descriptions = [
        "Food is to the right",
        "Food is to the left",
        "Food is below",
        "Food is above",
        "Direction is UP",
        "Direction is DOWN",
        "Direction is LEFT",
        "Direction is RIGHT",
        "Danger up",
        "Danger down",
        "Danger left",
        "Danger right"
    ]

    with open('q_table.txt', 'w') as f:
        for state, actions in q_table.items():
            state_description = [f"{desc}: {val}" for desc,
                                 val in zip(state_descriptions, state)]
            table = [[state_description, action, q_value]
                     for action, q_value in actions.items()]
            f.write(tabulate(table, headers=[
                    'State Description', 'Action', 'Q-Value'], tablefmt='grid'))
            f.write('\n\n')


def play_game():
    """
    Play the Snake game using the trained Q-learning model.
    """
    with open('q_table.pkl', 'rb') as f:
        q_table = pickle.load(f)

    game = SnakeGame()
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        state = game.get_state()
        action = max(q_table.get(state, {a: 0 for a in [UP, DOWN, LEFT, RIGHT]}), key=q_table.get(state, {a: 0 for a in [UP, DOWN, LEFT, RIGHT]}).get)
        game.step(action)

        screen.fill(BLACK)
        for x, y in game.snake:
            pygame.draw.rect(screen, GREEN, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, RED, pygame.Rect(game.food[0], game.food[1], CELL_SIZE, CELL_SIZE))

        # Render the score
        score_text = font.render(f'Score: {game.score}', True, WHITE)
        screen.blit(score_text, (10, 10))

        pygame.display.flip()
        clock.tick(30)

if __name__ == '__main__':
    train_q_learning()
    play_game()