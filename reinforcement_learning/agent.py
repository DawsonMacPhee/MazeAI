import torch
import random
import numpy
from collections import deque
from maze_game import Game, Direction

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent():
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0 # discout rate
        self.memory = deque(maxlen=MAX_MEMORY) # double ended queue
        self.model = None #TODO
        self.trainer = None #TODO

    # state - [UP, RIGHT, DOWN, LEFT, TILEMAP]
    def get_state(self, game):
        dir_u = game.direction = Direction.UP
        dir_r = game.direction = Direction.RIGHT
        dir_d = game.direction = Direction.DOWN
        dir_l = game.direction = Direction.LEFT

        return numpy.array([dir_u, dir_r, dir_d, dir_l, game.pathed_tilemap], dtype=int)

    def remember(self, state, action, reward, next_sate, done):
        self.memory.append((state, action, reward, next_sate, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / explotation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move[0], final_move[1], final_move[2])
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            agent.n_games += 1
            agent.train_long_memory()

            if score > record
                record = score
                # agent.model.save()

            print('Game:', agent.n_games, 'Score:', score, 'Record:', record)

            # TODO: plot

if __name__ == '__main__':
    train()