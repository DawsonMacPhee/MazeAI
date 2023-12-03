import torch
import random
import numpy
from collections import deque
from maze_game import Game
from model import Linear_QNet, QTrainer
import time

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.005

class Agent():
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.3 # randomness
        self.gamma = 0.5 # discout rate
        self.memory = deque(maxlen=MAX_MEMORY) # double ended queue
        self.model = Linear_QNet()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    # state - [[TILEMAP]]
    def get_state(self, game):
        return numpy.array(game.pathed_tilemap, dtype=int)

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
        final_move = [0, 0, 0, 0]
        if self.n_games < 500 and random.uniform(0, 1) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(numpy.expand_dims(numpy.expand_dims(state, axis=0), axis=0), dtype=torch.float)
            prediction = self.model(state0)
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

    total_collisions = 0
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        state_new, reward, done, total_moves, score = game.play_step(final_move[0], final_move[1], final_move[2], final_move[3])
        if reward == -10:
            total_collisions += 1

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            agent.n_games += 1
            agent.train_long_memory()
            agent.model.save()

            print('Game:', agent.n_games, '| Score:', score, '| Total Moves:', total_moves, '| Collisions:', total_collisions)

            total_collisions = 0

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

if __name__ == '__main__':
    train()