import torch
import random
import numpy
from collections import deque
from maze_game import Game
from model import Linear_QNet, QTrainer
import time

MAX_MEMORY = 2000
BATCH_SIZE = 100
LR = 0.001

class Agent():
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.1 # randomness
        self.gamma = 0.95 # discout rate
        self.memory = deque(maxlen=MAX_MEMORY) # double ended queue
        self.model = Linear_QNet()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    # state - [[TILEMAP]]
    def get_state(self, game):
        return numpy.array(game.pathed_tilemap, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        for i in range(8):
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state):
        # random moves: tradeoff exploration / explotation
        final_move = [0, 0, 0, 0]
        if self.n_games < 1500 and random.uniform(0, 1) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(numpy.expand_dims(state, axis=0), dtype=torch.float)
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
    total_backtracks = 0
    win_count = 0
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        state_new, reward, done, total_moves, score = game.play_step(final_move[0], final_move[1], final_move[2], final_move[3])
        if reward == -0.75:
            total_collisions += 1
        elif reward == -0.25:
            total_backtracks += 1

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        # train model
        agent.train_model()

        if done:
            agent.n_games += 1
            agent.model.save()

            win = False
            if reward == 1.0:
                win_count += 1

            print('Game:', agent.n_games, '| Score:', score, '| Total Moves:', total_moves, '| Collisions:', total_collisions, '| Backtracks:', total_backtracks, '| Wins:', win_count)

            total_collisions = 0
            total_backtracks = 0

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

if __name__ == '__main__':
    train()