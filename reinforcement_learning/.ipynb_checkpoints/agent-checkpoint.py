import torch
import random
import numpy
from collections import deque
from maze_game import Game
from model import Linear_QNet, QTrainer
from IPython import get_ipython
import time
import matplotlib.pyplot as plt

MAX_MEMORY = 1500
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

        plt.ion()

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
            state0 = torch.tensor(numpy.expand_dims(numpy.expand_dims(state, axis=0), axis=1), dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    fig, axes = plt.subplots(nrows=3, ncols=1)
    ax1, ax2, ax3 = axes
    plt.subplots_adjust(hspace=0.45, right=0.84)
    plt.show(block=False)

    plot_collisions = [0]
    plot_backtracks = [0]
    plot_total_moves = [0]
    plot_wins = [0]
    plot_n_games = [0]
    plot_times = [0]

    agent = Agent()
    game = Game()

    win_count = 0
    total_time = 0

    total_collisions = 0
    total_backtracks = 0
    last_time = 0

    start_time = time.time()
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
            total_time = time.time() - start_time
            agent.n_games += 1
            agent.model.save()

            win = False
            if reward == 1.0:
                win_count += 1

            print('Game:', agent.n_games, '| Score:', score, '| Total Moves:', total_moves, '| Collisions:', total_collisions, '| Backtracks:', total_backtracks, '| Wins:', win_count, '| Delta Time:', round((total_time - last_time) / 60, 2), '| Total Time:', round(total_time / 60, 2))

            plot_collisions.append(total_collisions)
            plot_backtracks.append(total_backtracks)
            plot_total_moves.append(total_moves)
            plot_wins.append(win_count)
            plot_n_games.append(agent.n_games)
            plot_times.append(round((total_time - last_time) / 60, 2))

            ax1.clear()
            ax1.set_title('Algorithm Effectiveness')
            ax1.set_xlabel('Game Number')
            ax1.set_ylabel('Number of Actions')
            ax1.plot(plot_n_games, plot_total_moves, label="Total Moves")
            ax1.plot(plot_n_games, plot_collisions, label="Collisions")
            ax1.plot(plot_n_games, plot_backtracks, label="Backtracks")
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            ax2.clear()
            ax2.set_title('Win Rate')
            ax2.set_xlabel('Game Number')
            ax2.set_ylabel('Number of Winning Games')
            ax2.plot(plot_n_games, plot_wins)

            ax3.clear()
            ax3.set_title('Elaspsed Time')
            ax3.set_xlabel('Game Number')
            ax3.set_ylabel('Time to Complete (minutes)')
            ax3.plot(plot_n_games, plot_times)

            total_collisions = 0
            total_backtracks = 0
            last_time = total_time

if __name__ == '__main__':
    train()