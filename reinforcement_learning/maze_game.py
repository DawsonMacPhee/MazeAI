import pygame
import numpy
import os
import random
from enum import Enum

sourceFileDir = os.path.dirname(os.path.abspath(__file__))

class Game():
    SIZE = (500, 500)
    TILE_SIZE = 45.5

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(Game.SIZE)
        self.textures = [
            pygame.transform.scale(pygame.image.load(os.path.join(sourceFileDir, "resources/wall.jpg")), (Game.TILE_SIZE + 1, Game.TILE_SIZE + 1)),
            pygame.transform.scale(pygame.image.load(os.path.join(sourceFileDir, "resources/road.jpg")), (Game.TILE_SIZE + 1, Game.TILE_SIZE + 1)),
            pygame.transform.scale(pygame.image.load(os.path.join(sourceFileDir, "resources/path.png")), (Game.TILE_SIZE + 1, Game.TILE_SIZE + 1)),
            pygame.transform.scale(pygame.image.load(os.path.join(sourceFileDir, "resources/player.png")), (Game.TILE_SIZE + 1, Game.TILE_SIZE + 1))
        ]

        # Find first unsolved level
        self.level = 1
        for i in range(10000):
            if os.path.isfile(os.path.join(sourceFileDir, "maze_labels/" + str(self.level) + ".maze")):
                self.level += 1
            else:
                self.start_level = self.level
                break
        
        self.load_level(self.level)
        self.reset()

    def load_level(self, level):
        self.tilemap = numpy.loadtxt(os.path.join(sourceFileDir, "../unlabeled_mazes_dataset_1k/matrices/" + str(level) + ".maze"), delimiter=',', dtype=numpy.int8)
        self.map_width = len(self.tilemap[0])
        self.map_height = len(self.tilemap)
        self.min_reward = -0.5 * self.map_width * self.map_height
        self.num_games = 0

    def get_random_start(self):
        if self.num_games < 25:
            rows = range(7, 10)
        elif self.num_games < 50:
            rows = range(4, 10)
        elif self.num_games < 75:
            rows = range(1, 10)
        else:
            return [1, 1]

        free_cells = [[c, r] for r in range(self.map_height) for c in rows if self.tilemap[r,c] == 1.0]
        if [9, 9] in free_cells:
            free_cells.remove([9, 9])
        return random.choice(free_cells)

    def reset(self):
        # Randomize starting pos
        start_pos = self.get_random_start()

        self.path = [start_pos]
        self.pathed_tilemap = self.tilemap.copy().astype(float)
        self.pathed_tilemap[start_pos[1]][start_pos[0]] = 0.3 # Starting position
        self.moves = 0
        self.total_reward = 0


    def run(self):
        self.draw_map()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.play_step(1, 0, 0, 0)
                elif event.key == pygame.K_RIGHT:
                    self.play_step(0, 1, 0, 0)
                elif event.key == pygame.K_DOWN:
                    self.play_step(0, 0, 1, 0)
                elif event.key == pygame.K_LEFT:
                    self.play_step(0, 0, 0, 1)
        self.draw_path()
        pygame.display.update()

    def draw_map(self):
        for row in range(self.map_height):
            for column in range(self.map_width):
                self.screen.blit(self.textures[self.tilemap[row][column]], (column*Game.TILE_SIZE, row*Game.TILE_SIZE))

    def draw_path(self):
        for i, coord in enumerate(self.path):
            if i < len(self.path) - 1:
                self.screen.blit(self.textures[2], (coord[0]*Game.TILE_SIZE, coord[1]*Game.TILE_SIZE))
            else:
                self.screen.blit(self.textures[3], (coord[0]*Game.TILE_SIZE, coord[1]*Game.TILE_SIZE))

    def play_step(self, up, right, down, left):
        self.moves += 1

        next_coord = self.path[len(self.path) - 1].copy()
        if up:
            next_coord[1] -= 1
        elif right:
            next_coord[0] += 1
        elif down:
            next_coord[1] += 1
        elif left:
            next_coord[0] -= 1

        reward = -0.04
        game_over = False
        ignore_move = False
        if next_coord == [9, 9]: # Handle sucessful finish
            reward = 1.0
            game_over = True
            #numpy.savetxt(os.path.join(sourceFileDir, "maze_labels/" + str(self.level) + ".maze"), self.pathed_tilemap, fmt='%d', delimiter=',')
        elif self.tilemap[next_coord[1]][next_coord[0]] == 0: # Handle collision
            reward = -0.75
            ignore_move = True
        elif next_coord in self.path: # Handle repeat path
            reward = -0.25

        if self.total_reward < self.min_reward: # Handle run out of time
            game_over = True

        if not ignore_move:
            last_coord = self.path[len(self.path) - 1]
            self.path.append(next_coord)

            self.pathed_tilemap[last_coord[1]][last_coord[0]] = 0.7
            self.pathed_tilemap[next_coord[1]][next_coord[0]] = 0.3

        self.run()

        # Handles resets if nessesary, after rendering the finished display
        moves = self.moves
        if game_over:
            self.num_games += 1
            self.reset()

        # DISABLING NEXT LEVEL FOR TESTING
        #if reward == 1.0:
            #self.level += 1
            #self.load_level(self.level)

        self.total_reward += reward
        return self.pathed_tilemap.copy(), reward, game_over, moves, self.level - self.start_level