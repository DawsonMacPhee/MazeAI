import pygame
import numpy
import os

sourceFileDir = os.path.dirname(os.path.abspath(__file__))

class Game():
    SIZE = (800, 800)
    TILE_SIZE = 19.5

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(Game.SIZE)
        self.textures = [
            pygame.image.load(os.path.join(sourceFileDir, "resources/wall.jpg")),
            pygame.image.load(os.path.join(sourceFileDir, "resources/road.jpg")),
            pygame.image.load(os.path.join(sourceFileDir, "resources/gem.png"))
        ]

        # Find first unsolved level
        self.level = 0
        for i in range(10000):
            if os.path.isfile(os.path.join(sourceFileDir, "maze_labels/" + str(self.level) + ".maze")):
                self.level += 1
            else:
                break
            
        self.load_level(self.level)
        self.reset()

    def load_level(self, level):
        self.tilemap = numpy.loadtxt(os.path.join(sourceFileDir, "../unlabeled_mazes_dataset_10k/matrices/" + str(level) + ".maze"), delimiter=',', dtype=numpy.int8)
        self.map_width = len(self.tilemap[0])
        self.map_height = len(self.tilemap)

    def reset(self):
        self.path = [[0, 1]]
        self.direction = [0, 1, 0, 0]
        self.pathed_tilemap = self.tilemap.copy()
        self.pathed_tilemap[1][0] = 2

    def run(self):
        while True:
            self.draw_map()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.play_step(1, 0, 0)
                    elif event.key == pygame.K_LEFT:
                        self.play_step(0, 1, 0)
                    elif event.key == pygame.K_RIGHT:
                        self.play_step(0, 0, 1)
            self.draw_path()
            pygame.display.update()

    def draw_map(self):
        for row in range(self.map_height):
            for column in range(self.map_width):
                self.screen.blit(self.textures[self.tilemap[row][column]], (column*Game.TILE_SIZE, row*Game.TILE_SIZE))

    def draw_path(self):
        for coord in self.path:
            self.screen.blit(self.textures[2], (coord[0]*Game.TILE_SIZE, coord[1]*Game.TILE_SIZE))

    def go_straight(self, nextCoord):
        if self.direction[0]:
            nextCoord[1] -= 1
        elif self.direction[1]:
            nextCoord[0] += 1
        elif self.direction[2]:
            nextCoord[1] += 1
        elif self.direction[3]:
            nextCoord[0] -= 1

    def turn_left(self, nextCoord):
        if self.direction[0]:
            nextCoord[0] -= 1
            self.direction = [0, 0, 0, 1]
        elif self.direction[1]:
            nextCoord[1] -= 1
            self.direction = [1, 0, 0, 0]
        elif self.direction[2]:
            nextCoord[0] += 1
            self.direction = [0, 1, 0, 0]
        elif self.direction[3]:
            nextCoord[1] += 1
            self.direction = [0, 0, 1, 0]

    def turn_right(self, nextCoord):
        if self.direction[0]:
            nextCoord[0] += 1
            self.direction = [0, 1, 0, 0]
        elif self.direction[1]:
            nextCoord[1] += 1
            self.direction = [0, 0, 1, 0]
        elif self.direction[2]:
            nextCoord[0] -= 1
            self.direction = [0, 0, 0, 1]
        elif self.direction[3]:
            nextCoord[1] -= 1
            self.direction = [1, 0, 0, 0]

    def play_step(self, forward, left, right):
        nextCoord = self.path[len(self.path) - 1].copy()
        if forward:
            self.go_straight(nextCoord)
        elif left:
            self.turn_left(nextCoord)
        elif right:
            self.turn_right(nextCoord)

        self.path.append(nextCoord)
        self.pathed_tilemap[nextCoord[1]][nextCoord[0]] = 2

        if nextCoord == [40, 39]:
            reward = 10
            numpy.savetxt(os.path.join(sourceFileDir, "maze_labels/" + str(self.level) + ".maze"), self.pathed_tilemap, fmt='%d', delimiter=',')
            self.level += 1
            self.load_level(self.level)
            self.reset()
        elif self.tilemap[nextCoord[1]][nextCoord[0]] == 0:
            reward = -10
            self.reset()
        else:
            reward = 0

        print(reward)

game = Game()
game.run()