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
        self.tilemap = numpy.loadtxt(os.path.join(sourceFileDir, "../unlabeled_mazes_dataset_10k/matrices/0.maze"), delimiter=',', dtype=numpy.int8)
        self.mapwidth = len(self.tilemap[0])
        self.mapheight = len(self.tilemap)

        self.path = [[0, 1]]
        self.direction = [0, 1, 0, 0]

    def reset(self):
        self.path = [[0, 1]]
        self.direction = [0, 1, 0, 0]

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
        for row in range(self.mapheight):
            for column in range(self.mapwidth):
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
        self.path.append(nextCoord)

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
        self.path.append(nextCoord)

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
        self.path.append(nextCoord)

    def play_step(self, forward, left, right):
        nextCoord = self.path[len(self.path) - 1].copy()
        if forward:
            self.go_straight(nextCoord)
        elif left:
            self.turn_left(nextCoord)
        elif right:
            self.turn_right(nextCoord)

        if nextCoord == [40, 39]:
            reward = 10
            self.reset()
        elif self.tilemap[nextCoord[1]][nextCoord[0]] == 0:
            reward = -10
            self.reset()
        else:
            reward = 0

        print(reward)

game = Game()
game.run()