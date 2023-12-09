import pygame
import numpy
import os
import model as m

sourceFileDir = os.path.dirname(os.path.abspath(__file__))

class Game():
    SIZE = (500, 500)
    TILE_SIZE = 45.5
    
    def __init__(self,max_moves):
        pygame.init()
        self.screen = pygame.display.set_mode(Game.SIZE)
        self.textures = [
            pygame.transform.scale(pygame.image.load(os.path.join(sourceFileDir, "resources/wall.jpg")), (Game.TILE_SIZE + 1, Game.TILE_SIZE + 1)),
            pygame.transform.scale(pygame.image.load(os.path.join(sourceFileDir, "resources/road.jpg")), (Game.TILE_SIZE + 1, Game.TILE_SIZE + 1)),
            pygame.transform.scale(pygame.image.load(os.path.join(sourceFileDir, "resources/path.png")), (Game.TILE_SIZE + 1, Game.TILE_SIZE + 1)),
            pygame.transform.scale(pygame.image.load(os.path.join(sourceFileDir, "resources/player.png")), (Game.TILE_SIZE + 1, Game.TILE_SIZE + 1)),
            pygame.transform.scale(pygame.image.load(os.path.join(sourceFileDir, "resources/exit.jpg")), (Game.TILE_SIZE + 1, Game.TILE_SIZE + 1))
        ]        
        self.max_moves = max_moves
        self.reset()    
        
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
        
    def findStart(self):
        for i in range(0,len(self.tilemap)):
            for j in range(0,len(self.tilemap[0])):
                if self.tilemap[i][j] == 3:
                    self.start_pos = [i,j]
                    return
    
    def draw_map(self):
        for row in range(self.map_height):
            for column in range(self.map_width):
                if (self.tilemap[row][column] == 3): 
                    self.screen.blit(self.textures[1], (column*Game.TILE_SIZE, row*Game.TILE_SIZE))
                else:
                    self.screen.blit(self.textures[self.tilemap[row][column]], (column*Game.TILE_SIZE, row*Game.TILE_SIZE))

    def draw_path(self):
        for i, coord in enumerate(self.path):
            if i < len(self.path) - 1:
                self.screen.blit(self.textures[2], (coord[0]*Game.TILE_SIZE, coord[1]*Game.TILE_SIZE))
            else:
                self.screen.blit(self.textures[3], (coord[0]*Game.TILE_SIZE, coord[1]*Game.TILE_SIZE))

    def play_step(self, up, right, down, left):
        self.moves += 1
        moves = self.moves

        next_coord = self.path[len(self.path) - 1].copy()
        if up:
            next_coord[1] -= 1
        elif right:
            next_coord[0] += 1
        elif down:
            next_coord[1] += 1
        elif left:
            next_coord[0] -= 1
            
        game_over = False
        ignore_move = False
        if self.tilemap[next_coord[1]][next_coord[0]] == 4: # Handle sucessful finish
            reward = 10
            game_over = True
            numpy.savetxt(os.path.join(sourceFileDir, "maze_labels/" + str(self.level) + ".maze"), self.pathed_tilemap, fmt='%d', delimiter=',')
        elif self.moves >= 500: # Handle run out of time
            ignore_move = True
            game_over = True
        elif self.tilemap[next_coord[1]][next_coord[0]] == 0: # Handle collision
            reward = -10
            ignore_move = True