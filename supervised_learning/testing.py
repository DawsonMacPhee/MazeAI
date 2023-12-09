import model as m
import os
import numpy
import torch
import argparse, sys
import torchvision.transforms.functional as F
import pygame

sourceFileDir = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-directory', help='Path to Directory')
    parser.add_argument('-time', help='Path to Directory')
    parser.add_argument('-mazes', help='Path to Directory')
    
    args = parser.parse_args()

    if args.directory is None:
        print("Please provide all arguments:")
        parser.print_help()
        exit(1)
        
    if args.time is None:
        time = 200
    else:
        time = int(args.time)
        
    if args.mazes is None:
        maze_num = 10000
    else:
        maze_num = int(args.mazes)

    test_model(args.directory,100,time,maze_num)
    
    return

def test_model(directory,max_runs,time,maze_num):
    model = m.SingleStepNet.load()
    if model is None:
        model = m.SingleStepNet()
    
    game = Game(directory,max_runs,maze_num)
    with torch.no_grad():
        while playTurn(model,game):
            pygame.time.wait(time)
    
    
def playTurn(model,game):
    current_maze = game.tilemap   
    move_chosen = model(torch.unsqueeze(torch.stack((torch.from_numpy(current_maze).float(),),),1))
    move = [0,0,0,0]
    move[torch.argmax(move_chosen).item()] = 1
    return game.play_step(move[0], move[1], move[2], move[3])

def normalize(numpy_maze):
    maze = numpy_maze.astype('float64')
    maze[maze == 1] = 0.75
    maze[maze == 2] = 0.5
    maze[maze == 3] = 0.25
    maze[maze == 4] = 1
    return maze    


class Game():
    SIZE = (500, 500)
    TILE_SIZE = 45.5
    
    def __init__(self,directory,max_moves,maze_num):
        pygame.init()
        self.screen = pygame.display.set_mode(Game.SIZE)
        self.textures = [
            pygame.transform.scale(pygame.image.load("./resources/wall.jpg"), (Game.TILE_SIZE + 1, Game.TILE_SIZE + 1)),
            pygame.transform.scale(pygame.image.load("./resources/road.jpg"), (Game.TILE_SIZE + 1, Game.TILE_SIZE + 1)),
            pygame.transform.scale(pygame.image.load("./resources/path.png"), (Game.TILE_SIZE + 1, Game.TILE_SIZE + 1)),
            pygame.transform.scale(pygame.image.load("./resources/player.png"), (Game.TILE_SIZE + 1, Game.TILE_SIZE + 1)),
            pygame.transform.scale(pygame.image.load("./resources/exit.jpg"), (Game.TILE_SIZE + 1, Game.TILE_SIZE + 1))
        ]    
        self.directory = f'{directory}/matrices'
        self.max_moves = max_moves
        self.level = 0
        self.max_mazes = maze_num
        self.load_level(self.level)
    
    def load_level(self, level):
        maze_file = f"{self.directory}/{level}.maze"
        if os.path.isfile(maze_file) and self.max_mazes > level:
            self.tilemap = normalize(numpy.loadtxt(maze_file, delimiter=',', dtype=numpy.int8))
            self.map_width = len(self.tilemap[0])
            self.map_height = len(self.tilemap)
            self.findStart()
            self.reset()
            return True
        else:
            if level == 0:
                print("No mazes found")
                exit(1)
            else:
                print("Finished playing")
            return False
        
    def reset(self):
        self.path = [self.start_pos,]
        self.pathed_tilemap = self.tilemap.copy()
        self.moves = 0
        
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
                if self.tilemap[i][j] == 0.25:
                    self.start_pos = [i,j]
                    return
    
    def draw_map(self):
        for row in range(self.map_height):
            for column in range(self.map_width):
                if (self.tilemap[row][column] == 0): 
                    self.screen.blit(self.textures[0], (column*Game.TILE_SIZE, row*Game.TILE_SIZE))
                else:
                    self.screen.blit(self.textures[1], (column*Game.TILE_SIZE, row*Game.TILE_SIZE))

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
            
        ignore_move = False
        next_map = False
        if self.tilemap[next_coord[1]][next_coord[0]] == 1: # Handle sucessful finish
            next_map = True
            print("Solved Maze!")
        elif self.moves >= self.max_moves: # Handle run out of time
            print("Ran out of moves")
            ignore_move = True
            next_map = True
        elif self.tilemap[next_coord[1]][next_coord[0]] == 0: # Handle collision
            ignore_move = True
        
        if not ignore_move:
            last_coord = self.path[len(self.path) - 1]
            self.path.append(next_coord)
            self.pathed_tilemap[last_coord[1]][last_coord[0]] = 0.5
            self.pathed_tilemap[next_coord[1]][next_coord[0]] = 0.25
            self.tilemap[next_coord[1]][next_coord[0]] = 0.5
            self.tilemap[last_coord[1]][last_coord[0]] = 0.25
            state_new = self.pathed_tilemap
        else:
            last_coord = self.path[len(self.path) - 1]

            state_new = self.pathed_tilemap
            state_new[last_coord[1]][last_coord[0]] = 0.5
            state_new[next_coord[1]][next_coord[0]] = 0.25

        self.run()

        # Handles resets if nessesary, after rendering the finished display
        if next_map:
            self.level += 1
            if not self.load_level(self.level):
                pygame.quit()
                return False
        return True
        
        
          
if __name__ == "__main__":
    main()
