import numpy
import os
import copy

class PathFinder():
    def __init__(self, maze):
        self.maze = maze
        self.findStart()
        self.map_width = len(self.maze[0])
        self.map_height = len(self.tmaze)
        self.path = [self.start_pos,]
        self.pathed_maze = self.maze.copy()
        self.reset()

    def reset(self):
        self.path = [self.start_pos,]
        self.pathed_maze = self.maze.copy()
        
    def findStart(self):
        for i in range(0,len(self.tilemap)):
            for j in range(0,len(self.tilemap[0])):
                if self.tilemap[i][j] == 3:
                    self.start_pos = [i,j]
                    return