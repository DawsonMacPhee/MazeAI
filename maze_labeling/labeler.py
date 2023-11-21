import math
import time
from sys import argv
import argparse, sys
import numpy
from PIL import Image

from hashlib import sha256
from pathlib import Path

import sys
import os

from pathfinder import PathFinder

#sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/.." + directory)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-directory', help='Path to Directory')
    
    args = parser.parse_args()

    if args.directory is None:
        print("Please provide all arguments:")
        parser.print_help()
        exit(1)

    
    
    # number of cells in a single row
    createLabeledData(args.directory)
    # number of mazes to generate in the dataset
  

    print(f'Labeled Data Created')
    
def createLabeledData(directory):
    cwd_path = './'
    if directory.startswith(cwd_path):
        dataset_directory = directory
    else:
        dataset_directory = cwd_path + directory
    
    matrices_dir = f'{dataset_directory}/matrices/'
    full_path_dir  = f'{dataset_directory}/full_path/'
    steps_dir  = f'{dataset_directory}/steps/'

    
    
    Path(full_path_dir).mkdir(exist_ok=True)
    print(steps_dir)
    Path(steps_dir).mkdir(exist_ok=True)
    
    i = 0
    source_maze = f"{matrices_dir}{i}.maze"
    while os.path.isfile(f"{matrices_dir}{i}.maze"):
        maze = numpy.loadtxt(source_maze, delimiter=',', dtype=numpy.int8)
        pathfinder = PathFinder(maze)
        path = pathfinder.findPath()
        found_path = numpy.zeros((len(maze),len(maze[0])), dtype=numpy.int8)
        
        Path(f"{steps_dir}/maze_{i}/").mkdir(exist_ok=True)
        current_pos = pathfinder.start_pos.position
        pathed_maze = maze.copy()
        for index, cord in enumerate(path[1:]):
            found_path[cord[0]][cord[1]] = 1
            if cord[0] < current_pos[0]:
                next_move = numpy.asarray((1,0,0,0))
            elif cord[1] > current_pos[1]:
                next_move = numpy.asarray((0,1,0,0))
            elif cord[0] > current_pos[0]:
                next_move = numpy.asarray((0,0,1,0))
            elif cord[1] < current_pos[1]:
                next_move = numpy.asarray((0,0,0,1))
            
            numpy.savetxt(f"{steps_dir}/maze_{i}/{i}_{index}.maze", pathed_maze, fmt='%d', delimiter=',')
            numpy.savetxt(f"{steps_dir}/maze_{i}/{i}_{index}.move", next_move, fmt='%d', delimiter=',')
            
            pathed_maze[current_pos[0]][current_pos[1]] = 2
            pathed_maze[cord[0]][cord[1]] = 3
            current_pos = cord
                           

        goal = pathfinder.goal_pos.position
        found_path[goal[0]][goal[1]] = 1      
        numpy.savetxt(f"{full_path_dir}/{i}.path", found_path, fmt='%d', delimiter=',')
        
        i += 1
        source_maze = f"{matrices_dir}{i}.maze"
        

if __name__ == "__main__":
    main()