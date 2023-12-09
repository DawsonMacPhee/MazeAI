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
    Path(steps_dir).mkdir(exist_ok=True)
    total_time = 0
    i = 0
    source_maze = f"{matrices_dir}{i}.maze"
    while os.path.isfile(f"{matrices_dir}{i}.maze"):
        maze = numpy.loadtxt(source_maze, delimiter=',', dtype=numpy.int8)
        pathfinder = PathFinder(maze)
        start_time = time.perf_counter_ns()
        path = pathfinder.findPath()
        end_time = time.perf_counter_ns()
        total_time += end_time - start_time
        found_path = numpy.zeros((len(maze),len(maze[0])), dtype=numpy.int8)
        mazes = []
        moves = []
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
            mazes.append(pathed_maze.copy())
            moves.append(next_move.copy())
            
            pathed_maze[current_pos[0]][current_pos[1]] = 2
            pathed_maze[cord[0]][cord[1]] = 3
            current_pos = cord
        
        numpy.savez_compressed(f"{steps_dir}/maze_{i}_mazes",*mazes)
        numpy.savez_compressed(f"{steps_dir}/maze_{i}_moves",*moves)               

        goal = pathfinder.goal_pos.position
        found_path[goal[0]][goal[1]] = 1      
        numpy.savetxt(f"{full_path_dir}/{i}.path", found_path, fmt='%d', delimiter=',')
        
        i += 1
        source_maze = f"{matrices_dir}{i}.maze"
    time_seconds = total_time/1000000000 # divide time by 1 billion to convert to seconds
    print(f"Found the path for {i} mazes in {time_seconds} seconds")
        

if __name__ == "__main__":
    main()
    