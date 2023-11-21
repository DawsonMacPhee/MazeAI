import model as m
import os
import numpy
import torch
import argparse, sys

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-directory', help='Path to Directory')
    
    args = parser.parse_args()

    if args.directory is None:
        print("Please provide all arguments:")
        parser.print_help()
        exit(1)
        
    epochs = 1
    
    train_model(args.directory,epochs_num = epochs)
    print(f"Finished training {epochs} epochs.")
    return

def train_model(directory,epochs_num = 3):
    model = m.SingleStepNet.load()
    if model is None:
        model = m.SingleStepNet()
    
    trainer = m.StepTrainer(model,0.25)
    
    data_set = load_labels(directory)   
    
    for epoch in range (0,epochs_num):
        for batch in data_set:
            mazes, moves = batch
            trainer.train_step(mazes, moves)           
    
    
    model.save()


def load_labels(directory,batch_size = 200):
    if (batch_size < 0):
        print("Batch size must be greater than zero")
    
    cwd_path = './'
    if directory.startswith(cwd_path):
        dataset_directory = directory
    else:
        dataset_directory = cwd_path + directory
    
    steps_dir  = f'{dataset_directory}/steps/'

    data = []
    batch = []
    
    maze_num = 0
    source_folder = f"{steps_dir}/maze_{maze_num}"
    while os.path.isfile(f"{source_folder}{maze_num}.maze"):
        step_num = 0
        step_maze_file = f"{source_folder}/{maze_num}_{step_num}.maze"
        step_label_file = f"{source_folder}/{maze_num}_{step_num}.move"
        while os.path.isfile(step_maze_file) and os.path.isfile(step_label_file):
            step_maze = torch.from_numpy(numpy.loadtxt(step_maze_file , delimiter=',', dtype=numpy.int8))
            step_label = torch.from_numpy(numpy.loadtxt(step_label_file, delimiter=',', dtype=numpy.int8))
            batch.append([step_maze,step_label])

        maze_num += 1
        
        if (maze_num % batch_size == 0):
            data.append(batch)
            batch = []
          
        source_folder = f"{steps_dir}/maze_{maze_num}"
        
    if len(batch) > 0:
        data.append(batch)
        
    return data
          
if __name__ == "__main__":
    main()