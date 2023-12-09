import model as m
import os
import numpy
import torch
import argparse, sys
import torchvision.transforms.functional as F

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-directory', help='Path to Directory')
    parser.add_argument('-epochs', help='Number of epochs')
    
    args = parser.parse_args()

    if args.directory is None:
        print("Please provide all arguments:")
        parser.print_help()
        exit(1)
        
    if args.epochs is None:
        epochs = 3
    else:
        epochs = int(args.epochs)
    
    train_model(args.directory,epochs_num = epochs)
    print(f"Finished training {epochs} epochs.")
    return

def train_model(directory,epochs_num = 3):
    model = m.SingleStepNet.load()
    if model is None:
        model = m.SingleStepNet()
    
    trainer = m.StepTrainer(model,0.01)
    
    data_set = load_labels(directory)
    accuracy_list = []
    accuracy = 0
    for epoch in range (0,epochs_num):
        correct = 0
        total = 0
        for batch in data_set:
            mazes, moves = batch
            moves_pred = trainer.train_step(mazes, moves)
            for i, pred in enumerate(moves_pred):
                total += 1
                chosen_move = pred.argmax().item()
                if chosen_move == moves[i].argmax().item():
                    correct += 1              
        if total > 0:
            print(f"Epoch {epoch} complete")  
            accuracy = 100*(correct/total)   
            accuracy_list.append(accuracy)        
            print(f"Accuracy: {accuracy:.2f}%") 
        else:
            print("No Data Found")
            exit(1)    
    
    model.save()


def load_labels(directory,batch_size = 200):
    if (batch_size < 0):
        print("Batch size must be greater than zero")
        return None
    
    cwd_path = './'
    if directory.startswith(cwd_path):
        dataset_directory = directory
    else:
        dataset_directory = cwd_path + directory
    
    steps_dir  = f'{dataset_directory}/steps'

    data = []
    step_mazes = [] 
    step_labels = []
    maze_num = 0
    step_maze_file = f"{steps_dir}/maze_{maze_num}_mazes.npz"
    step_label_file = f"{steps_dir}/maze_{maze_num}_moves.npz"
    while os.path.isfile(step_maze_file) and os.path.isfile(step_label_file):
        step_mazes_npz = numpy.load(step_maze_file)
        step_labels_npz = numpy.load(step_label_file)
        
        for item in step_mazes_npz:
            normalize_maze = normalize(step_mazes_npz[item])
            step_mazes.append(torch.from_numpy(normalize_maze).float())
            step_labels.append(torch.from_numpy(step_labels_npz[item]).float())
          
        maze_num += 1
        if (maze_num % batch_size == 0):
            data.append((torch.unsqueeze(torch.stack(step_mazes),1),torch.stack(step_labels)))
            step_mazes = [] 
            step_labels = [] 
         
        step_maze_file = f"{steps_dir}/maze_{maze_num}_mazes.npz"
        step_label_file = f"{steps_dir}/maze_{maze_num}_moves.npz"
        
    if len(step_mazes) > 0:
        data.append((torch.unsqueeze(torch.stack(step_mazes),1),torch.stack(step_labels)))
        step_mazes = [] 
        step_labels = [] 
    print("Finished Loading Data")    
    return data

def normalize(numpy_maze):
    maze = numpy_maze.astype('float64')
    maze[maze == 1] = 0.75
    maze[maze == 2] = 0.5
    maze[maze == 3] = 0.25
    maze[maze == 4] = 1
    return maze
    
    
          
if __name__ == "__main__":
    main()