import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class SingleStepNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride = 1, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, 3, stride = 1, padding = 1)

        self.linear1 = nn.Linear(256, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 4)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, input):
        # First Convolution Block
        output = self.conv1(input)

        output = nn.functional.leaky_relu(output)
        output = self.conv2(output)

        output = nn.functional.leaky_relu(output)
        output = self.pool(output)

        # Second Convolution Block
        output = self.conv3(output)

        output = nn.functional.leaky_relu(output)
        output = self.conv4(output)

        output = nn.functional.leaky_relu(output)
        output = self.pool(output)

        # Third Convolution Block
        output = self.conv5(output)

        output = nn.functional.leaky_relu(output)
        output = self.conv6(output)

        output = nn.functional.leaky_relu(output)
        output = self.pool(output)

        # Flatten
        output = torch.flatten(output, 1)

        # Fully Connected Layers
        output = self.linear1(output)

        output = nn.functional.leaky_relu(output) 
        output = self.linear2(output)

        output = nn.functional.leaky_relu(output) 
        output = self.linear3(output)

        return output

    def save(self, file_name='steps_model.pth'):
        model_folder_path = './supervised_learning/model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
    def load(file_name='steps_model.pth'):
        model_file =  f'./supervised_learning/model/{file_name}'
        if not os.path.exists(model_file):
            return None
        model = SingleStepNet()
        model.load_state_dict(torch.load(model_file))
        model.eval()
        return model
            
        
class StepTrainer():
    
    def __init__(self, model, lr):
        self.lr = lr
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

    def train_step(self,mazes,moves):
        self.optimizer.zero_grad()
        
        moves_pred = self.model(mazes)
        
        loss = self.criterion(moves, moves_pred)
        loss.backward()
        self.optimizer.step()