import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class SingleStepNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Sequential(
            # First convulation block
            nn.Conv2d(1, 32, 3, padding = 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride = 1, padding = 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),            
            nn.BatchNorm2d(64),
            # Second convulation block
            nn.Conv2d(64, 128, 3, stride = 1, padding = 1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, stride = 1, padding = 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),       
            # Third convulation block
            # nn.Conv2d(128, 256, 3, stride = 1, padding = 1),
            # nn.LeakyReLU(),
            # nn.Conv2d(256, 256, 3, stride = 1, padding = 1),
            # nn.LeakyReLU(),
            # nn.MaxPool2d(2,2),
            # nn.BatchNorm2d(256)
        )
        self.flatten = nn.Flatten(1)
        self.connected = nn.Sequential(
            # Fully connected layer
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=1)
        )
        


    def forward(self, input):   
        output = self.conv(input)
        output = self.flatten(output)
        output = self.connected(output)
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
        loss = self.criterion(moves_pred,moves)
        loss.backward()
        self.optimizer.step()
        return moves_pred