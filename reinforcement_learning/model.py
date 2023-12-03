import torch
import os

class Linear_QNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, 3, padding = 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, stride = 1, padding = 1)
        self.conv4 = torch.nn.Conv2d(128, 128, 3, stride = 1, padding = 1)
        self.conv5 = torch.nn.Conv2d(128, 256, 3, stride = 1, padding = 1)
        self.conv6 = torch.nn.Conv2d(256, 256, 3, stride = 1, padding = 1)

        self.linear1 = torch.nn.Linear(256, 1024)
        self.linear2 = torch.nn.Linear(1024, 512)
        self.linear3 = torch.nn.Linear(512, 4)

        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, input):
        # First Convolution Block
        output = self.conv1(input)

        output = torch.nn.functional.leaky_relu(output)
        output = self.conv2(output)

        output = torch.nn.functional.leaky_relu(output)
        output = self.pool(output)

        # Second Convolution Block
        output = self.conv3(output)

        output = torch.nn.functional.leaky_relu(output)
        output = self.conv4(output)

        output = torch.nn.functional.leaky_relu(output)
        output = self.pool(output)

        # Third Convolution Block
        output = self.conv5(output)

        output = torch.nn.functional.leaky_relu(output)
        output = self.conv6(output)

        output = torch.nn.functional.leaky_relu(output)
        output = self.pool(output)

        # Flatten
        output = torch.flatten(output, 1)

        # Fully Connected Layers
        output = self.linear1(output)

        output = torch.nn.functional.leaky_relu(output) 
        output = self.linear2(output)

        output = torch.nn.functional.leaky_relu(output) 
        output = self.linear3(output)

        return output

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer():
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        print_info = False

        if len(state.shape) == 2:
            print_info = True
            state = torch.unsqueeze(torch.unsqueeze(state, 0), 0)
            next_state = torch.unsqueeze(torch.unsqueeze(next_state, 0), 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        else:
            state = torch.unsqueeze(state, 1)
            next_state = torch.unsqueeze(next_state, 1)

        # predict Q values with current state
        pred = self.model(state)
        #if print_info:
            #print(state)
            #print(pred)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                next_pred = self.model(torch.unsqueeze(next_state[idx], 0))
                Q_new = reward[idx] + self.gamma * torch.max(next_pred)

            target[idx][torch.argmax(action[idx]).item()] = Q_new
            #if print_info:
                #print(idx, target[idx])

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()