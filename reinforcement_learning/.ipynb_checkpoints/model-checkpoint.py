import torch
import os
import numpy

class Linear_QNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.set_printoptions(linewidth=400)

        self.conv1 = torch.nn.Conv2d(1, 4, 3, padding = 1)
        self.conv2 = torch.nn.Conv2d(4, 8, 3, stride = 1, padding = 1)
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.linear1 = torch.nn.Linear(200, 512)
        self.linear2 = torch.nn.Linear(512, 128)
        self.linear3 = torch.nn.Linear(128, 4)

    def forward(self, input):
        # Convolution Block
        output = self.conv1(input)

        output = torch.nn.functional.leaky_relu(output)
        output = self.conv2(output)

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
        self.training_batch_size = 25
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state_batches = numpy.array_split(state, self.training_batch_size)
        action_batches = numpy.array_split(action, self.training_batch_size)
        reward_batches = numpy.array_split(reward, self.training_batch_size)
        next_state_batches = numpy.array_split(next_state, self.training_batch_size)
        done_batches = numpy.array_split(done, self.training_batch_size)

        for i in range(len(state_batches)):
            state = torch.tensor(state_batches[i], dtype=torch.float)
            state = torch.unsqueeze(state, 1)
            action = torch.tensor(action_batches[i], dtype=torch.long)
            reward = torch.tensor(reward_batches[i], dtype=torch.float)
            next_state = torch.tensor(next_state_batches[i], dtype=torch.float)
            next_state = torch.unsqueeze(next_state, 1)
            done = done_batches[i]

            # predict Q values with current state
            pred = self.model(state)
            target = pred.clone()

            for idx in range(len(done)):
                Q_new = reward[idx]

                if not done[idx]:
                    next_pred = self.model(torch.unsqueeze(next_state[idx], 0))
                    Q_new = reward[idx] + self.gamma * torch.max(next_pred)

                target[idx][torch.argmax(action[idx]).item()] = Q_new

            self.optimizer.zero_grad()
            loss = self.criterion(target, pred)
            loss.backward()
            self.optimizer.step()