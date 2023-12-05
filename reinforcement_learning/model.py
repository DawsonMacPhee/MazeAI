import torch
import os

class Linear_QNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.set_printoptions(linewidth=400)

        self.linear1 = torch.nn.Linear(121, 512)
        self.linear2 = torch.nn.Linear(512, 128)
        self.linear3 = torch.nn.Linear(128, 4)

    def forward(self, input):
        # Flatten
        output = torch.flatten(input, 1)

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
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            print_info = True

        # predict Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]

            if not done[idx]:
                next_pred = self.model(torch.unsqueeze(next_state[idx], 0))
                Q_new = reward[idx] + self.gamma * torch.max(next_pred)

                #if print_info:
                    #print("~~~~~~~~~~")
                    #print(state[idx])
                    #print(pred[idx])
                    #print(next_state[idx])
                    #print(next_pred)

            target[idx][torch.argmax(action[idx]).item()] = Q_new

            #if print_info:
                #print(target[idx])

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()