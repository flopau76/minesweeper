"""
NEURAL NETWORK SOLVER
-------------------------------
working principle:
 - we make the model choose a cell from the border
    - every cell in the border is embeded using it's 5x5 neighborhood
    - the model learn using a QLearning algorithm
 
-quantify:
    -success rate
    -computation time
"""
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import env

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

from tqdm import tqdm
import itertools

import time
import datetime

import os


# default parameters
# ------------------
NN_L1 = 128 # neurons in the first layer
NN_L2 = 128 # neurons in the second layer
NN_L3 = 128 # neurons in the third layer
NN_L4 = 128 # neurons in the fourth layer

MIN_PROBA = 1e-5 # probability under which select a random cell instead of border cell

GUESS_REWARD = -1 # randomly guessed the next play
CONTINUE_REWARD = 1 # the reward gained each turn
LOSS_REWARD = -50   # the reward if we lose the game
WIN_REWARD = 50    # the reward if we won the game

EPSILON_START = 0.95   # the initial epsilon value
EPSILON_MIN = 0.01    # the minimum epsilon value
EPSILON_DECAY = 0.99975 # the decay epsilon 

LR_LAST_EPOCH = -1
LR_START = 0.001
LR_MIN = 1e-6
LR_DECAY = 0.9999

NUM_EPISODES = 1000 # the number of episode per training
GAMMA = 0.99 # the gamma value for QLearning
TARGET_NETWORK_SYNC_PERIOD = 5
BATCH_SIZE = 64

REPLAY_BUFFER_CAPACITY = 1000*BATCH_SIZE
MIN_REPLAY_BUFFER_SIZE = 10*BATCH_SIZE

NUM_TRAININGS = 1 # the number of trainings
NUM_TESTS = 100 # the number of tests

# ------------------

class MLPModel(nn.Module):
    '''
        Implements a multi layer perceptron model for minesweeper

        Properties
        ----------
        layer1: The first linear layer followed by ReLU activation
        layer2: The second linear layer followed by ReLU activation
        layer3: The third linear layer followed by ReLU activation
        layer4: The fourth linear layer followed by ReLU activation
        layer5: The fifth linear layer followed by Sigmoid activation
    '''
    def __init__(self, n_observations=25, nn_l1=NN_L1, nn_l2=NN_L2, nn_l3=NN_L3, nn_l4=NN_L4):
        super(MLPModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(n_observations, nn_l1),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(nn_l1, nn_l2),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(nn_l2, nn_l3),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(nn_l3, nn_l4),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Linear(nn_l4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''Forward pass for the Neural Network'''
        batch_size = x.size(0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x.squeeze(0)


class EpsilonGreedy():
    def __init__(self, envWrapper, q_network, epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY):
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.envWrapper = envWrapper
        self.q_network = q_network
        self.device = self.envWrapper.device

    def __call__(self, state):
        if np.random.random() < self.epsilon:
            action = self.envWrapper.get_random_cell()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            uncovered_cells = self.envWrapper.get_uncovered_cells()
            q_values = self.q_network(state_tensor).detach().cpu().numpy().flatten()
            q_values[uncovered_cells] = -np.inf  # Set invalid actions to -inf
            action = np.argmax(q_values)
        return action

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer: torch.optim.Optimizer, lr_decay: float = LR_DECAY, last_epoch: int = LR_LAST_EPOCH, min_lr: float = LR_MIN):
        self.min_lr = min_lr
        super().__init__(optimizer, lr_decay, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr) for base_lr in self.base_lrs]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)



class MinesweeperEnvWrapper():
    def __init__(self, difficulty="beginner"):
        self.env = env.Minesweeper(difficulty)
        self.state = self.preprocess_state()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.status = 0
    
    def reset(self):
        self.env.reset()
        self.state = self.preprocess_state()
        return self.state
    
    def step(self, action):
        bordering_cells = self.get_bordering_cells()
        random = action in bordering_cells
        status = self.env.take_action(action, uncover_neighbors=True)   
        self.state = self.preprocess_state()
        next_state = self.state
        self.status = status
        done = bool(status)
        reward = self.reward(status, random)
        return next_state, reward, done
    
    def reward(self, status, random):
        if status == 0 and not random:  # if the model uncovers a safe tile
            return CONTINUE_REWARD
        elif status == 0:
            return GUESS_REWARD
        elif status == 1:  # if the model uncovers a mine
            return LOSS_REWARD
        elif status == 2:  # if the model wins
            return WIN_REWARD
        else:
            raise ValueError("Invalid status")

    def get_covered_cells(self):
        return self.env.get_covered_cells_int(covered=True)

    def get_uncovered_cells(self):
        return self.env.get_covered_cells_int(covered=False)

    def get_random_cell(self):
        ''' Chose a random cell to uncover'''
        all_covered_cells = self.get_covered_cells()
        action = random.choice(all_covered_cells)
        return action
    
    def preprocess_state(self):
        """Transform the observed grid into a 9xheightxwidth array which will be used as input to the CNN.

        Returns:
           np.ndarray: Tensor of shape (24, height, width) where each channel represents a different tile value.
        """
        return self.get_state()
    
    def get_5x5_neighborhood(self, cell):
        '''Get a list of exactly 24 neighbors id (put a tuple (-1,-1) if it is not a valid cell)'''
        if type(cell) == tuple:
            i, j = cell
        else:
            i,j = env.int2tupple(a, self.env.n_cols)    
        adj = [
                (i-2, j-2), (i-2, j-1), (i-2, j), (i-2, j+1), (i-2, j+2),
                (i-1, j-2), (i-1, j-1), (i-1, j), (i-1, j+1), (i-1, j+2),
                (i, j-2), (i, j-1), (i,j), (i, j+1), (i, j+2),
                (i+1, j-2), (i+1, j-1), (i+1, j), (i+1, j+1), (i+1, j+2),
                (i+2, j-2), (i+2, j-1), (i+2, j), (i+2, j+1), (i+2, j+2)
            ]
        res = []
        for a, b in adj:
            if a >= 0 and a < self.env.n_rows and b >= 0 and b < self.env.n_cols:
                res.append((a,b))
            else:
                res.append((-1,-1))
        return res

    def convert_cells_to_feature(self, cell_list):
        '''
        get the features from the list of cell
        params:
            cell_list: a list of 24 neighbors
        return 
            the observations of the cells (or -5 if the cell doesn't exist)
        '''
        arr = np.array([self.env.O[i,j] if (not i==-1 and not j==-1) else -5 for (i,j) in cell_list])
        return arr

    def get_bordering_cells(self):
        '''Get a mask for the bordering cells of uncovered cells'''
        border = []
        uncovered = self.env.get_covered_cells(covered=False)
        for cell in uncovered:
            neigbhors = self.env.get_neighbors(cell)
            for (i,j) in neigbhors:
                if self.env.O[i,j] == -2: # covered tile
                    border.append(env.tupple2int((i,j), self.env.n_cols))
        
        return border

    def get_state(self):
        '''
        get the current state of the environment
        return:
            bordering_cells: the cells we want to uncover
            state: the features to represent these cells
        '''
        all_cells = [(i,j) for i in range(self.env.n_rows) for j in range(self.env.n_cols)]
        state = np.array([self.convert_cells_to_feature(self.get_5x5_neighborhood(cell)) for cell in all_cells])
        state = (state + 5.) / (8. + 5.) # [0,1]
        return state

class MLP_Solver():
    def __init__(self, envWrapper: MinesweeperEnvWrapper, q_network: MLPModel, target_network: MLPModel):
        self.envWrapper = envWrapper
        self.q_network = q_network
        self.target_network = target_network
        self.device = self.envWrapper.device

    def train_agent(
            self, 
            optimizer: torch.optim.Optimizer, 
            loss_fn: callable, 
            policy: EpsilonGreedy, 
            lr_scheduler: _LRScheduler, 
            replay_buffer: ReplayBuffer,
            num_episodes: int = NUM_EPISODES, 
            gamma: float = GAMMA, 
            target_network_sync_period: int = TARGET_NETWORK_SYNC_PERIOD,
            print_every: int = 500,
            batch_size: int = BATCH_SIZE
    ) -> tuple[list[float]]:
        episode_reward_list = []
        episode_loss_list = []
        mean_reward = 0
        mean_loss = 0

        for i in range(1, num_episodes):
            state = self.envWrapper.reset()
            episode_reward = 0
            episode_loss = 0
            time_episode = 0
            done = False

            while not done:
                time_episode += 1
                action = policy(state)
                next_state, reward, done = self.envWrapper.step(action)
                episode_reward += reward

                if len(replay_buffer.buffer) >= replay_buffer.buffer.maxlen:
                    replay_buffer.buffer.pop()

                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state

            if len(replay_buffer) >= MIN_REPLAY_BUFFER_SIZE:  # Start training after the replay buffer has been filled with enough samples
                replayed_states, replayed_actions, replayed_rewards, replayed_next_states, replayed_dones = replay_buffer.sample(batch_size)

                replayed_states_tensor = torch.tensor(replayed_states, dtype=torch.float32, device=self.device)
                replayed_actions_tensor = torch.tensor(replayed_actions, dtype=torch.int64, device=self.device)
                replayed_rewards_tensor = torch.tensor(replayed_rewards, dtype=torch.float32, device=self.device)
                replayed_next_states_tensor = torch.tensor(replayed_next_states, dtype=torch.float32, device=self.device)
                replayed_dones_tensor = torch.tensor(replayed_dones, dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    replayed_next_q_values = self.target_network(replayed_next_states_tensor).squeeze()
                replayed_targets_tensor = replayed_rewards_tensor + gamma * torch.max(replayed_next_q_values, dim=1).values * (1-replayed_dones_tensor)
                replayed_q_values = self.q_network(replayed_states_tensor).squeeze()

                replayed_results = np.zeros(BATCH_SIZE)
                for j in range(BATCH_SIZE):
                    replayed_results[j] = replayed_q_values[j][replayed_actions_tensor[j]]
                replayed_results_tensor = torch.tensor(replayed_results, dtype=torch.float32, device=self.device, requires_grad=True)
                loss = loss_fn(replayed_results_tensor, replayed_targets_tensor)

                episode_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            
                mean_loss += loss.item() / print_every
            mean_reward += episode_reward / print_every
            
            if i % target_network_sync_period == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            if i % print_every == 0:
                print(f"Episode: {i}, Mean reward: {mean_reward}, Epsilon: {policy.epsilon}, LR: {lr_scheduler.get_lr()[0]}, Mean loss: {mean_loss}")
                mean_reward = 0
                mean_loss = 0
            
            episode_reward_list.append(episode_reward)
            episode_loss_list.append(episode_loss / time_episode)
            policy.decay_epsilon()

        return episode_reward_list, episode_loss_list


    def test_agent(self, num_tests: int = NUM_TESTS) -> list[int]:  
        episode_reward_list = []
        nb_wins = 0

        for _ in range(num_tests):
            state = self.envWrapper.reset()
            done = False
            episode_reward = 0

            while not done:
                # Convert the state to a PyTorch tensor and add a batch dimension (unsqueeze)
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                    q_values = self.q_network(state_tensor).detach().cpu().numpy().flatten()

                uncovered_cells = self.envWrapper.get_uncovered_cells()
                q_values[uncovered_cells] = -np.inf  # Set invalid actions to -inf

                action = np.argmax(q_values)  # Choose the action with the highest Q-value - greedy policy           
                next_state, reward, done = self.envWrapper.step(action)               
                state = next_state
                episode_reward += reward
            
            if self.envWrapper.status == 2:
                nb_wins += 1

            episode_reward_list.append(episode_reward)
            # print(f"Episode reward: {episode_reward}")
        
        print(f"MLP strategy: {nb_wins / num_tests} wins on average ({nb_wins} / {num_tests})")
        return episode_reward_list



def load_or_train_model(solver, force_retrain=False, verbose=0):
    '''Helper function to check if the model must be trained or loaded from a file'''
    current_time = datetime.datetime.now()
    model_file = current_time.strftime("%Y-%m-%d_%H-%M-%S.pth")

    if (not force_retrain) and (os.path.isfile(model_file)):
        print("Loading pre-trained model...")
        solver.model = torch.load(model_file, map_location=solver.device)
        return
    
    print("Training new model...")
    optimizer = torch.optim.Adam(q_network.parameters(), lr=LR_START)
    loss_fn = nn.MSELoss()
    epsilon_greedy = EpsilonGreedy(envWrapper=envWrapper, q_network=q_network)
    lr_scheduler = MinimumExponentialLR(optimizer)
    buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY)

    episode_reward_list, episode_loss_list = solver.train_agent(
        optimizer=optimizer, 
        loss_fn=loss_fn, 
        policy=epsilon_greedy, 
        lr_scheduler=lr_scheduler, 
        replay_buffer=buffer
    )
    torch.save(solver.q_network, model_file)



if __name__ == "__main__":
    envWrapper = MinesweeperEnvWrapper(difficulty="beginner")
    q_network = MLPModel().to(envWrapper.device)
    target_network = MLPModel().to(envWrapper.device)

    solver = MLP_Solver(envWrapper, q_network, target_network)

    # train the model multiple times to get the time in average
    sum_times = 0
    for _ in range(NUM_TRAININGS):
        tic = time.perf_counter()
        load_or_train_model(solver=solver, force_retrain=True)
        toc = time.perf_counter()
        cur_time = toc - tic
        print(f"Trained the model in {toc - tic:0.4f} seconds")
        sum_times += cur_time
    print(f"Training {NUM_TRAININGS} models in {(sum_times / NUM_TRAININGS):0.4f} seconds on average")
    
    # test the model multiple times to get the winrate
    nb_wins = 0
    over = solver.test_agent()    
    print()