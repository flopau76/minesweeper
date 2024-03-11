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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import env

import torch
import torch.nn as nn

from tqdm import tqdm
import itertools

import time

import os


# default parameters
# ------------------
NN_L1 = 128 # neurons in the first layer
NN_L2 = 128 # neurons in the second layer
NN_L3 = 128 # neurons in the third layer
NN_L4 = 128 # neurons in the fourth layer

MIN_PROBA = 1e-3 # probability under which select a random cell instead of border cell

CONTINUE_REWARD = 1 # the reward gained each turn
LOSS_REWARD = -50   # the reward if we lose the game
WIN_REWARD = 50     # the reward if we won the game

EPSILON_START = 0.82   # the initial epsilon value
EPSILON_MIN = 0.013    # the minimum epsilon value
EPSILON_DECAY = 0.9675 # the decay epsilon value

NUM_EPISODES = 150 # the number of episode per training
GAMMA = 0.9 # the gamma value for QLearning

NUM_TRAININGS = 20 # the number of trainings
LR = 0.01 # the learning rate
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
    def __init__(self, n_observations, nn_l1=NN_L1, nn_l2=NN_L2, nn_l3=NN_L3, nn_l4=NN_L4):
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
    '''
        Implements an epsilon greedy algorithm

        Properties
        ----------
        model: The current model
        epsilon: The current epsilon value
        epsilon_min: The minimum epsilon value
        epsilon_decay: The epsilon decay value
    '''
    def __init__(self, model, epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY):
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = model

    def get_random_cell(self, game):
        ''' Chose a random cell to uncover'''
        A = game.get_covered_cells(covered=True)
        a = random.choice(A)
        return a

    def __call__(self, observable_cells_ids, state, game):
        '''Caller for the epsilon greedy algorithm'''
        val = np.random.random_sample()
        if(val < self.epsilon or state.size == 0): # if first step return random action
            action = self.get_random_cell(game)
            return action

        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        values = self.model(state_tensor)
        if torch.max(values) < MIN_PROBA:
            action = self.get_random_cell(game)
            return action
        index = torch.argmin(values)
        # print("values: ", values)
        # print("index: ", index)
        action = observable_cells_ids[index.item()]
        return action

    def decay_epsilon(self):
        '''Update epsilon's value'''
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)        



class NeuralNetworkStrategy():
    '''
        Implements the MLP strategy for minesweeper

        Properties
        ----------
        verbose: The verbosity level
        env: Minesweeper
    '''
    def __init__(self, verbose=1):
        self.verbose = verbose
        self.reset_env()

    def reset_env(self, game=None):
        '''Reset the environment'''
        if game is None:
            game = env.Minesweeper("beginner", display=False) # train on beginner mode
        self.env = game

    def get_bordering_cells(self):
        '''Get the bordering cells of uncovered cells'''
        border = set()
        uncovered = self.env.get_covered_cells(covered=False)
        for cell in uncovered:
            neigbhors = self.env.get_neighbors(cell)
            for (i,j) in neigbhors:
                if self.env.O[i,j] == -2: # covered tile
                    border.add((i,j))
        return list(border)

    def get_5x5_neighborhood(self, cell):
        '''Get a list of exactly 25 neighbors id (put a tuple (-1,-1) if it is not a valid cell)'''
        if type(cell) == tuple:
            i, j = cell
        else:
            i,j = env.int2tupple(a, self.env.n_cols)    
        adj = [
                (i-2, j-2), (i-2, j-1), (i-2, j), (i-2, j+1), (i-2, j+2),
                (i-1, j-2), (i-1, j-1), (i-1, j), (i-1, j+1), (i-1, j+2),
                (i, j-2), (i, j-1), (i, j+1), (i, j+2),
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

    def get_random_cell(self):
        ''' Chose a random cell to uncover'''
        A = self.env.get_covered_cells(covered=True)
        a = random.choice(A)
        return a

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

    def get_state(self):
        '''
        get the current state of the environment

        return:
            bordering_cells: the cells we want to uncover
            state: the features to represent these cells
        '''
        bordering_cells = self.get_bordering_cells()
        state = np.array([self.convert_cells_to_feature(self.get_5x5_neighborhood(cell)) for cell in bordering_cells])
        return bordering_cells, state

    def get_reward(self, over):
        '''Get the reward given the `over' state'''
        reward = 0
        if over == 0: # continue
            reward = CONTINUE_REWARD
        elif over == 1: # loss
            reward = LOSS_REWARD
        elif over == 2: # win
            reward = WIN_REWARD
        return reward

    def train_model_episode(self, model, epsilon_greedy, optimizer, loss_fn, device, num_episodes=NUM_EPISODES, gamma=GAMMA):
        '''One training'''
        episode_reward_list = []

        for episode_index in tqdm(range(1, num_episodes)):
            self.reset_env(game = None)
            bordering_cells, state = self.get_state()
            episode_reward = 0

            # print("state: ", state)

            for t in itertools.count():
                action = epsilon_greedy(bordering_cells, state, self.env)
                # print("action: ", action)

                over = self.env.take_action(action, uncover_neighbors=True)
                # print("over: ", over)
                
                reward = self.get_reward(over)
                # print("reward: ", reward)
                
                episode_reward += reward
                next_bordering_cells, next_state = self.get_state()

                if(not state.size == 0): # not empty state
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device)
                    reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)

                    with torch.no_grad():
                        values_next = model(next_state_tensor)
                        max_value_next = torch.max(values_next)
                        target_value = reward_tensor + gamma * max_value_next

                    values = model(state_tensor)
                    # print("values: ", values)
                    predicted_value = torch.max(values)
                    # print("predicted value: ", predicted_value)
                    # print("target value: ", target_value)

                    loss = loss_fn(predicted_value, target_value)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if not (over==0):
                    break

                state = next_state
                bordering_cells = next_bordering_cells

            episode_reward_list.append(episode_reward)
            epsilon_greedy.decay_epsilon()

        return episode_reward_list



    def train_model(self, device, save_file, number_of_trainings=NUM_TRAININGS, lr=LR):
        '''Completely train the model'''
        trains_result_list = [[], [], []]

        for train_index in range(number_of_trainings):
            model = MLPModel(n_observations=24).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
            loss_fn = torch.nn.MSELoss()

            epsilon_greedy = EpsilonGreedy(model=model)

            episode_reward_list = self.train_model_episode(
                model=model, 
                epsilon_greedy=epsilon_greedy, 
                optimizer=optimizer, 
                loss_fn=loss_fn, 
                device=device
            )

            trains_result_list[0].extend(range(len(episode_reward_list)))
            trains_result_list[1].extend(episode_reward_list)
            trains_result_list[2].extend([train_index for _ in episode_reward_list])

        naive_trains_result_df = pd.DataFrame(np.array(trains_result_list).T, columns=["num_episodes", "mean_final_episode_reward", "training_index"])
        naive_trains_result_df["agent"] = "Naive"

        # Save the action-value estimation function of the last train
        torch.save(model, save_file)
        self.model = model

    def uncover_cell(self, cell):
        ''' Uncover a cell'''
        over = self.env.take_action(cell, uncover_neighbors=True)
        if self.verbose >=2:
            self.env.update_display()
            plt.waitforbuttonpress()
        
        return over

    def solve(self, device):
        ''' Solve the game'''
        over = False
        while not over:
            bordering_cells, state = self.get_state()

            action = self.get_random_cell()
            if(state.size > 0):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                values = self.model(state_tensor)
                max_value = torch.max(values)
                if(max_value < 1e-3):
                    action = self.get_random_cell()
                else:
                    action = bordering_cells[torch.argmin(values)]

            over = self.uncover_cell(action)
                
        self.env.show_mines()
        self.env.print_env()
        self.env.update_display()
        if over == 1:
            print("You lost")
        elif over == 2:
            print("You won")
        return over


def load_or_train_model(solver, device, number_of_trainings=NUM_TRAININGS, lr=LR, force_retrain=False):
    '''Helper function to check if the model must be trained or loaded from a file'''
    model_file = "first_mlp.pth"

    if (not force_retrain) and (os.path.isfile(model_file)):
        print("Loading pre-trained model...")
        solver.model = torch.load(model_file, map_location=device)
    else:
        print("Training new model...")
        solver.train_model(device=device, save_file=model_file, number_of_trainings=number_of_trainings, lr=lr)
    return


if __name__ == "__main__":
    solver = NeuralNetworkStrategy(verbose=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set the device to CUDA if available, otherwise use CPU
    print("device: ", device)
    
    # train the model multiple times to get the time in average
    sum_times = 0
    nb_retrainings = 1
    for _ in range(nb_retrainings):
        tic = time.perf_counter()
        load_or_train_model(solver=solver, device=device, force_retrain=True)
        toc = time.perf_counter()
        cur_time = toc - tic
        print(f"Trained the model in {toc - tic:0.4f} seconds")
        sum_times += cur_time
    print(f"Training {nb_retrainings} models in {(sum_times / nb_retrainings):0.4f} seconds on average")
    
    # test the model multiple times to get the winrate
    nb_tests = 100
    nb_wins = 0
    for _ in range(nb_tests):
        solver.reset_env(game = env.Minesweeper("beginner", display=True)) # select the grid to try on
        over = solver.solve(device=device)
        if over == 2:
            nb_wins += 1
    
    print(f"Random strategy: {nb_wins / nb_tests} wins on average")
    print()