import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
import env, SPS, CSP, MLP, RND

strategies = [SPS.single_point_strategy, CSP.Constraint_satisfaction_strategy, RND.Random_strategy] #, MLP.MLP]
difficulties = ["beginner", "intermediate", "expert"]
nb_tests = 1000

Res = {"solver": [], "difficulty":[], "winrate": [], "time": []}

if __name__ == "__main__":
    # test the model multiple times to get the winrate
    for strategy in strategies:
        for difficulty in difficulties:
            print(f"testing {strategy} on {difficulty}")
            nb_wins = 0
            start = time()
            for _ in range(nb_tests):
                solver = strategy(game=difficulty, verbose=0)
                over = solver.solve()
                if over == 2:
                    nb_wins += 1
            end = time()
            Res["solver"].append(strategy)
            Res["difficulty"].append(difficulty)
            Res["winrate"].append(nb_wins / nb_tests)
            Res["time"].append(end - start)

    df = pd.DataFrame(Res)
    print(df)
        
