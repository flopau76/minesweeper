import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
import env, SPS, CSP, MLP, RND

difficulties = ["beginner", "intermediate", "expert"]
nb_tests = 1000


# test CSP with random and smart guess
if __name__ == "__main__":
    Res = []
    for difficulty in difficulties:
        for choice in ['R', 'S']:
            print(f"testing CSP with {choice} choise on {difficulty}:", end=" ")
            nb_wins = 0
            start = time()
            for _ in range(nb_tests):
                solver = CSP.Constraint_satisfaction_strategy(game=difficulty, verbose=0, choice=choice)
                over = solver.solve()
                if over == 2:
                    nb_wins += 1
            end = time()
            Res.append([choice, difficulty, nb_wins / nb_tests, end - start])
            print(f"winrate: {nb_wins/nb_tests}, time: {end-start}")

    df = pd.DataFrame(Res, columns =['choice', 'difficulty', 'winrate', 'time'])
    print(df)

# test SPS and CSP with random guess
if __name__ == "__main2__":
    Res = []
    for strategy in [SPS.single_point_strategy, CSP.Constraint_satisfaction_strategy]:
        for difficulty in difficulties:
            print(f"testing {strategy} on {difficulty}:", end=" ")
            nb_wins = 0
            start = time()
            for _ in range(nb_tests):
                solver = strategy(game=difficulty, verbose=0, choice='R')
                over = solver.solve()
                if over == 2:
                    nb_wins += 1
            end = time()
            Res.append([strategy, difficulty, nb_wins / nb_tests, end - start])
            print(f"winrate: {nb_wins/nb_tests}, time: {end-start}")

    df = pd.DataFrame(Res, columns =['solver', 'difficulty', 'winrate', 'time'])
    print(df)
        
