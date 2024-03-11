"""
RANDOM STRATEGY
---------------------
working principle:
 -naive agent which selects a random covered tile each turn

-quantify:
    -success rate
    -computation time
"""

import random
import matplotlib.pyplot as plt
import env
import numpy as np

class Random_strategy():
    ''' 
        Implements random strategy for minesweeper

        Properties
        ----------
        game: Minesweper
        verbose: The verbosity

    '''
    def __init__(self, verbose=1):
        self.game = env.Minesweeper("beginner", display=bool(verbose-1))
        self.verbose = verbose

    def get_random_cell(self):
        ''' Chose a random cell to uncover'''
        A = self.game.get_covered_cells()
        a = random.choice(A)
        return a

    def uncover_cell(self, cell):
        ''' Uncover the given cell'''
        over = self.game.take_action(cell, uncover_neighbors=True)
        if self.verbose >=2:
            self.game.update_display()
            plt.waitforbuttonpress()
        return over

    def solve(self):
        ''' Solve the game'''
        over = False
        while not over:
            a = self.get_random_cell()
            if self.verbose>=1: print("Randomly uncovering cell", a)
            over = self.uncover_cell(a)
            if over:
                break
        self.game.show_mines()
        if self.verbose >= 1:
            self.game.print_env()
            if over == 1:
                print("You lost")
            else:
                print("You won")
        if self.verbose >= 2:
            self.game.update_display()
            plt.waitforbuttonpress()  
        return over


if __name__ == "__main__":
    # test the model multiple times to get the winrate
    nb_tests = 1000
    nb_wins = 0
    for _ in range(nb_tests):
        solver = Random_strategy(verbose=0)
        over = solver.solve()
        if over == 2:
            nb_wins += 1
    
    print(f"Random strategy: {nb_wins / nb_tests} wins on average")
    print()
