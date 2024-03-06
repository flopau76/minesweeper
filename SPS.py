"""
SINGLE POINT STRATEGY
---------------------
working principle:
 -for each uncovered cell, we count the amount of remaining mines in its neighbors.
    -if the amount of remaining mines is equal to the amount of covered neighbors, we flag all its neighbors.
    -if the amount of remaining mines is 0, we uncover all its neighbors.

stack: it contains all the (interesting) cells left to check
 -each time a cell is uncovered, we add it to the stack. we also add all its neighbors to the stack as their amount of uncovered neighbors has changed.
 -each time a cell is flagged, we add all its neighbors to the stack as their amount of covered neighbors and mines has changed.

randomness:
 -when the stack is empty, nothing can be deduced with certainty from the working principle
 -we therfore uncover a random cell

 
TODO:
-quantify:
    -success rate
    -computation time
"""

import matplotlib.pyplot as plt
import env
import numpy as np
import random

class single_point_strategy():
    ''' 
        Implements single point strategy for minesweeper

        Properties
        ----------
        game: Minesweper

    '''
    def __init__(self, verbose=1):

        self.game = env.Minesweeper("beginner", display=bool(verbose-1))
        self.stack = []
        self.verbose = verbose

    def uncover_cell(self, cell):
        ''' Uncover a cell and add it and its neighbors to the stack'''
        over = self.game.take_action(cell)
        if self.verbose >=2:
            self.game.update_display()
            plt.waitforbuttonpress()
        if over:
            return over
        self.stack += self.game.get_neighbors(cell)    # add uncovered cell and its neighbors to stack
        self.stack.append(cell)
        return over
    
    def flag_cell(self, cell):
        ''' Flag a cell and add its neighbors to the stack'''
        self.game.take_action(cell, flag=True)
        if self.verbose >=2:
            self.game.update_display()
            plt.waitforbuttonpress()
        self.stack += self.game.get_neighbors(cell)    # add neigbhors of flag to stack

    def get_random_cell(self):
        ''' Chose a random cell to uncover'''
        A = self.game.get_covered_cells()
        a = random.choice(A)
        return a
        
    def check_cell(self, cell):
        ''' Check a cell for a mine, depending on its value and the value of its neighbors'''
        val = self.game.get_value(cell)
        if self.verbose>=1: print("Checking cell", cell, "with value", val)

        # cell covered: nothing to do
        if val == -2:
            return False
        # cell containing a mine (should not happen)
        if val == -1:
            return 1
        
        flagged_neighbors = sum([self.game.O[neighbor]==-3 for neighbor in self.game.get_neighbors(cell)])
        covered_neighbors = sum([self.game.O[neighbor]==-2 for neighbor in self.game.get_neighbors(cell)])
        # no covered neighbors: nothing to do
        if covered_neighbors == 0:
            return False
        # flagging all covered neighbors
        if val - flagged_neighbors == covered_neighbors:
            if self.verbose >=1: print("Flagging all neighbors")
            for neighbor in self.game.get_neighbors(cell):
                if self.game.get_value(neighbor) == -2:
                    self.flag_cell(neighbor)
            if self.verbose>=1: self.game.print_env()
        # uncovering all covered neighbors
        elif val - flagged_neighbors == 0:            
            if self.verbose>=1: print("Uncovering all neighbors")
            for neighbor in self.game.get_neighbors(cell):
                if self.game.get_value(neighbor) == -2:
                    over = self.uncover_cell(neighbor)
                    if over:
                        return over
            if self.verbose>=1: self.game.print_env()
        # nothing to deduce with certainty
        else:
            return False

    def solve(self):
        ''' Solve the game'''
        over = False
        while not over:
            if not self.stack:
                # if stack is empty, uncover a random cell
                a = self.get_random_cell()
                if self.verbose>=1: print("Randomly uncovering cell", a)
                over = self.uncover_cell(a)
                if over:
                    break
            else:
                # examine cells from the stack
                cell = self.stack.pop()
                over = self.check_cell(cell)
                if over:
                    break
        self.game.show_mines()
        self.game.print_env()
        self.game.update_display()
        if over == 1:
            print("You lost")
        else:
            print("You won")


if __name__ == "__main__":
    solver = single_point_strategy(verbose=2)
    solver.solve()

    print()
