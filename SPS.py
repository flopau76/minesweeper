"""
SINGLE POINT STRATEGY
---------------------
Working principle:
 -for each uncovered cell, we count the amount of remaining mines in its neighbors.
    -if the amount of remaining mines is equal to the amount of covered neighbors, we flag all its neighbors.
    -if the amount of remaining mines is 0, we uncover all its neighbors.

stack: it contains all the (interesting) cells left to check
 -each time a cell is uncovered, we add it to the stack. we also add all its neighbors to the stack as their amount of uncovered neighbors has changed.
 -each time a cell is flagged, we add all its neighbors to the stack as their amount of covered neighbors and mines has changed.

Randomness:
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
    def __init__(self, game="beginner", verbose=1, choice="R"):

        self.game = env.Minesweeper(game, display=(verbose>=2))
        self.stack = set()
        self.verbose = verbose

        # choice of the cell when no action is possible with certainty
        choice_dict = {"R": ("Random", self.get_random_cell) , "UI": ("User Input", self.wait4input)}
        try:
            self.choice = choice_dict[choice][0]
            self.chose_cell = choice_dict[choice][1]
        except KeyError:
            string = "Choice must be in: "
            for key, val in choice_dict.items():
                string += key + " (" + val[0] + "), "
            raise ValueError(string)

    def uncover_cell(self, cell):
        ''' Uncover a cell and add it and its neighbors to the stack'''
        over = self.game.take_action(cell)
        if self.verbose >=2:
            self.game.update_display()
            plt.waitforbuttonpress()
        if over:
            return over
        self.stack.update(self.game.get_neighbors(cell)  )  # add uncovered cell and its neighbors to stack
        self.stack.add(cell)
        return over
    
    def flag_cell(self, cell):
        ''' Flag a cell and add its neighbors to the stack'''
        self.game.take_action(cell, flag=True)
        if self.verbose >=2:
            self.game.update_display()
            plt.waitforbuttonpress()
        self.stack.update(self.game.get_neighbors(cell))    # add neigbhors of flag to stack
        
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

    def get_random_cell(self):
        ''' Chose a random cell to uncover'''
        A = self.game.get_covered_cells()
        a = random.choice(A)
        if self.verbose>=1: print("     RANDOMLY chosing cell", a)
        return a, False
    
    def wait4input(self):
        ''' Wait for user input'''
        i, j, flag = self.game.wait4input()
        return (i, j), flag
    
    def solve(self):
        ''' Solve the game'''
        over = False
        while not over:
            if not self.stack:
                # if stack is empty, use specified strategy to chose cell
                cell, flag = self.chose_cell()
                if flag:
                    self.flag_cell(cell)
                else:
                    over = self.uncover_cell(cell)
                if over:
                    break
            else:
                # examine cells from the stack
                cell = self.stack.pop()
                over = self.check_cell(cell)
                if over:
                    break
              
        self.game.show_mines()
        if self.verbose>=1:
            self.game.print_env()
            if over == 1:
                print("You lost")
            else:
                print("You won")
        if self.verbose >=2:
            self.game.update_display()
            plt.waitforbuttonpress()  
        return over


if __name__ == "__main__":
    solver = single_point_strategy(verbose=2)
    solver.solve()

    print()
