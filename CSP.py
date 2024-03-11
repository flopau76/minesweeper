"""
CONSTRAINT SATISFACTION PROBLEM
-------------------------------
Idea:
    - each uncovered cell gives a constraint: the number of mines in its neighborhood
    - if several constraints overlap, they might be simplified
 
Working principle:
    - each time a cell is uncovered, we add the corresponding constraints
    - after adding a constraint, we simplify the existing ones by removing inclusions
    - if a constraint concerns only one cell, we can take action (uncover or flag the cell)
    - we can then suppress this constraint, as we already simplified the existing ones containing it and did the action
 
Randomness:
 -when the stack is empty, nothing can be deduced with certainty from the working principle
 -we therfore uncover a random cell
"""

from __future__ import annotations  # for type hinting in the class itself
from typing import List, Set, Union
import matplotlib.pyplot as plt
import numpy as np
import random
import env

class Constraint():
    def __init__(self, value=0, cells:Union[None,list]=None):
        self.value = value
        if cells is None:
            self.cells = []
        self.cells = cells
        self.cells.sort()

    def __eq__(self, constraint: Constraint) -> bool:
        ''' Two constraints are equal if they have the same value and the same cells'''
        return self.value == constraint.value and self.cells == constraint.cells
    
    def __ge__(self, constraint: Constraint) -> bool:
        ''' Partial order: A>=B if B.cells included in A.cells'''
        if len(self.cells) < len(constraint.cells):
            return False
        i=0; j=0
        while i<len(self.cells) and j<len(constraint.cells):
            if self.cells[i] == constraint.cells[j]:
                i+=1
                j+=1
            else:
                i+=1
        return j==len(constraint.cells)

    def __sub__(self, constraint: Constraint) -> Union[Constraint, None]:
        ''' if A >= B, we can do A-B = ( value.A-value.B, cells.A\cells.B ) '''
        new_cells = []
        i=0; j=0
        while i<len(self.cells) and j<len(constraint.cells):
            if self.cells[i] == constraint.cells[j]:
                i+=1
                j+=1
            else:
                new_cells.append(self.cells[i])
                i+=1
        if j<len(constraint.cells):
            return None # the constraint is not included in the current one
        while i<len(self.cells):
            new_cells.append(self.cells[i])
            i+=1
        return Constraint(self.value-constraint.value, new_cells)
    
    def __str__(self):
        return "Value: "+str(self.value)+", Cells: "+str(self.cells)


class Constraint_satisfaction_strategy():
    ''' Implements constraint satisfaction strategy for minesweeper'''
    def __init__(self, game="beginner", verbose=1, choice="R"):

        self.game = env.Minesweeper(game, display=(verbose>=2))
        self.constraints = []
        self.verbose = verbose

        # choice of the cell when no action is possible with certainty
        if choice == "R":
            self.chose_cell = self.get_random_cell
        elif choice == "I":
            self.chose_cell = self.wait4input
        else:
            raise ValueError("Choice must be 'R' (random) or 'I' (user input)")

    def uncover_cell(self, cell) -> bool:
        ''' Uncover a cell and update the constraints '''
        if self.game.get_value(cell) != -2:
            return False
        over = self.game.take_action(cell)
        if self.verbose >=1: print(f"Uncovering cell {cell}: {self.game.get_value(cell)}" )
        if self.verbose >=2:
            self.game.update_display()
            plt.waitforbuttonpress()

        if over:
            return over  
              
        # add new constraints and update old ones:
        self.add_constraints(cell)
        over = self.simplify_constraints()
        return over
    
    def flag_cell(self, cell) -> None:
        ''' Flag a covered cell '''
        if self.game.get_value(cell) != -2:
            return
        self.game.take_action(cell, flag=True)
        if self.verbose >=1: print("Flagging cell", cell)
        if self.verbose >=2:
            self.game.update_display()

    def add_constraints(self, cell):
        ''' Create constraints for a cell that has (just) been uncovered, and add them to the list of constraints'''
        if self.game.get_value(cell) < 0:
            raise ValueError("Cell still covered")  # should not happen, but good to check
        
        # there is no mine in the cell (needed for simplification of existing constraints)
        self.constraints.append(Constraint(0, [cell]))  # !!! : constraint may allready exist
        
        # create constraints for the neighbors
        remaining_mines = self.game.get_value(cell)
        uncovered_neighbors = []
        for neigh in self.game.get_neighbors(cell):
            if self.game.get_value(neigh) == -2:
                uncovered_neighbors.append(neigh)
            elif self.game.get_value(neigh) == -3:
                remaining_mines -= 1
        # if the result is trivial, we simplify it
        if remaining_mines == 0:
            for cell in uncovered_neighbors:
                self.constraints.append(Constraint(0, [cell]))  # !!! : constraint may allready exist
        elif remaining_mines == len(uncovered_neighbors):
            for cell in uncovered_neighbors:
                self.constraints.append(Constraint(1, [cell]))
        else:
            const = Constraint(remaining_mines, uncovered_neighbors)
            self.constraints.append(const)

    def simplify_constraints(self):
        ''' Simplify all constraints by removing inclusions and trivial constraints'''
        changes = True
        while changes:
            changes = False
            # first: sort by increasing nb of cells
            self.constraints.sort(key=lambda x: len(x.cells))

            # then: remove inclusions
            i, j = 0, 1 # note: we can't use i in range(len(...)) because the number of constraints is modified
            while i < len(self.constraints):
                while j < len(self.constraints):
                    # if two constraints are equal, we remove one
                    if self.constraints[j] == self.constraints[i]:
                        self.constraints.pop(j)
                        j -= 1
                    # elif one constraint contains the other, we simplify it
                    elif self.constraints[j] >= self.constraints[i]:
                        changes = True
                        self.constraints[j] -= self.constraints[i]
                        # if the result is a trivial constraint, we simplify it further
                        if self.constraints[j].value == 0:
                            const = self.constraints.pop(j)
                            j -= 1
                            for cell in const.cells:
                                self.constraints.append(Constraint(0, [cell]))
                        elif self.constraints[j].value == len(self.constraints[j].cells):
                            const = self.constraints.pop(j)
                            j -= 1
                            for cell in const.cells:
                                self.constraints.append(Constraint(1, [cell]))
                    j += 1
                i += 1
                j = i+1 

        over = False
        # finally: remove trivial constraints and do the implied actions
        i = 0
        while i < len(self.constraints):
            if len(self.constraints[i].cells) == 1:
                const = self.constraints.pop(i)
                i -= 1
                if const.value == 0:
                    for cell in const.cells:
                        if self.game.get_value(cell) == -2:
                            over = self.uncover_cell(cell)
                else:
                    self.flag_cell(const.cells[0])
            i += 1
        return over

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
            cell, flag = self.chose_cell()
            if flag:
                self.flag_cell(cell)
            else:
                over = self.uncover_cell(cell)
                
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
    solver = Constraint_satisfaction_strategy(verbose=2)
    solver.solve()

    print()