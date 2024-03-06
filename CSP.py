"""
CONSTRAINT SATISFACTION PROBLEM
-------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import env

class Constraint():
    def __init__(self, value=0, cells=None):
        self.value = value
        if cells is None:
            self.cells = []
        self.cells = cells
        self.cells.sort()
    
    def contains(self, constraint):
        ''' Check if the current constraint contains the other one'''
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

    def __sub__(self, constraint):
        ''' Substract one constraint from another. A-B -> cells.A\cells.B, value.A-value.B'''
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
    def __init__(self, verbose=1):

        self.game = env.Minesweeper("beginner", display=bool(verbose-1))
        self.constraints = []
        self.stack_uncover = []
        self.stack_flag = []
        self.verbose = verbose

    def uncover_cell(self, cell):
        ''' Uncover a cell and update the constraints'''
        over = self.game.take_action(cell)
        if self.verbose >=2:
            self.game.update_display()
            plt.waitforbuttonpress()
        if over:
            return over        
        # add new constraints and update old ones:
        self.create_constraint(cell)
        print("Constraints before simplification:")
        for const in self.constraints:
            print(const)  
        self.simplify_constraints()
        return over
    
    def flag_cell(self, cell):
        ''' Flag a cell '''
        self.game.take_action(cell, flag=True)
        if self.verbose >=2:
            self.game.update_display()
            plt.waitforbuttonpress()

    def get_random_cell(self):
        ''' Chose a random cell to uncover'''
        A = self.game.get_covered_cells()
        a = random.choice(A)
        return a

    def create_constraint(self, cell):
        ''' Create constraints for a cell that has been uncovered'''
        if self.game.get_value(cell) < 0:
            raise ValueError("Cell still covered")  # not neccessary, but good to check
        
        neighbors_mines = 0
        neighbors_uncovered = []
        for neigh in self.game.get_neighbors(cell):
            if self.game.get_value(neigh) == -2:
                neighbors_uncovered.append(neigh)
            elif self.game.get_value(neigh) == -1:
                neighbors_mines += 1
        self.constraints.append(Constraint(neighbors_mines, neighbors_uncovered))

    def simplify_constraints(self):
        ''' Simplify the constraints'''
        changes = True
        while changes:
            changes = False
            # first: sort by increasing nb of cells
            self.constraints.sort(key=lambda x: len(x.cells))
            # then: remove inclusions
            for i in range(len(self.constraints)):
                for j in range(i+1, len(self.constraints)):
                    # if one constraint contains the other, we simplify it
                    if self.constraints[j].contains(self.constraints[i]):
                        changes = True
                        self.constraints[j] -= self.constraints[i]
                        # if the result is a trivial constraint, we simplify it further
                        if self.constraints[j].value == 0:
                            const = self.constraints.pop(j)
                            for cell in const.cells:
                                self.constraints.append(Constraint(0, [cell]))
                        elif self.constraints[j].value == len(self.constraints[j].cells):
                            const = self.constraints.pop(j)
                            for cell in const.cells:
                                self.constraints.append(Constraint(1, [cell]))

        # finally: remove trivial constraints and add them to stack
        for i in range(len(self.constraints)):
            if len(self.constraints[i].cells) == 0:
                const = self.constraints.pop(i)
                if const.value == 0:
                    for cell in const.cells:
                        self.stack_uncover.append(cell)
                elif const.value == len(const.cells):
                    for cell in const.cells:
                        self.stack_flag.append(cell)

    def solve(self):
        ''' Solve the game'''
        over = False
        while not over:
            # examine cells from the stack
            if self.stack_uncover:
                cell = self.stack_uncover.pop()
                over = self.uncover_cell(cell)
                if over:
                    break
                if self.verbose >=1: print("Uncovering cell", cell)
            elif self.stack_flag:
                cell = self.stack_flag.pop()
                self.flag_cell(cell)
                if self.verbose >=1: print("Flagging cell", cell)
            else:   # stack is empty -> uncover a random cell
                a = self.get_random_cell()
                if self.verbose>=1: print("Randomly uncovering cell", a)
                over = self.uncover_cell(a)
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
    solver = Constraint_satisfaction_strategy(verbose=2)
    solver.solve()

    print()