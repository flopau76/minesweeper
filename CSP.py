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
 
How to resolve uncertainty:
Some times, nothing can be deduced with certainty. In this case, we select one of the following strategies, defined by the argument "choice":
    - R: Random: a cell is chosen randomly
    - UI: user Input: the user indicates wich cell to uncover
    - S: Smart: we compute all possible solutions for tha constraints and chose the cell with the lowest mine probability
Smart divide the constraints into independent subsets and solve each subset separately via backtracking by checking all possible configurations.

TODO: replace random guess with different strategies and compare them
    - chose a random cell
    - chose a corner/edge cell
    - chose a cell with overlapping constraints
"""

from __future__ import annotations  # for type hinting in the class itself
from typing import List, Set, Union
import matplotlib.pyplot as plt
import numpy as np
import random
import env

class Constraint():
    def __init__(self, value=0, cells:Union[None,set]=None):
        self.value = value
        if cells is None:
            self.cells = set()
        self.cells = set(cells)

    def __eq__(self, constraint: Constraint) -> bool:
        ''' Two constraints are equal if they have the same value and the same cells'''
        return self.value == constraint.value and self.cells == constraint.cells
    
    def __ge__(self, constraint: Constraint) -> bool:
        ''' Partial order: A>=B if B.cells included in A.cells'''
        return len(constraint.cells)>0 and self.cells >= constraint.cells

    def __sub__(self, constraint: Constraint) -> Union[Constraint, None]:
        ''' if A >= B, we can do A-B = ( value.A-value.B, cells.A\cells.B ) '''
        return Constraint(self.value-constraint.value, self.cells-constraint.cells)

    def __isub__(self, constraint: Constraint) -> Union[Constraint, None]:
        ''' in place substraction '''
        self.value -= constraint.value
        self.cells -= constraint.cells
        return self
    
    def __str__(self):
        return "Value: "+str(self.value)+", Cells: "+str(self.cells)
    
    def __repr__(self):
        return f"{self.cells} -> {self.value}"


def simplify_constraints(constraints):
    ''' Simplify a list of constraints by removing inclusions and resolving trivial constraints (equal to 0 or len(cells))
    Modify constraints in place '''
    changes = True
    while changes:
        changes = False
        # first: sort by increasing nb of cells
        constraints.sort(key=lambda x: len(x.cells))

        # then: remove inclusions
        i, j = 0, 1 # note: we can't use i in range(len(...)) because the number of constraints is modified
        while i < len(constraints):
            while j < len(constraints):
                # if two constraints are equal, we remove one
                if constraints[j] == constraints[i]:
                    constraints.pop(j)
                    j -= 1
                # elif one constraint contains the other, we simplify it
                elif constraints[j] >= constraints[i]:
                    changes = True
                    constraints[j] -= constraints[i]
                    # if the result is a trivial constraint, we simplify it further
                    if constraints[j].value == 0:
                        const = constraints.pop(j)
                        j -= 1
                        for cell in const.cells:
                            constraints.append(Constraint(0, [cell]))
                    elif constraints[j].value == len(constraints[j].cells):
                        const = constraints.pop(j)
                        j -= 1
                        for cell in const.cells:
                            constraints.append(Constraint(1, [cell]))
                j += 1
            i += 1
            j = i+1 
    return None


class Constraint_satisfaction_strategy():
    ''' Implements constraint satisfaction strategy for minesweeper'''
    def __init__(self, game="beginner", verbose=1, choice="S", corner=False, seed=None):

        self.game = env.Minesweeper(game, display=(verbose>=2), seed=seed)
        self.constraints = []
        self.verbose = verbose
        self.seed = seed
        self.corner = corner

        # choice of the cell when no action is possible with certainty
        choice_dict = {"R": ("Random", self.get_random_cell) , "UI": ("User Input", self.wait4input), 
                       "S": ("Smart", self.get_best_cell)}
        try:
            self.choice = choice_dict[choice][0]
            self.chose_cell = choice_dict[choice][1]
        except KeyError:
            string = "Choice must be in: "
            for key, val in choice_dict.items():
                string += key + " (" + val[0] + "), "
            raise ValueError(string)

    def uncover_cell(self, cell) -> bool:
        ''' Uncover a cell and update the constraints '''
        if self.game.get_value(cell) != -2:
            return False
        over = self.game.take_action(cell)
        if self.verbose >=1: print(f"Uncovering cell {env.int2tupple(cell, self.game.n_cols)}: {self.game.get_value(cell)}" )
        if self.verbose >=2:
            self.game.update_display()
            plt.waitforbuttonpress()

        if over:
            return over  
              
        # add new constraints and update old ones:
        self.add_constraints(cell)
        simplify_constraints(self.constraints)
        over = self.suppress_single_constraints()
        return over
    
    def flag_cell(self, cell) -> None:
        ''' Flag a covered cell '''
        if self.game.get_value(cell) != -2: # cell is not covered: should not happen
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
        self.constraints.append(Constraint(0, [cell]))  # constraint may allready exist
        
        # create constraints for the neighbors
        remaining_mines = self.game.get_value(cell)
        uncovered_neighbors = []
        for neigh in self.game.get_neighbors_int(cell):
            if self.game.get_value(neigh) == -2:
                uncovered_neighbors.append(neigh)
            elif self.game.get_value(neigh) == -3:
                remaining_mines -= 1
        # if the result is trivial, we simplify it
        if remaining_mines == 0:
            for cell in uncovered_neighbors:
                self.constraints.append(Constraint(0, [cell]))  # constraint may allready exist
        elif remaining_mines == len(uncovered_neighbors):
            for cell in uncovered_neighbors:
                self.constraints.append(Constraint(1, [cell]))
        else:
            const = Constraint(remaining_mines, uncovered_neighbors)
            self.constraints.append(const)

    def suppress_single_constraints(self):
        ''' Remove single constraints (only one cell concerned) and do the implied actions'''
        over = False
        i = 0
        while i < len(self.constraints):
            if len(self.constraints[i].cells) == 1:
                const = self.constraints.pop(i)
                i -= 1
                cell = const.cells.pop()
                if self.game.get_value(cell) == -2:
                    if const.value == 0:
                        over = self.uncover_cell(cell)
                    else:
                        self.flag_cell(cell)
            i += 1
        return over

    def get_random_cell(self):
        ''' Chose a random cell to uncover'''
        A = self.game.get_covered_cells_int()
        if self.seed is not None:
            random.seed(self.seed)
        a = random.choice(A)
        if self.verbose>=1: print("     RANDOMLY chosing cell", a)
        return [(a, False)]
    
    def get_best_cell(self):
        ''' divides the constraints into independent subsets and solve each subset separately '''

        def divide_constraints(constraints):
            ''' Divide the constraints into a list of independent subset of constraints 
            Each subset is a tuple (set of variables involved, list of constraints)'''
            # initialising each constraint into its own subset
            subsets = []
            for const in constraints:
                subsets.append((set(const.cells),[const]))
            # merging subsets with common variables
            changes = True
            while changes:
                changes = False
                i = 0
                while i < len(subsets):
                    j = i+1
                    while j < len(subsets):
                        if len(subsets[i][0].intersection(subsets[j][0])) > 0:
                            subset = subsets.pop(j)
                            subsets[i][0].update(subset[0])
                            subsets[i][1].extend(subset[1])
                            changes = True
                            j -= 1
                        j += 1
                    i += 1
            return subsets
    
        def check_feasibility(constraints):
            ''' Check if the constraints are feasible (ie no contradiction)'''
            for const in constraints:
                if const.value < 0 or const.value > len(const.cells):
                    return False
            return True

        def suppress_single_constraints(constraints, X):
            ''' Remove single constraints (only one cell concerned)
            Instead of modifiying the game, we store the change in X '''
            feasibility = check_feasibility(constraints)
            if not feasibility:
                return False
            i = 0
            while i < len(constraints):
                if len(constraints[i].cells) == 1:
                    const = constraints.pop(i)
                    i -= 1
                    cell = const.cells.pop()
                    if const.value == 0:
                        X[cell] = 0
                    elif const.value == 1:
                        X[cell] = 1
                    else: # game is not feasible: should not happen
                        raise ValueError("Game is not feasible")
                        return False
                i += 1
            return True
        
        def solve_subset(X, subset):
            ''' Given a partially completed grid X and a subset of constraints, computes all possible solutions to the subset
            The solutions are grouped in dictionnarys, depending on the number of mines used

            Returns
            ----
                n_dict with value(int): the number of possible configurations
                res_dict with values(array): the sum of those configurations (total nb of mine per cell)
            '''
            var_set, constraints = subset

            # get the variable with the most constraints
            Y = np.zeros(len(X))
            for const in constraints:
                for cell in const.cells:
                    Y[cell] += 1
            Y = Y*(X<0)
            x_min = np.argmax(Y)
            
            # if all variables are fixed
            if Y[x_min] == 0:
                feasible = suppress_single_constraints(constraints, X)
                if feasible:
                    X = X*(X>0)
                    n_mines = np.sum(X)
                    return {n_mines: 1}, {n_mines: X}
                else:
                    return {}, {}
                
            # trying to set x_min to 0
            X_bis = X.copy()
            constraints_bis = []
            for const in constraints:
                constraints_bis.append(Constraint(const.value, const.cells.copy()))
            constraints_bis.append(Constraint(0, [x_min]))
            simplify_constraints(constraints_bis)
            feasible = suppress_single_constraints(constraints_bis, X_bis)
            if feasible:
                n_dict, res_dict = solve_subset(X_bis, (var_set,constraints_bis))
            else:
                n_dict, res_dict = {}, {}

            # trying to set x_min to 1
            X_bis = X.copy()
            constraints_bis = []
            for const in constraints:
                constraints_bis.append(Constraint(const.value, const.cells.copy()))
            constraints_bis.append(Constraint(1, [x_min]))
            simplify_constraints(constraints_bis)
            feasible = suppress_single_constraints(constraints_bis, X_bis)
            if feasible:
                n1_dict, res1_dict = solve_subset(X_bis, (var_set, constraints_bis))
                for key in res1_dict.keys():
                    if key in res_dict:
                        n_dict[key] += n1_dict[key]
                        res_dict[key] += res1_dict[key]
                    else:
                        n_dict[key] = n1_dict[key]
                        res_dict[key] = res1_dict[key]
            return n_dict, res_dict

        def get_configs(N_mines, n_min, n_max):
            ''' Each subset can be solved using different number of mines
            A configuration is the indication, for each subset, of a number of mines used
            It is valid if the total number of mines used corresponds to the number of remaining mines
            
            Arguments
            ----
                N_mines: list (of size n_subset) of list of integers: each list contains the possible number of mines used for a subset
                n_min, n_max: number of remaining mines to use
            
            Returns
            ----
                Configs: list of configurations (list of integers of size n_subsets with an index to use for each subset)'''
            Configs = []
            config = np.zeros(n_subsets, dtype=int)
            config[0] = -1
            while True:
                # iterate over the possible conbinations
                j = 0
                while j < n_subsets and config[j] == len(N_mines[j])-1:
                    config[j] = 0
                    j += 1    
                if j == n_subsets:
                    break            
                config[j] += 1

                # check if the conbination is valid (ie contains the correct total number of mines)
                count = 0
                for i in range(n_subsets):
                    count += N_mines[i][config[i]]
                if n_min <= count <= n_max:
                    Configs.append(config.copy())
            return Configs
        
        # initialising game representation with current state
        # X contains -1 if covered, 0 if no mine, 1 if mine
        X = - np.ones(self.game.n_rows * self.game.n_cols, dtype=int)
        for a in range(self.game.n_rows * self.game.n_cols):
            if self.game.get_value(a) == -1:
                X[a] = 1
            elif self.game.get_value(a) >= 0:
                X[a] = 0

        # creating a mask for covered cells
        mask_covered = np.zeros(len(X), dtype=int)
        for cell in self.game.get_covered_cells_int():
            mask_covered[cell] = 1
        # creating a mask for covered cells involved in a constraint
        mask_constraints = np.zeros(len(X), dtype=int)
        for const in self.constraints:
            for cell in const.cells:
                mask_constraints[cell] = 1

        # very special case where all constraints disappeared (ex: after uncovering a 3 in a corner)
        if len(self.constraints) == 0:
            return [(np.random.choice(np.where(mask_covered==1)[0], 1, replace=False)[0], False)]

        # dividing the constraints into independent subsets
        if self.verbose>=1: print("COMPUTING subsets")
        subsets = divide_constraints(self.constraints)
        n_subsets = len(subsets)

        # solving each subset independently
        # the solutions for each subset are grouped depending on the number of mines used
        if self.verbose>=1: print("SOLVING the subsets")
        N_mines = []
        N_sol = []
        Sol = []
        for subset in subsets:
            n_dict, res_dict = solve_subset(X, subset)
            N_mines.append(list(n_dict.keys()))
            N_sol.append([n_dict[key] for key in N_mines[-1]])
            Sol.append([res_dict[key] for key in N_mines[-1]])
            if self.verbose>=1:
                for key in n_dict.keys():
                    if self.verbose>=1:
                        print(f"     {key} mines: {n_dict[key]} configurations:")
                        print(res_dict[key].reshape(self.game.n_rows, self.game.n_cols).astype(int))
        
        # counting the remaining mines
        n_remaining = self.game.n_mines
        for cell in range(len(X)):
            if self.game.get_value(cell) == -3:
                n_remaining -= 1
        uncertainty = np.sum(mask_covered - mask_constraints)
        # getting the possible configurations
        Configs = get_configs(N_mines, n_remaining-uncertainty, n_remaining)

        # combining the different subsets
        XX = np.zeros(len(X))
        total_solutions = 0
        for config in Configs:
            # number of solutions in this configuration
            prod_sol = 1
            sum_mines = 0
            for i in range(n_subsets):
                prod_sol *= N_sol[i][config[i]]
                sum_mines += N_mines[i][config[i]]
            total_solutions += prod_sol
            # adding the configuration to the total
            for i in range(n_subsets):
                XX += Sol[i][config[i]]*prod_sol/N_sol[i][config[i]]
        if self.verbose>=1:
            print("     total possible configurations: ", total_solutions)
            print(XX.astype(int).reshape(self.game.n_rows, self.game.n_cols))

        # checking if we have a trivial solution:
        res = []
        for cell in range(len(X)):
            if mask_constraints[cell] == 1:
                if XX[cell] == 0:
                    res.append((cell, False))
        if res:
            return res

        # getting the best guess
        XX = XX * mask_constraints + total_solutions * (1-mask_constraints)
        best_guess = np.argmin(XX)
        best_p = np.min(XX)/total_solutions
        average_p = n_remaining / np.sum(mask_covered)

        if self.verbose>=1:
            print(f"     best_guess: {env.int2tupple(best_guess, self.game.n_cols)} with p={best_p} versus average_p={average_p}")

        if best_p <= average_p:
            return [(best_guess, False)]
        
        else:
            if self.seed is not None:
                random.seed(self.seed)
            return [(np.random.choice(np.where(mask_covered-mask_constraints==1)[0], 1, replace=False)[0], False)]
            # TODO: implement other options: corner, edge, cell, with overlapping constraints

    def wait4input(self):
        ''' Wait for user input'''
        i, j, flag = self.game.wait4input()
        return [(i, j), flag]
    
    def solve(self):
        ''' Solve the game'''
        over = False
        over = self.uncover_cell(0) # wich cell to uncover first
        stack = []
        if self.corner:
            stack = [(0,0)]
        while not over:
            if not stack:
                stack = self.chose_cell()
            cell, flag = stack.pop()
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
    


F = np.zeros((3,2))
F[2,1] = 1

G = np.zeros((3,5))
G[2,0] = 1
G[2,3] = 1

H = np.zeros((8,8))
H[1,2] = 1
H[2,1] = 1


if __name__ == "__main__":
        
    solver = Constraint_satisfaction_strategy(game="beginner", verbose=2, choice="S", seed=33)
    solver.solve()

if __name__ == "__main1__":
    wins = 0
    for i in range(100):
        print(i)
        solver = Constraint_satisfaction_strategy(game="beginner", verbose=0, choice="S", seed=i)
        over = solver.solve()
        if over == 2:
            wins += 1
    print(f"Win rate: {wins}/100")