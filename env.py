"""
Mineweeper Environment
-----------------------
This file implements the Minesweeper environment

It can be launched directly to play a game in a terminal. 
"""

import numpy as np
import os

# Only for plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def int2tupple(y, n_cols):
    ''' utility function: convert int in (0,...,n-1) to matrix coordinates (row, col)'''
    return int(y) // n_cols, int(y) % n_cols

def tupple2int(x, n_cols):
    ''' utility function: convert matrix coordinates (row, col) to int'''
    i, j = x
    return i*n_cols + j

def clear_terminal():
    os.system("clear")


class Minesweeper():
    '''
    Class to represent the Minesweeper environment. 

    Attributes
    ----------
    n_rows, n_cols : int
        size of grid
    n_mines : int
        number of mines
    E : array_like(int, ndim=2, dtype=int)
        grid with the number of mines in the adjacent tiles: -1 if mine, 0-8 otherwise
    O : array_like(bool, ndim=2, dtype=int)
        observable grid: -1 if mine, -2 if covered, -3 if flagged, 0-8 if otherwise
    n_remaining : int
        number of tiles left to be explored
    display : None or tuple (fig, ax, im)
        visual representation of the environments
    '''
 
    def __init__(self, game="beginner", display=False, seed=None):
        ''' Initialize the environment.
            
            Parameters
            ----------
            game : str or tuple of shape 3 or array_like(int, ndim=2) of shape (n_rows,n_columns)
                If str, it can be "beginner", "intermediate" or "expert". 
                If tuple, it should contain the number of rows, columns and mines to initialize randomly. 
                If array_like, it should be a matrix of shape (n_rows,n_columns) containing True in the positions of the mines.
            display : bool
                if True, a visual representation of the environment is created
        '''
        if type(game) == str:
            if game == "beginner":
                self.__init__((8, 8, 10), display, seed)
            elif game == "intermediate":
                self.__init__((16, 16, 40), display, seed)
            elif game == "expert":
                self.__init__((16, 30, 99), display, seed)
            else:
                raise ValueError("Invalid game: string not known")
            return None
        elif type(game) == tuple:
            if len(game) != 3:
                raise ValueError("Invalid game: tuple should contain 3 elements")
            self.n_rows, self.n_cols, self.n_mines = game
            # Generate a random grid with mine locations
            G = np.zeros((self.n_rows, self.n_cols), dtype=bool)
            if seed is not None:
                np.random.seed(seed)
            mines = np.random.choice(self.n_rows*self.n_cols, self.n_mines, replace=False)
            for mine in mines:
                i, j = int2tupple(mine, self.n_cols)
                G[i,j] = True
        elif type(game) == np.ndarray:
            if game.ndim != 2:
                raise ValueError("Invalid game: array should be 2D")
            self.n_rows, self.n_cols = game.shape
            self.n_mines = np.sum(game)
            G = game
        else:
            raise ValueError("Invalid game")

        # array containg -1 if there is a mine, the number of mines in the adjacent tiles otherwise
        self.E = np.zeros((self.n_rows, self.n_cols), dtype=int)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if G[i,j] == True:
                    self.E[i,j] = -1
                    for a, b in self.get_neighbors((i,j)):
                        if G[a,b] == False:
                            self.E[a,b] += 1

        # numbers of tiles left to be explored
        self.n_remaining = self.n_rows * self.n_cols - self.n_mines
        # observed grid
        self.O = -2 * np.ones((self.n_rows, self.n_cols), dtype=int)

        if display:
            self.display = self.create_display()
        else:
            self.display = None

# _________________________________________________________________________________________
# Getting information about the environment
    def get_covered_cells(self, covered=True):
        ''' utility function: get the indices of the (un)covered cells'''
        if covered:
            A = np.where(self.O == -2)
            return list(zip(A[0], A[1]))
        else:
            A = np.where(self.O > -2)
            return list(zip(A[0], A[1]))
        
    def get_covered_cells_int(self, covered=True):
        ''' same as previous but with int index and not tupple '''
        res = self.get_covered_cells(covered)
        return [tupple2int(x, self.n_cols) for x in res]

    def get_neighbors(self, a):
        ''' utility function: get all adjacent tiles, including diagonals
        
        Parameters
        ----------
        a : int or tupple of shape 2
            the tile to reveal
            
        Returns
        -------
        res : list of tupple of shape 2'''
        if type(a) == tuple:
            i, j = a
        else:
            i,j = int2tupple(a, self.n_cols)    
        adj = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j-1), (i, j+1), (i+1, j-1), (i+1, j), (i+1, j+1)]
        res = []
        for a, b in adj:
            if a >= 0 and a < self.n_rows and b >= 0 and b < self.n_cols:
                res.append((a,b))
        return res
    
    def get_neighbors_int(self, a):
        ''' same as previous but with int index and not tupple '''
        res = self.get_neighbors(a)
        return [tupple2int(x, self.n_cols) for x in res]

    def get_value(self, a):
        ''' utility function: get the value of a tile
        
        Parameters
        ----------
        a : int or tupple of shape 2
            the tile to reveal'''
        if type(a) == tuple:
            i, j = a
        else:
            i,j = int2tupple(a, self.n_cols)    
        return self.O[i,j]

# _________________________________________________________________________________________
# Taking actions and updating the environment
    def take_action(self, a, flag=False, uncover_neighbors=False):
        ''' Take an action (i.e., reveal or flag a tile).
            
            Parameters
            ----------
            a : int or tupple of shape 2
                the tile to reveal
            flag : bool
                if True, flag the tile as containing a mine
            uncover_neighbors : bool
                if True, automatically uncover all cells neighboring a cell with 0 mines.
                
            Returns
            -------
            done : int
                0 if the game is not over, 1 if lost, 2 if won
        '''
        # TODO: sanity check to verify if input is valid
        if type(a) == tuple:
            i,j = a  
        else:
            i,j = int2tupple(a, self.n_cols)
        assert self.O[i,j] <= -2 , "Tile already uncovered"  # TODO: To be removed ?
        if flag:
            if self.O[i,j] == -3:
                self.O[i,j] = -2
            elif self.O[i,j] == -2:
                self.O[i,j] = -3
            return 0
        self.O[i,j] = self.E[i,j]
        if self.E[i,j] != -1:
            self.n_remaining -= 1
            if self.E[i,j] == 0 and uncover_neighbors:
                self.uncover_neighbors((i,j))
            if self.n_remaining == 0:
                return 2
            return 0
        return 1
    
    def uncover_neighbors(self, a):
        ''' Uncover all cells neighboring a cell with 0 mines.
            
            Parameters
            ----------
            a : int or tupple of shape 2
                the tile to reveal
        '''
        if type(a) == tuple:
            i, j = a
        else:
            i,j = int2tupple(a, self.n_cols)    
        if self.E[i,j] == 0:
            for a, b in self.get_neighbors((i,j)):
                if self.O[a,b] == -2:
                    self.O[a,b] = self.E[a,b]
                    self.n_remaining -= 1
                    if self.E[a,b] == 0:
                        self.uncover_neighbors((a,b))

# _________________________________________________________________________________________
# Clearing and resetting the environment                           
    def clear(self):
        ''' Clear the environment, keeping the same mine location. '''
        self.n_remaining = self.n_rows * self.n_cols - self.n_mines
        self.O = -1 * np.ones((self.n_rows, self.n_cols), dtype=int)
    
    def reset(self):
        ''' Clear the environment, reseting the game with new mines. '''
        self.__init__((self.n_rows, self.n_cols, self.n_mines))

# _________________________________________________________________________________________
# Plotting and user interface only
    def show_mines(self):
        ''' Update O with the position of all mines (usefull only at the end of the game) '''
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.E[i,j] == -1:
                    self.O[i,j] = -1

    def create_display(self):
        ''' Creates a visual representation of the environment. '''
        plt.close()
        fig, ax = plt.subplots(figsize=[8,4])

        colors = {
            -3 : "gray",   # Flagged
            -2 : "gray",   # Unexplored
            -1 : "black",   # Mine
            0 : "white",     # Empty
            1 : "green",    
            2 : "yellow",      
            3 : "orange",   
            4 : "pink",
            5 : "red",
            6 : "purple",
            7 : "brown",
            8 : "brown"
        }
        
        # Plot the tiles
        im = ax.imshow(self.O, cmap=ListedColormap(list(colors.values())), vmin=-3, vmax=8)   
        
        # Ticks and grid
        ax.set_xticks(np.arange(0, self.n_cols, 1))
        ax.set_xticks(np.arange(-0.5, self.n_cols, 1), minor=True)
        ax.set_xticklabels(np.arange(0, self.n_cols, 1))

        ax.set_yticks(np.arange(0, self.n_rows, 1))
        ax.set_yticks(np.arange(-0.5, self.n_rows, 1), minor=True)
        ax.set_yticklabels(np.arange(0, self.n_rows, 1))

        ax.grid(which='minor', color='k')

        # Add numbers to tiles
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.O[i,j] != -2:
                    if self.O[i,j] == -3:
                        ax.text(j, i, 'F', va='center', ha='center')
                    elif self.O[i,j] == -1:
                        ax.text(j, i, 'X', va='center', ha='center', color='red')
                    else:
                        ax.text(j, i, self.O[i,j], va='center', ha='center')
            
        plt.tight_layout()
        plt.ion()
        plt.show()
        return fig, ax, im
 
    def update_display(self):
        ''' Update the visual representation of the environment. '''
        assert self.display is not None, "No display available"

        fig, ax, im = self.display

        # Clear the numbers
        for txt in ax.texts:
            txt.remove()        
        # Add new number in tiles
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.O[i,j] != -2:
                    if self.O[i,j] == -3:
                        ax.text(j, i, 'F', va='center', ha='center')
                    elif self.O[i,j] == -1:
                        ax.text(j, i, 'X', va='center', ha='center', color='red')
                    else:
                        ax.text(j, i, self.O[i,j], va='center', ha='center')

        # Update the colors
        im.set_data(self.O)
        plt.draw()

    def print_env(self):
        ''' Print a visual representation of the environment. '''
        print()
        print("\t\t\tMINESWEEPER\n")
    
        # row numbers
        st = "   "
        for i in range(self.n_cols):
            st = st + '{:6}'.format(i)
        print(st)   
    
        for r in range(self.n_rows):
            # top horizontal line
            st = "     "
            if r == 0:
                for col in range(self.n_cols):
                    st = st + "______" 
                print(st)
    
            # first line
            st = "     "
            for col in range(self.n_cols):
                st = st + "|     "
            print(st + "|")
            
            # second line with line number and content
            st = '{:4}'.format(r) + " "
            for col in range(self.n_cols):
                if self.O[r,col] != -2:
                    if self.O[r,col] == -3:
                        st = st + "|  F  "
                    elif self.O[r,col] == -1:
                        st = st + "|  X  "
                    else:
                        st = st + "|  " + str(self.O[r,col]) + "  "
                else:
                    st = st + "|     "
            print(st + "|") 
    
            # third line with bottom line
            st = "     "
            for col in range(self.n_cols):
                st = st + "|_____"
            print(st + '|')
        
        print()
        return 0

    def wait4input(self):
        '''
            Wait for the user to input the tile to reveal.
            
            Returns
            -------
            i, j : int
                the row and column of the tile to reveal
            flag : bool
                if True, the user wants to flag the tile
        '''
        while True:
            # Input from the user
            inp = input("Enter row and column (add F to flag a cell) = ").split()

            if len(inp) == 2:
                flag = False
                # Try block to handle errant input
                try: 
                    val = list(map(int, inp))
                except ValueError:
                    clear_terminal()
                    self.print_env()
                    print("Wrong input: Please enter two numbers separated by a space")
                    continue

            elif len(inp) == 3 and inp[2] == "F":
                # Try block to handle errant input
                flag = True
                try: 
                    val = list(map(int, inp[:2]))
                except ValueError:
                    clear_terminal()
                    self.print_env()
                    print("Wrong input: Please enter two numbers")
                    continue

            else:
                clear_terminal()
                self.print_env()
                print("Wrong input: Please enter two numbers separated by a space")
                continue

    
            # Get row and column number
            i, j = val

            # Check if  within the grid
            if i >= self.n_rows or i < 0 or j > self.n_cols or j < 0:
                clear_terminal()
                self.print_env()
                print("Wrong Input: indices out of bound")
                continue

            # Check if the tile is already uncovered
            if self.O[i,j] > -2:
                clear_terminal()
                self.print_env()
                print("Tile already uncovered")
                continue

            return i,j,flag
        

if __name__ == "__main__":     

    # creating the environment
    game = Minesweeper("beginner", display=True)

    # Variable for maintaining Game Loop
    over = False
		
	# The GAME LOOP	
    while not over:
        game.print_env()
        game.update_display()
        i, j, flag = game.wait4input()
        over = game.take_action((i,j), flag=flag, uncover_neighbors=True)

        # If landing on a mine --- GAME OVER	
        if over == 1:
            print("Landed on a mine. GAME OVER!!!!!")
            game.show_mines()
            continue

        elif over == 2:
            print("Congratulations!!! YOU WON")
            continue
        clear_terminal()

    game.update_display()
    # wait for keypress to quit
    input()
    plt.close()