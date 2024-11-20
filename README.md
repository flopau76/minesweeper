# Project

This project investigates machine learning to play minesweeper.

## How to run

You can play the game yourself using the command:
```sh
python3 env.py
```

You can run the different strategies using the command:
```sh
python3 strategy.py
```

where strategy is one of the following:

## Strategies

`RND`:
    Select a random covered tile every turn. This strategy is here only as a baseline

`SPS`:
    Single Point Strategy.

`CSP`:
    Constraint Satisfaction Problem.

`MLP`:
    Multi layer perceptron.

The strategy `CNN` results can be viewed in the notebook `CNN_solver.ipynb`.
Reloading all cells takes at least 45 minutes.
