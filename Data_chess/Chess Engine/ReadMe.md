# Chess Engines 

## Introduction
This repository contains the source code of the chess engines that I have developed. The engines are written in C++ and use the Universal Chess Interface (UCI) protocol to communicate with the GUI. The engines are designed to be simple and easy to understand.

## Engines
1. **Stockfish** - A  chess engine that uses the minimax algorithm with alpha-beta pruning to search the game tree. It also uses a simple evaluation function to evaluate the positions.
2. **LCZero** - A  chess engine that uses a neural network to evaluate the positions. The neural network is trained using reinforcement learning.


## Usage 
For better performances of `lc0` engine, you might consider using better pretrained network that the one by default. You can download the network from the [LCZero website](https://lczero.org/play/networks/bestnets/). Just download the network and put it in the `LC0` folder that contains the `.exe` file.
- `lc0.py` and `stockfish.py` are two demo python script that we used to test the engines.
- `Game.py` is a python script that can be used to make two engines play against each other. 
  - To Launch games, run the following command:
    ```bash
    python Game.py <number_of_games> <time_limit_per_player> <engine1> <engine2> <displayLive>
    ``` 
    - `<number_of_games>`: Number of games to be played.
    - `<time_limit_per_player>`: Time limit for each player in seconds.
    - `<engine1>`: Engine 1 (stockfish or lc0). The engines can be the same.
    - `<engine2>`: Engine 2 (stockfish or lc0). The engines can be the same.
    - `<displayLive>`: Display the game live or not (True or False) (Requires to setup a chess GUI).
    - Example:
    ```bash
    python Game.py 10 1 stockfish lc0 True
    ```
    
## Why This program ?
We created this game program for two main reasons :
1. To compare the performance of the two engines.
2. To provide us more data for our prediction model.

## References
1. [Stockfish](https://stockfishchess.org/)
2. [LCZero](https://lczero.org/)
3. [How do Chess Engines work? Looking at Stockfish and AlphaZero | Oliver Zeigermann (Youtube)](https://youtu.be/P0jd8AHwjXw)