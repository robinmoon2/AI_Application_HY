# 🎮 Predict Game Winners
> **Disclaimer:** This project follows a previous one on student performance, which we found less engaging ([Student performance](Data%20Analysis)).

## 👥 Team

| Name             | School                   | Email                        |
|------------------|--------------------------|------------------------------|
| **Théo Hardy**    | ESILV Engineering School | theo.hardy@edu.devinci.fr     |
| **Gaël Le Mouel** | ESILV Engineering School | gael.lemouel@gmail.com        |
| **Robin L'hyver** | ESILV Engineering School | robinlhyver@gmail.com         |

---

## 📊 Data Set

**Data Set used :**

[![Kaggle Data Set (Kaggle)](https://img.shields.io/badge/Kaggle%20Data%20Set-Kaggle-blue?style=flat-square)](https://www.kaggle.com/datasets/datasnaek/chess/data)
[![Elite Data Set (Nikonoel)](https://img.shields.io/badge/Elile%20Data%20Set-Nikonoel-blue?style=flat-square)](https://database.nikonoel.fr/)


### Data Set Description

The Kaggle Data Set contains around **20,000 chess games** with the following columns:
- `id`: Game ID
- `rated`: Whether the game is rated
- `created_at`: Date and time the game was created
- `last_move_at`: Date and time of the last move
- `turns`: Number of turns in the game
- `victory_status`: Game outcome (mate, draw, resign, outoftime)
- `winner`: Winner of the game (white, black, draw)
- `increment_code`: Increment code ([time control](https://www.chess.com/terms/chess-time-controls))
- `white_id`: White player ID
- `white_rating`: White player rating
- `black_id`: Black player ID
- `black_rating`: Black player rating
- `moves`: Moves in standard chess notation 
- `opening_eco`: Opening code ([List](https://www.365chess.com/eco.php))
- `opening_name`: Opening name
- `opening_ply`: Number of opening moves

The data set comes from the [Lichess](https://lichess.org/) website, which is a free online chess game platform.

After analysing the data, we found that the Kaggle dataset do not contain enough observations.
Lichess provides a database of **Every game played on Lichess** on its [webstie](https://database.lichess.org/).
To decided to use the **Lichess Elite Database** which contains all the 2300 elo or more game datas. 
Elo corresponds to the player's rating, which is a measure of the player's skill level. The higher the Elo, the better the player.
2300 elo is the minimum rating to be considered a ***master*** in chess. 
The high quality games in the database might help us to have a more accurate prediction.


---

## 🔍 Project Overview

This project explores the various factors that affect **game outcomes**. Using **machine learning** techniques, we aim to:

- 🏆 **Predict the winner** of a game based on key factors from the dataset.
- 🏅**Classify the best opening moves** to increase the chances of winning.
- ❓**Identify Blunders or Critical Position** to detect the **turning points** of the game.
- ♟️**Evaluate the best move** to make in a given position.
- 📈 Assess the **influences** on game outcomes, including player statistics and game conditions.
- 🤖 Evaluate whether the **dataset** is reliable for making accurate predictions using machine learning models.
- 🚀 **Improve our understanding of AI** and its applications in analyzing game data.

### Why This Project? 🎯

We are big fans of games and are interested in understanding the factors that contribute to winning. This project helps us explore how factors like **player ratings**, **game time**, and **opening moves** influence the outcome of a game.
As computer science students, we aim to **enhance our knowledge of artificial intelligence** by applying it to real-world data and evaluating how effective the dataset is in making reliable predictions.

## 🔬 Methodology
### Data Preprocessing
TBW

### Existing Models

Chess is very famous in the AI community, and many models have been developed to predict the best move in a given position such as **AlphaZero** and **Stockfish**. To help us develop our own, we will use these models. (see more : [Chess Engine](Chess_Project/Chess%20Engine))

#### AlphaZero

AlphaZero is a computer program developed by Google DeepMind in 2017. It uses the same reinforcement learning techniques as AlphaGo Zero, but it is also trained on chess. AlphaZero is a general-purpose algorithm that can learn to play other games as well. It uses a deep neural network to evaluate positions and select moves.

#### Stockfish

Stockfish is a free and open-source chess engine developed by Tord Romstad, Marco Costalba, and Joona Kiiski. It is one of the strongest chess engines in the world. Stockfish uses alpha-beta pruning and other techniques to evaluate positions and select moves. The evaluation function is based on material balance, piece mobility, pawn structure, king safety, and other factors and is entirely handcrafted.

#### Which one is better ?

TBW

## 📈 Data Analysis
TBW

## Predicting the winner and the victory status of a game.

To predict the winner and victory status we tried three strategies in order to see which one had the best accuracy. To do the prediction we used at must 5 out of 16 features : the number of turns in the game, the players ratings, the opening code used and the number of play of this opening. We chose the Random Forest algorithm because it is the best supervised classification algorithm in our opinion.
The three stratgies were the following : 
- prediction with opening code feature encoded with binary encoding, 
- prediction with opening code feature encoded with label encoding, 
- prediction with only the players ratings and the number of turns in the game.

In the first strategy we used Binary encoding to use the opening code feature. This type of encoding allow us to reduce the dimensionality compare to a one-hot encoding but still use a powerful encoding like the on-hot one. We found a 36% accuracy of the Random Forest algorithm.
For the second strategy we used a Label encoding this time of the opening feature. It does not increase the dimensionality of our dataset but it will create a little bias in the algorithm as the label encoding is used for ordered feature and the opening code is not one of this kind. The random forest algorithm give us a 37% accuracy.
The last strategy was to reduce the number of feature to only the one which are really correlated to the winner and victory status of the game. We found a 35% accuracy.

In conlusion, although the random forest is the best supervised algorithm for classification problem it appears that it cannot predict the winner and the victory status of the game easily.

---

## References
- [Complete guide to encoding categorical features](https://kantschants.com/complete-guide-to-encoding-categorical-features)
- [Elo Rating System in Chess](https://www.chess.com/terms/elo-rating-chess)