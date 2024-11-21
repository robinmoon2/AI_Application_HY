# 🎮 Predict Game Winners

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

> **Disclaimer:** This project follows a previous one on student performance, which we found less engaging ([Student performance](Data%20Analysis)).
---

## References
- [Complete guide to encoding categorical features](https://kantschants.com/complete-guide-to-encoding-categorical-features)
- [Elo Rating System in Chess](https://www.chess.com/terms/elo-rating-chess)