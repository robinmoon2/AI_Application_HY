# 🎮 Chess and AI
> **Disclaimer:** This project follows a previous one on student performance, which we found less engaging ([Student performance](Data%20Analysis)).

## 👥 Team

| Name              | School       | Email                         |
|-------------------|--------------|-------------------------------|
| **Théo Hardy**    | ESILV Paris  | theo.hardy@edu.devinci.fr     |
| **Gaël Le Mouel** | ESILV Paris  | gael.lemouel@gmail.com        |
| **Robin L'hyver** | ESILV Paris  | robinlhyver@gmail.com         |

---

## 💻 Project Overview

This project explores the various factors that affect **game outcomes**. Using **machine learning** techniques, we aim to:

- 🏆 **Predict the winner** of a game based on key factors from the dataset.
- 🏅**Classify the best opening moves** to increase the chances of winning.
- ❓**Identify Blunders or Critical Position** to detect the **turning points** of the game.
- 📈 Assess the **influences** on game outcomes, including player statistics and game conditions.
- 🤖 Evaluate whether the **dataset** and the **models** that we chose are reliable for making accurate predictions using machine learning models.
- 🚀 **Improve our understanding of AI** and its applications in analyzing chess data.

### Why This Project? 🎯

We are big fans of games and are interested in understanding the factors that contribute to winning. This project helps us explore how factors like **player ratings**, **game time**, and **opening moves** influence the outcome of a game.
As computer science students, we aim to **enhance our knowledge of artificial intelligence** by applying it to real-world data and evaluating how effective the dataset is in making reliable predictions.


## 📊 Data Set

**Data Set used :**

[![Kaggle Data Set (Kaggle)](https://img.shields.io/badge/Kaggle%20Data%20Set-Kaggle-blue?style=flat-square)](https://www.kaggle.com/datasets/datasnaek/chess/data)
[![Elite Data Set (Nikonoel)](https://img.shields.io/badge/Elile%20Data%20Set-Nikonoel-blue?style=flat-square)](https://database.nikonoel.fr/)


### Kaggle Data Set

The Kaggle Data Set contains around **20,000 chess games** with the following features:
- `id`: Game ID
- `rated`: Whether the game is rated
- `created_at`: Date and time the game was created
- `last_move_at`: Date and time of the last move of the game
- `turns`: Move number in the game
- `victory_status`: Game outcome (mate, draw, resign, outoftime)
- `winner`: Winner of the game (white, black, draw)
- `increment_code`: Time control, first number is the time in minutes  per player and the second one is the number of second added to each player time for every moves ([Time control](https://www.chess.com/terms/chess-time-controls))
- `white_id`: White player ID
- `white_rating`: White player rating
- `black_id`: Black player ID
- `black_rating`: Black player rating
- `moves`: Moves in standard chess notation 
- `opening_eco`: Opening code ([Opening List](https://www.365chess.com/eco.php))
- `opening_name`: Opening name
- `opening_ply`: move number per opening

The data set comes from the [Lichess](https://lichess.org/) website, which is a free online chess game platform.

After analysing the data, we found that the Kaggle dataset do not contain enough data.


### Elite Data Set

Lichess provides a database of **Every game played on Lichess** on its [webstie](https://database.lichess.org/).
We decided to use the **Lichess Elite Database** created by a user of the Lichess API. The data set contains all the 2300 elo or more game datas. 
Elo corresponds to the player's rating, which is a measure of the player's skill level. The higher the Elo, the better the player.
2300 elo is the minimum rating to be considered a ***master*** in chess. 
The high quality games in the database might help us to have a more accurate prediction.

The Elite Data Set contains games in the format of **PGN** [Portable Game Notation (Chess.com)](https://www.chess.com/terms/chess-pgn) files. 
The PGN format is a standard format for recording chess games. Here is an example of a PGN file with its features (*text in curly brackets are comments*):

```PGN
[Event "Rated Blitz game"]    {Event name}
[Date "2020.02.01"]           {Date of the game}
[Round "-"]                   {Round of the event (if available)}
[White "bluepower"]           {White player name}
[Black "Piratalokoo"]         {Black player name}
[Result "0-1"]                {Result of the game}
[WhiteElo "2299"]             {White player Elo}
[BlackElo "2469"]             {Black player Elo}
[ECO "A15"]                   {Opening code}
[Opening "English Opening: Anglo-Indian Defense, King's Indian Formation"]
[TimeControl "180+0"]         {Time control}
[UTCDate "2020.02.01"]      
[UTCTime "00:00:11"]
[Termination "Normal"]        {Game termination status}
[WhiteRatingDiff "-3"]        {White player rating change after the game}
[BlackRatingDiff "+4"]        {Black player rating change after the game}

{All the moves of the game}
1. Nf3 Nf6 2. c4 g6 3. Nc3 d5 4. cxd5 Nxd5 5. Qb3 Nb6 6. d4 Be6 7. Qd1 c5
8. dxc5 Qxd1+ 9. Nxd1 Na4 10. Be3 Nd7 11. b4 Bg7 12. Rc1 Bxa2 13. Bd4 e5
14. Ba1 O-O 15. e4 a5 16. Bb5 Bh6 17. Ne3 axb4 18. Bxd7 f6 19. Bxa4 Rxa4
20. O-O b3 21. Rc4 Rfa8 22. Ng4 Rxc4 23. Nxh6+ Kg7 24. Ng4 Rxe4 25. Ne3 Rc8
26. Rc1 Rb4 27. Bb2 Rb5 28. Nd2 Rbxc5 29. Re1 Rc2 30. Nxc2 Rxc2 0-1
```

A big part of the features provided in the PGN files do not interest us for our analysis.
**We will mostly focus on the features that we found in the Kaggle dataset.**
To use the data set, we first converted the data from the PGN files to a new dataset in CSV format.
To do this, we used the python programs in the [Elite PGN files](Chess_Project/Elite%20PGN%20files) folder.
The programs use the [python-chess](https://python-chess.readthedocs.io/en/latest/) library that provides tools to easily read and write PGN files.
One of the programs is specifically designed to extract the game move list alongside the outcome of the game from the PGN files. 
The other extracts other features such as the *players' ratings*, the *opening code*, the *turn number* and the *victory status*.
As the programs were taking a lot of time to convert the data, we decided to use samples of the data to test our models.
We first extracted 10000 games from the PGN files to test our models and then extracted 1,000,000 games.

- 1M row CSV sample extracted from the PGN files : [1M Games Elite Data Set (CSV)](Chess_Project/Data/elite_chess_games_features-1M_Games.zip)
- 100k row move list CSV sample extracted from the PGN files : [100k Games Elite Data Set (CSV)](Chess_Project/Data/elite_chess_games_moves.csv)


---

## 🔬 Methodology
### Data Preprocessing

Before beginning the statistic analysis we needed to clean our datasets. 
For instance, we erased rows with **missing values** and deleted **duplicate rows**.

More of that, we have done some **feature engineering** to improve the quality of our dataset.

To train models, we needed to encode the categorical features.
We principally used the **One-Hot Encoding** technique to encode the categorical features in the dataset. 
This technique converts each category value into a new column and assigns a 1 or 0 (True/False) value to the column.


### Existing Models

Chess is very famous in the AI community, and many models have been developed to predict the best move in a given position such as **AlphaZero** and **Stockfish**. 
To help us develop our own, we will use these models. (see more : [Chess Engine](Chess_Project/Chess%20Engine))

#### AlphaZero

[AlphaZero (Deepmind)](https://deepmind.google/discover/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/) is a computer program developed by Google DeepMind in 2017. It uses the same reinforcement learning techniques as **AlphaGo**, but it is also trained on chess.
AlphaZero is a general-purpose algorithm that can learn to play other games as well. It uses a deep neural network to evaluate positions and select moves.

#### Stockfish

Stockfish is a free and open-source chess engine developed by Tord Romstad, Marco Costalba, and Joona Kiiski.
At its beginning in 2008, Stockfish was a chess engine that used the minimax algorithm with alpha-beta pruning to search the game tree. It also used a simple evaluation function to evaluate the positions.
The engine was entirely handcrafted, and the evaluation function was based on the knowledge of the game's authors.
With the introduction of [efficiently updatable neural network (wikipedia)](https://en.wikipedia.org/wiki/Efficiently_updatable_neural_network)
Stockfish has been able to use a neural network to evaluate the positions [(Remove classical evaluation)](https://github.com/official-stockfish/Stockfish/commit/af110e02ec96cdb46cf84c68252a1da15a902395).

#### Which one is better ?
According to google, Alpha zero might be better than Stockfish as this very pretty looking animation shows:
![AlphaZero vs Stockfish](images/AphaZeroPerformances.gif)
*Alpha Zero performances according to Google DeepMind*

However, there are few parameters to take into account when comparing the two engines.
The two engines **do not use the same type of Neural Network** when it comes to play the game. 
The "Efficiently updatable neural network" (NNUE) used by Stockfish is very efficient on CPU 
when CNN used by AlphaZero requires a lot of GPU power.

To compare the two engines, we created a python script that makes the two engines play against each other. The script is available in the [Chess Engine](Chess_Project/Chess%20Engine) folder. 
It results that running locally with the *"t3-512x15x16h-distill-swa-2767500"* neural network provided by LCZero on its [website](https://lczero.org/play/networks/bestnets/), Stockfish wins almost every time.

## 📈 Data Analysis

### Important Features :

#### Turns : 

This feature shows us the number of turns in the game. It is a good indicator of the game length. This feature will be useful for the project and for the learning model that we will use.

![turns distribution](images/turns_distribution.png)

We can see a pin around 50 turns . This is normal because the average game length is around 40 turns. Then we have some extra values that are higher than 100 turns. 


#### Ranking : 

The rank of each player is important, it tells us the difference of level between 2 players and there knowledge of the game.

We can see that the white and black rating are the same :

| white_rating | black_rating | raking difference |
|--------------|--------------|-------------------|
![rating distribution](images/white_rating.png)|![rating distribution](images/black_rating.png)|![rating distrib](images/rating_difference.png)

We see that the ranking difference between the players is not important and we can assume that the players have the same levels.


#### correlation matrix

To help us with every features presents in the dataset. With this feature we can see the correlation between each features and the target variable.
![correlation matrix all](images/correlation_all.png)

## 🔎Evaluation & Analysis

### Predicting the winner and the victory status of a game using the dataset features

To predict the winner and victory status we tried three strategies in order to see which one had the best accuracy. To do the prediction we used at must 5 out of 16 features : the number of turns in the game, the players ratings, the opening code used and the number of play of this opening. We chose the Random Forest algorithm because it is the best supervised classification algorithm in our opinion.
The three strategies were the following : 
- prediction with opening code feature encoded with binary encoding, 
- prediction with opening code feature encoded with label encoding, 
- prediction with only the players ratings and the number of turns in the game.

In the first strategy we used Binary encoding to use the opening code feature. This type of encoding allow us to reduce the dimensionality compare to a one-hot encoding but still use a powerful encoding like the on-hot one. We found a 36% accuracy of the Random Forest algorithm.
For the second strategy we used a Label encoding this time of the opening feature. It does not increase the dimensionality of our dataset but it will create a little bias in the algorithm as the label encoding is used for ordered feature and the opening code is not one of this kind. The random forest algorithm give us a 37% accuracy.
The last strategy was to reduce the number of feature to only the one which are really correlated to the winner and victory status of the game. We found a 35% accuracy.

In conclusion, although the random forest is the best supervised algorithm for classification problem it appears that it cannot predict the winner and the victory status of the game easily.

### Predicting the winner

### Large Language Models and Chess 

We tried to use **LLM models** to see if they could play chess or predict game outcomes. 
We first tried using [Google Gemini API](https://aistudio.google.com/apikey) asking it to evaluate the position of a game as Stockfish would have done it. 
At our surprise, **the model literally told that it was not able to perform good on the task and that we should use a chess engine instead**.
Let's dive into the subject to see why : 

We took a look at this funny video on YouTube : [Google vs ChatGPT: INSANE CHESS (YouTube, 2023)](https://youtu.be/FojyYKU58cw).
It shows an example of two LLM models playing chess against each other.
The Youtuber used the ChatGPT model to play against the Bard model (older version of Gemini).
The very first moves are not that bad as they are from very common openings. 
However, the game quickly became... *chaotic*. It started with this move from ChatGPT :
[![ChatGPT move](images/ChatGPTWeirdMove.png)](https://youtu.be/FojyYKU58cw?t=98)
*ChatGPT illegal move in this sample game against Bard*

ChatGPT decided to play a pawn backward which is not allowed in chess.
As the game goes on, the moves are more and more chaotic and the game ends up in a draw by Bard...
in a very advantageous position for it. The model claimed that the moves were repeated 3 times which was false.

Regarding this game, we could claim that LLMs are not good at playing chess.
Taking a closer look into the subject and watching more serious videos and papers (see [References](#LLM-and-Chess)), it turns out that we need to consider some other factors.
LLM models used in the video are "Chat" models that are trained to put the good words in the right order regarding a **context**. 
These kinds of models can not really "*reason*" about the game and the position of the pieces without a proper context.
As the chess openings are very famous and very well documented, the models can play the first moves of the game without major problems.
However, as the game goes on, the models are not able to reason about the very particular position of the game, and they start to make mistakes.
It is quite understandable as it is estimated there are **around 10<sup>40</sup>** possible legal position in chess board.
This number grows to 10<sup>120</sup> when considering all the possible positions of the pieces on the board 
(including the illegal ones that are clearly taken in account by LLMs as we see in the video).

The best way to make LLMs play chess it to give them a **context**. 
As they are very good at predicting the next word in a sentence, we give them the beginning of a PGN file and ask them to predict the next part of it.
for instance :
```PGN
[White "Garry Kasparov"]
[Black "Magnus Carlsen"]
[Result "1/2-1/2"]
[WhiteElo "2900"]
[BlackElo "2800"]

1. e4
```
Then we ask it to predict the next word.

One of the work that tends to make LLMs very good chess is the one experimented by *Google Deepmind*.

---

## References
- [Complete guide to encoding categorical features](https://kantschants.com/complete-guide-to-encoding-categorical-features)
- [Elo Rating System in Chess](https://www.chess.com/terms/elo-rating-chess)
- ### Chess Engines
  - [Stockfish](https://stockfishchess.org/)
  - [LCZero](https://lczero.org/)
  - [AlphaZero: Shedding new light on chess, shogi, and Go (DeepMind, 2018)](https://deepmind.google/discover/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/)
  - [How do Chess Engines work? Looking at Stockfish and AlphaZero | Oliver Zeigermann (Youtube, 2019)](https://youtu.be/P0jd8AHwjXw)
- ### LLM and Chess
  - [Is LLM Chess the FUTURE of the Game or a Total Flop?(YouTube, 2024)](https://youtu.be/vBCZj5Yp_8M)
  - [Playing chess with large language models (Carlini, 2022)](https://nicholas.carlini.com/writing/2023/chess-llm.html)
  - [OK, I can partly explain the LLM chess weirdness now (Dynomight, 2024)](https://dynomight.net/more-chess/#parting-thoughts)
  - [Grandmaster-Level Chess Without Search (Deepmind, 2024)](https://arxiv.org/pdf/2402.04494v1)
  - [Amortized Planning with Large-Scale Transformers: A Case Study on Chess (Deepmind, 2024)](https://arxiv.org/pdf/2402.04494)