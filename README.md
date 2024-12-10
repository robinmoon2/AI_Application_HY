# 🎮 Chess and AI
> **Disclaimer:** This project follows a previous one on student performance, which we found less engaging ([Student performance](Old_Project)).

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
- 100k row move list CSV sample extracted from the PGN files : [100k Games Elite Data Set (CSV)](Chess_Project/Data/elite_chess_games_moves_100k_Games.csv)


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

[AlphaZero (Deepmind)](https://deepmind.google/discover/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/) is a computer program developed by Google DeepMind in 2017.
It uses the same reinforcement learning techniques as **AlphaGo**, but it is also trained on chess.
AlphaZero is a general-purpose algorithm that can learn to play other games as well. It uses a deep neural network to evaluate positions and select moves.

#### Stockfish

Stockfish is a free and open-source chess engine developed by Tord Romstad, Marco Costalba, and Joona Kiiski.
At its beginning in 2008, Stockfish was a chess engine that used the minimax algorithm with alpha-beta pruning to search the game tree. 
It also used a simple evaluation function to evaluate the positions.
The engine was entirely handcrafted, and the evaluation function was based on the knowledge of the game's authors.
With the introduction of [efficiently updatable neural network (wikipedia)](https://en.wikipedia.org/wiki/Efficiently_updatable_neural_network)
Stockfish has been able to use a neural network to evaluate the positions.

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
For our models, we will use Stockfish as it is more efficient and easier to run on mid-range computers.

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

### Predicting the winner only with the early moves, without information about the players.

As we saw in the previous sections, the players' ratings and their difference are often the most important features to predict the winner of a game.
We wanted to go further and see if we could **predict the winner of a game only with the first moves of the game**.

The problem that we had face is that the moves themselves as they are written in our CSV file are not enough to predict the winner of the game.
We decided to use pretrained models to put some numerical values on the moves.
[Stockfish](https://stockfishchess.org/) engine is what we used to evaluate the position of the game after each move.
Stockfish is really powerful. However, even thought we limited the time to evaluate each position, it took a lot of time to evaluate all the games.
We decided to use a sample of 1500 games to evaluate the moves.

In order to verify the accuracy of Stockfish evaluation, the first thing that we did was to take the last evaluation of the game and compare it to the winner of the game.
We found that the Stockfish evaluation were very precise. Here are the results :

![Simple Model Result](images/Simple_Model_Result_Stockfish.png)
*Simple model results with Stockfish evaluation*

Stockfish evaluation is a number between -1000 and 1000. The evaluation is positive when white is winning and negative when black is winning.
The draw Margin is the margin that we consider as a draw. If the evaluation is between `-draw_margin` and `draw_margin`, we consider the game as a draw.
We obtain a **86% accuracy** with a draw margin of 12. The accuracy is very good, therefore we can consider that the **Stockfish evaluation is precise enough for our model**.

We needed to do some **feature engineering** (mean, variance, sign_changes, etc...) to extract trends from the stockfish evaluation.
As we did not want to take into account the end of the game, we took a percentage of the moves from the beginning of the game.
Once the trends were extracted, we used some simple models from the [scikit-learn](https://scikit-learn.org/) library to predict the winner of the game:
- **Random Forest**
- **Gradient Boosting**
- **Support Vector Machine** with a linear kernel

Here are our first results :

![Pretrained model performances](images/Pretrained_Model_Performances.png)
*Performances of the pretrained models depending on the percentage of the move taken into account*

Those graphs evidences some points :
- The **Support Vector Machine** model is properly set up as the prediction accuracy stay below 50%.
This can also be explained by the fact that in this first attempt, we considered the draws in the dataset, although it 
represents only 10% of our data. 
- For our other models, to reach 55% of accuracy, we need to take half of the game into account.
- The maximum accuracy is around 75% for the **Random Forest** and **Gradient Boosting** models.
- Our models seem te biased as the curves are clearly not smooth. 

After some adjustments such as :
- **Removing the draws** from the dataset
- Taking the time to extract **5000 games evaluation** from the PGN files and stockfish to have a more consistent dataset
- **Scaling the features** to have a better performance

We obtain the following results:

![Pretrained model performances 5000 drawless](images/Pretrained_Model_Performances_5000_games.png)
*Performances of the pretrained models depending on the percentage of the move taken into account with adjustments*

Performances seems better :
- The three models have almost the same accuracy that increases with the number of moves taken into account even though the models are different in their approach. 

- The maximum accuracy is around **78% for the three models** taking 80% of the games into account.
- The seems to be **more consistent** as the curves are smoother.
- The **computation time to train the three models is very low** once the evaluations extracted (less than 1 sec for 4000 games).

However, we have to note that removing the draws from the dataset means that the mmodel only have to do a binary classification.
This means that 50% accuracy is the minimum accuracy that we can have and therefore is not a good performance. 
Based on this assumption, we can say that our models are not good enough to predict the winner of a game only with the early moves.

With our results, we can say that we can predict the outcome of a game with 60% accuracy with half of the game...
However, even if we know that our dataset has a game length of 82 moves on average, we don't really know the exact move count of a game in live.
We would like to know how many moves (in absolute value) we need to predict the outcome of a game with a 60% accuracy.

Here are the results after computation:

![Pretrained Model Performances on 5000 games absolute](images/Pretrained_Model_Performances_absolute.png)
*Performances of the pretrained models depending on the move ratio taken into account on 5000 games*

We can note that the accuracy seems to have **reached its peak around 85%** for 80 games... Which seems logic as we evaluated the precision of Stockfish evaluation to 86% and the mean game length is 82 moves.
We can say that we can predict the outcome of a game with 70% accuracy with 40 moves.

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
We took a closer look into the subject and watched more serious videos and papers (see [References](#LLM-and-Chess)), it turns out that we need to consider some other factors.
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
Other advantage of this technique is that the model have the complete context of the game as all the moves are given to it.

We tried to use this technique to make a GPT model predict the winner of a game. We used the Gemini model from Google.

---

## 📖References

- [Complete guide to encoding categorical features](https://kantschants.com/complete-guide-to-encoding-categorical-features)
- ### Data Preprocessing (Useful resources about Chess Data)
  - [Portable Game Notation (Chess.com)](https://www.chess.com/terms/chess-pgn)
  - [ECO Codes (365Chess.com)](https://www.365chess.com/eco.php)
  - [Chess Time Controls (Chess.com)](https://www.chess.com/terms/chess-time-controls)
  - [Elo Rating System in Chess (Chess.com)](https://www.chess.com/terms/elo-rating-chess)
  - [Chess Notation (Chess.com)](https://www.chess.com/terms/chess-notation)
- ### Chess Engines
  - [Stockfish](https://stockfishchess.org/)
  - [LCZero](https://lczero.org/)
  - [AlphaZero: Shedding new light on chess, shogi, and Go (DeepMind, 2018)](https://deepmind.google/discover/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/)
  - [Remove classical evaluation (Github, 2023)](https://github.com/official-stockfish/Stockfish/commit/af110e02ec96cdb46cf84c68252a1da15a902395)
  - [How do Chess Engines work? Looking at Stockfish and AlphaZero | Oliver Zeigermann (Youtube, 2019)](https://youtu.be/P0jd8AHwjXw)
- ### ML Models 
  - [Gradient Boosting vs Random Forest (Geeksforgeeks.org,2024)](https://www.geeksforgeeks.org/gradient-boosting-vs-random-forest/)
- ### LLM and Chess
  - [Is LLM Chess the FUTURE of the Game or a Total Flop?(YouTube, 2024)](https://youtu.be/vBCZj5Yp_8M)
  - [Playing chess with large language models (Carlini, 2023)](https://nicholas.carlini.com/writing/2023/chess-llm.html)
  - [OK, I can partly explain the LLM chess weirdness now (Dynomight, 2024)](https://dynomight.net/more-chess/#parting-thoughts)
  - [Grandmaster-Level Chess Without Search (Deepmind, 2024)](https://arxiv.org/pdf/2402.04494v1)
  - [Amortized Planning with Large-Scale Transformers: A Case Study on Chess (Deepmind, 2024)](https://arxiv.org/pdf/2402.04494)