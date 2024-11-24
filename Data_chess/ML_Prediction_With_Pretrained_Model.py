import chess
import chess.engine
import chess.pgn
import pandas as pd
import numpy as np
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Random Forest
def forest_test(X, y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size = 0.20, random_state = 42)
    start = time.process_time()
    trained_forest = RandomForestClassifier(random_state=42).fit(X_Train,Y_Train)
    print(time.process_time() - start)
    prediction_forest = trained_forest.predict(X_Test)
    print(confusion_matrix(Y_Test,prediction_forest))
    print(classification_report(Y_Test,prediction_forest))

def extract_games(pgn_dir, no_of_games):
    games = []
    games_length = 0

    # Get a list of PGN files sorted by dates
    pgn_files = sorted(
        [os.path.join(pgn_dir, f) for f in os.listdir(pgn_dir) if f.endswith('.pgn')],
        key=lambda x: os.path.basename(x), reverse=True
    )

    start_time = time.time()
    for file in pgn_files:
        with open(file) as pgn_file:
            while games_length < no_of_games:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)
                games_length +=1
                if games_length % (no_of_games//10) == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Processed {games_length} games in {elapsed_time:.2f} seconds")
    print("done")
    print(f"Total games processed: {games_length}")

    return games

# Simple ML model using Stockfish evaluation
def simpleML_model():
## Load the pre-trained model and the dataset
    stockfish_path = "Chess Engine/stockfish/stockfish-windows-x86-64-avx2.exe"  # Update with your path
    df = pd.read_csv("Data/elite_chess_games_moves.csv")

    ## Use the Stockfish engine to evaluate the positions
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    df["score"] = df["10_fen"].apply(lambda x: engine.analyse(chess.Board(x), chess.engine.Limit(time=0.1))["score"].white().score(mate_score=10000))

    ## making predictions based on score
    df["prediction"] = df["score"].apply(lambda x: 1 if x > 0 else -1)

    error = (np.absolute(df["prediction"] - df["winner"])).mean()

    print(f"Total error: {error}")

    engine.quit()

def advancedML_model(no_of_games, load_data = True):
    if(load_data):
        stockfish_path = "Chess Engine/stockfish/stockfish-windows-x86-64-avx2.exe"  # Update with your path

        # More advanced ML models can be used to improve the prediction accuracy.
        pgn_dir = "Test API/Lichess Elite Database/Lichess Elite Database/"

        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        games = extract_games(pgn_dir, no_of_games)

        data = []
        start_time = time.time()
        for game in games:
            board = game.board()
            evaluation_scores = []
            score_len = 0
            # number of moves in the game
            move_count = len(list(game.mainline_moves()))
            for move in game.mainline_moves():
                board.push(move)
                info = engine.analyse(board, chess.engine.Limit(time=0.1))
                score = info["score"].white().score(mate_score=10000)
                evaluation_scores.append(score)
                score_len += 1
                if score_len > move_count // 2:
                    break

            # Trend features
            score_diff = [j - i for i, j in zip(evaluation_scores[:-1], evaluation_scores[1:])]
            avg_score = sum(evaluation_scores) / len(evaluation_scores)
            trend_slope = (evaluation_scores[-1] - evaluation_scores[0]) / len(evaluation_scores)
            # Outcome label
            result = game.headers["Result"]
            if result == "1-0":
                winner = 1
            elif result == "0-1":
                winner = -1
            else:
                winner = 0

            # Add to dataset
            data.append({
                "avg_score": avg_score,
                "trend_slope": trend_slope,
                "final_score": evaluation_scores[-1],
                "winner": winner
            })
            if len(data) % (no_of_games // 100) == 0:
                elapsed_time = time.time() - start_time
                print(f"Processed {len(data)} games in {elapsed_time:.2f} seconds")
        print("done")
        engine.quit()

    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv("chess_game_features.csv", index=False)


    print("Random Forest Model : ")
    df = pd.read_csv("Data/chess_game_features.csv")
    # Features and labels
    X = df[["avg_score", "trend_slope", "final_score"]]
    y = df["winner"]

    forest_test(X, y)

advancedML_model(1000)



