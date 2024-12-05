import sys
import chess
import chess.engine
import chess.pgn
import pandas as pd
import numpy as np
import os
import time


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm


import matplotlib.pyplot as plt
import seaborn as sns

def forest_test(X, y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.20, random_state=42)
    start = time.process_time()
    trained_forest = RandomForestClassifier(class_weight="balanced", random_state=42).fit(X_Train, Y_Train)
    print(f"Training time: {time.process_time() - start:.2f} seconds")

    prediction_forest = trained_forest.predict(X_Test)

    # Print confusion matrix and classification report
    print("Confusion Matrix:")
    print(confusion_matrix(Y_Test, prediction_forest))
    print("\nClassification Report:")
    print(classification_report(Y_Test, prediction_forest))

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(Y_Test, prediction_forest)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Plot feature importance
    feature_importance = trained_forest.feature_importances_
    features = X.columns
    indices = np.argsort(feature_importance)[::-1]

    plt.figure(figsize=(12, 10))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), feature_importance[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()

    print()

def gradient_boosting_test(X, y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.20, random_state=42)
    start = time.process_time()
    gbc = GradientBoostingClassifier().fit(X_Train, Y_Train)
    print(f"Training time: {time.process_time() - start:.2f} seconds")

    prediction_ngb = gbc.predict(X_Test)

    # Print confusion matrix and classification report
    print("Confusion Matrix:")
    print(confusion_matrix(Y_Test, prediction_ngb))
    print("\nClassification Report:")
    print(classification_report(Y_Test, prediction_ngb))

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(Y_Test, prediction_ngb)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class'])

    print()

def support_vector_machine_test(X, y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.20, random_state=42)
    start = time.process_time()
    svc = svm.SVC(class_weight="balanced").fit(X_Train, Y_Train)
    print(f"Training time: {time.process_time() - start:.2f} seconds")

    prediction_svc = svc.predict(X_Test)

    # Print confusion matrix and classification report
    print("Confusion Matrix:")
    print(confusion_matrix(Y_Test, prediction_svc))
    print("\nClassification Report:")
    print(classification_report(Y_Test, prediction_svc))

    print()

def initialize_stockfish():
    if sys.platform.startswith("linux"):
        stockfish_path = "Chess Engine/stockfish/stockfish-ubuntu-x86-64-avx2"
    elif sys.platform.startswith("win"):
        stockfish_path = "Chess Engine/stockfish/stockfish-windows-x86-64-avx2.exe"
    else :
        print("Unsupported OS")
        return None
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    return engine

def initialize_lc0():
    if sys.platform.startswith("win"):
        lc0_path = "Chess Engine/lc0/lc0.exe"
    else :
        print("Unsupported OS")
        return None
    engine = chess.engine.SimpleEngine.popen_uci(lc0_path)
    return engine

def get_stockfish_evaluations(moves: str, depth: int = 15):
    board = chess.Board()
    evaluations = ""
    for move in moves.split()[:len(moves.split()) // 2]:
        board.push_uci(move)
        info = sf_engine.analyse(board, chess.engine.Limit(depth=depth))
        evaluations += str(info["score"].white().score()) + " "
    return evaluations

def get_lc0_evaluations(moves: str, depth: int = 15):
    board = chess.Board()
    evaluations = ""
    for move in moves.split()[:len(moves.split()) // 2]:
        board.push_uci(move)
        info = lc0_engine.analyse(board, chess.engine.Limit(depth=depth))
        evaluations += str(info["score"].white().score()) + " "
    return evaluations

def extract_trends(df):
    if not df['evaluations'].any():
        return df
    index = 0
    for evaluation in df['evaluations']:
        evaluation = str(evaluation).split()
        try:
            evaluation = [int(e) for e in evaluation]
        except ValueError:
            df = df.drop(index)
            index +=1
            continue

        df.at[index, 'avg_eval'] = np.mean(evaluation)
        df.at[index, 'max_eval'] = np.max(evaluation)
        df.at[index, 'min_eval'] = np.min(evaluation)
        df.at[index, 'eval_diff'] = evaluation[-1] - evaluation[0]
        df.at[index, 'eval_variance'] = np.var(evaluation)
        # Rate of change
        rate_diff = np.diff(evaluation)
        df.at[index, "avg_rate_change"] = np.mean(rate_diff)
        df.at[index, "max_rate_change"] = np.max(rate_diff)
        # sign changes
        df.at[index,"sign_changes"] = np.sum(np.diff(np.sign(evaluation))!=0)
        # time in advantage
        df.at[index, "white_advantage_time"] = np.sum(np.array(evaluation) > 0)/len(evaluation)
        # eval_range
        df.at[index, "eval_range"] = np.max(evaluation) - np.min(evaluation)

        index+=1
    return df

def get_eval_csv(row_number=10):
    df = pd.read_csv('Data/elite_chess_games_moves.csv').head(row_number)

    one_percent = len(df) // 100
    if one_percent == 0:
        one_percent = 1
    # Loop through the DataFrame in chunks of 1%
    start_time = time.time()
    evaluation_list = []
    for index, data in df.iterrows():
        moves = data['moves']
        evaluations = get_stockfish_evaluations(moves)
        evaluation_list.append(evaluations)
        if (index + 1) % one_percent == 0:
            elapsed_time = time.time() - start_time
            print(f"Processed {index + 1} games in {elapsed_time:.2f} seconds")
    df['evaluations'] = evaluation_list
    df.to_csv('Data/elite_chess_games_evaluations.csv', index=False)
    print(df['evaluations'])
    print("done\n Data saved to 'Data/elite_chess_games_evaluations.csv'")
    return df

def get_trends_csv():
    df = pd.read_csv('Data/elite_chess_games_evaluations_1k_rows.csv')
    df = extract_trends(df)
    df.to_csv('Data/elite_chess_games_trends.csv', index=False)
    print("done\n Data saved to 'Data/elite_chess_games_trends.csv'")
    return df


if __name__ == "__main__":
    sf_engine = initialize_stockfish()
    lc0_engine = initialize_lc0()
    extract = True
    game_to_extract = 10
    try :
        if str(sys.argv[1]).lower() in ["false"]:
            extract = False
    except IndexError:
        pass
    try :
        game_to_extract = int(sys.argv[2])
    except ValueError :
        print("Invalid argument, using default value of 10")
    except IndexError:
        pass
    if extract :
        get_eval_csv(game_to_extract)
    get_trends_csv()
    sf_engine.quit()

    # Random Forest Model
    print("Random Forest Model : ")
    df = pd.read_csv("Data/elite_chess_games_trends.csv")
    # Features and labels
    X = df[["avg_eval", "max_eval", "min_eval", "eval_diff", "eval_variance", "avg_rate_change", "max_rate_change", "sign_changes", "white_advantage_time", "eval_range"]]
    y = df["winner"]
    forest_test(X, y)

    # Gradient Boosting Model
    print("Gradient Boosting Model : ")
    gradient_boosting_test(X,y)

    # Support Vector Machine Model
    print("Support Vector Machine Model : ")
    support_vector_machine_test(X, y)

    # Random Forest Model without draws
    print("Random Forest Model without draws : ")
    df = df[df['winner']!=0]
    X = df[["avg_eval", "max_eval", "min_eval", "eval_diff", "eval_variance", "avg_rate_change", "max_rate_change", "sign_changes", "white_advantage_time", "eval_range"]]
    y = df["winner"]
    forest_test(X, y)

    # Gradient Boosting Model
    print("Gradient Boosting Model without draws : ")
    gradient_boosting_test(X, y)

    # Support Vector Machine Model
    print("Support Vector Machine Model without draws : ")
    support_vector_machine_test(X, y)





