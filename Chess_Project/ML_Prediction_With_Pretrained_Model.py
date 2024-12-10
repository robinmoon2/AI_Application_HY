import sys
import chess
import chess.engine
import chess.pgn
import pandas as pd
import numpy as np
import os
import time
from typing import Union

from Demos.mmapfile_demo import move_dest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaseEnsemble
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm


import matplotlib.pyplot as plt
import seaborn as sns

pd.options.mode.copy_on_write = True


## Models

def test_model(X, y, model : Union[BaseEnsemble, svm.SVC] = RandomForestClassifier()):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.20, random_state=42)
    scaler = StandardScaler()
    X_Train = scaler.fit_transform(X_Train)
    X_Test = scaler.transform(X_Test)
    start = time.process_time()
    trained_model = model.fit(X_Train, Y_Train)
    print(f"Training time: {time.process_time() - start:.2f} seconds")

    prediction_forest = trained_model.predict(X_Test)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(Y_Test, prediction_forest))

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(Y_Test, prediction_forest)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    if isinstance(model, RandomForestClassifier) or isinstance(model, GradientBoostingClassifier):
        # Plot feature importance
        feature_importance = trained_model.feature_importances_
        features = X.columns
        indices = np.argsort(feature_importance)[::-1]

        plt.figure(figsize=(12, 10))
        plt.title("Feature Importances")
        plt.bar(range(X.shape[1]), feature_importance[indices], align="center")
        plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
        plt.xlim([-1, X.shape[1]])
    plt.show()

    print()
    return classification_report(Y_Test, prediction_forest, output_dict=True)

def model_specific_test(X_Train,X_Test, y_Train,y_Test, model : Union[BaseEnsemble, svm.SVC] = RandomForestClassifier()):
    start = time.process_time()
    trained_model = model.fit(X_Train, y_Train)
    print(f"Training time: {time.process_time() - start:.2f} seconds")

    prediction_forest = trained_model.predict(X_Test)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_Test, prediction_forest))

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_Test, prediction_forest)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    print()
    return classification_report(y_Test, prediction_forest, output_dict=True)

## Engines

def initialize_stockfish():
    if sys.platform.startswith("linux"):
        stockfish_path = "Chess Engine/stockfish/stockfish-ubuntu-x86-64-avx2"
    elif sys.platform.startswith("win"):
        stockfish_path = "Chess Engine/stockfish/stockfish-windows-x86-64-avx2.exe"
    else :
        print("Unsupported OS")
        return None
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    print("Engine initialized")
    return engine

def initialize_lc0():
    if sys.platform.startswith("win"):
        lc0_path = "Chess Engine/lc0/lc0.exe"
    else :
        print("Unsupported OS")
        return None
    engine = chess.engine.SimpleEngine.popen_uci(lc0_path)
    print("Engine initialized")
    return engine

def get_evaluations(moves: str, engine : chess.engine.SimpleEngine, time :float = 0.5):
    board = chess.Board()
    evaluations = ""

    try :
        move_to_evaluate = moves.split()
    except Exception as e:
        print(e)
        return evaluations
    # Evaluating the moves
    try :
        for move in move_to_evaluate:
            board.push_uci(move)
            info = engine.analyse(board, chess.engine.Limit(time=time, depth=15))
            evaluations += str(info["score"].white().score()) + " "
    except Exception as e:
        print(e)
    return evaluations


## Simple test model
def simple_model(evaluation_csv : str = "Data/elite_games_evaluations_100_rows_stockfish.csv"):
    MIN_MARGIN = 10
    MAX_MARGIN = 15
    eval_df= pd.read_csv(evaluation_csv)
    X = eval_df["evaluations"]
    y = eval_df["winner"]

    for draw_margin in range(MIN_MARGIN, MAX_MARGIN +1):
        prediction_list = []
        for e in eval_df["evaluations"]:
            try:
                eval_str = e.split()
            except (AttributeError,TypeError):
                prediction_list.append(0)
                continue
            index = -1
            while eval_str[index] == "None":
                index -= 1
                if index == -len(eval_str):
                    break
            if index == -len(eval_str):
                last_eval =0
            else:
                last_eval = int(eval_str[index])
            if(last_eval > draw_margin ):
                prediction_list.append(1)
            elif(last_eval < -draw_margin):
                prediction_list.append(-1)
            else:
                prediction_list.append(0)
        eval_df[f"prediction_{draw_margin}"] = prediction_list
        eval_df[f"error_{draw_margin}"] = eval_df[f"prediction_{draw_margin}"] - eval_df["winner"] != 0
        accuracy = 1 - np.sum(eval_df[f"error_{draw_margin}"]) / len(eval_df)
        print(f"Accuracy with {draw_margin} draw margin: ", accuracy)
    # plot the acuracy function of draw margin
    plt.figure(figsize=(10, 7))
    plt.plot(range(MIN_MARGIN, MAX_MARGIN+1), [1 - np.sum(eval_df[f"error_{draw_margin}"]) / len(eval_df) for draw_margin in range(MIN_MARGIN, MAX_MARGIN+1)], marker="o")
    plt.xlabel("Draw Margin")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Draw Margin")
    plt.show()

    new_name = evaluation_csv.split(".")[0] + "_predictions.csv"
    eval_df.to_csv(new_name , index=False)
    print("Data updated in ", evaluation_csv)


## Data processing

def extract_trends(df_to_extract, move_ratio :float = 0.5, draws: bool = False, move_number : int = None):
    if not df_to_extract['evaluations'].any():
        return df_to_extract
    df_to_extract = df_to_extract[df_to_extract['winner'].notna()]
    if not draws:
        returned_df = df_to_extract[df_to_extract['winner'] != 0].reset_index(drop=True)
    else :
        returned_df = df_to_extract.copy()

    index = 0

    for evaluation in returned_df['evaluations']:

        try :
            evaluation = evaluation.split()
        except (AttributeError, TypeError):
            print("Splitting error at index ", index)
            returned_df.drop(index, inplace=True)
            returned_df = returned_df.reset_index(drop=True)
            continue
        if move_number is not None:
            if len(evaluation) < move_number:
                move_to_consider = len(evaluation)
            else :
                move_to_consider = move_number
        else :
            move_to_consider = int(len(evaluation) * move_ratio)
        if move_to_consider == 0:
            move_to_consider = 1
        evaluation = evaluation[:move_to_consider]
        returned_df.at[index, 'considered_evaluation'] = move_to_consider
        evaluation_list = []
        for e in evaluation:
            try:
                evaluation_list.append(int(e))
            except ValueError:
                continue
        returned_df.at[index, 'avg_eval'] = np.mean(evaluation_list)
        returned_df.at[index, 'max_eval'] = np.max(evaluation_list)
        returned_df.at[index, 'min_eval'] = np.min(evaluation_list)
        returned_df.at[index, 'eval_diff'] = evaluation_list[-1] - evaluation_list[0]
        returned_df.at[index, 'eval_variance'] = np.var(evaluation_list)
        # Rate of change
        rate_diff = np.diff(evaluation_list)
        if len(rate_diff) != 0:
            returned_df.at[index, "avg_rate_change"] = np.mean(rate_diff)
            returned_df.at[index, "max_rate_change"] = np.max(rate_diff)
        else :
            returned_df.at[index, "avg_rate_change"] = 0
            returned_df.at[index, "max_rate_change"] = 0
        # sign changes
        returned_df.at[index, "sign_changes"] = np.sum(np.diff(np.sign(evaluation_list)) != 0)
        # time in advantage
        returned_df.at[index, "white_advantage_time"] = np.sum(np.array(evaluation_list) > 0) / len(evaluation_list)
        # eval_range
        returned_df.at[index, "eval_range"] = np.max(evaluation_list) - np.min(evaluation_list)

        index+=1
    return returned_df

def get_eval_df(row_number: int = 10, stockfish = True, lc0 = False):
    move_df = pd.read_csv('Data/elite_chess_games_moves_100k_Games.csv').head(row_number)

    if lc0:
        engine = initialize_lc0()
    elif stockfish:
        engine = initialize_stockfish()
    else :
        print("No engine selected")
        return move_df
    if engine is None:
        print("Engine not initialized")
        return move_df
    one_percent = len(move_df) // 100
    if one_percent == 0:
        one_percent = 1
    # Loop through the DataFrame in chunks of 1%
    start_time = time.time()
    evaluation_list = []
    for index, data in move_df.iterrows():
        try :
            moves = data['moves']

            evaluations = get_evaluations(moves, engine,time=0.1)
            evaluation_list.append(evaluations)
            if (index + 1) % one_percent == 0:
                elapsed_time = time.time() - start_time
                print(f"Processed {index + 1} games in {elapsed_time:.2f} seconds")
        except KeyboardInterrupt:
            print("Process interrupted")
            print("Returning the processed data")
            move_df = move_df.iloc[:index]

            break
    move_df['evaluations'] = evaluation_list
    print(move_df['evaluations'])
    engine.quit()
    return move_df

def get_eval_csv(row_number: int = 10, stockfish = True, lc0 = False):
    df = get_eval_df(row_number, stockfish, lc0)
    if lc0:
        df.to_csv(f'Data/elite_chess_games_evaluations_{row_number}_rows_lc0.csv', index=False)
    else:
        df.to_csv(f'Data/elite_chess_games_evaluations_{row_number}_rows_stockfish.csv', index=False)
    print("Chess engine evaluation saved to CSV")
    return df

def get_trends_from_csv(csv_file:str, move_ratio: float = 0.5, move_number : int = None):
    evaluation_df = pd.read_csv(csv_file)
    trend_df = extract_trends(evaluation_df, move_ratio, draws=False, move_number= move_number)
    return trend_df

if __name__ == "__main__":
    trend_directory = "Data/move_trends"

    extract = True
    game_to_extract = 100
    is_ratio = False
    try :
        if str(sys.argv[1]).lower() in ["false"]:
            extract = False
    except IndexError:
        pass
    try :
        game_to_extract = int(sys.argv[2])
    except ValueError :
        print(f"Invalid argument, using default value of {game_to_extract}")
    except IndexError:
        pass
    try :
        if str(sys.argv[3]).lower() in ["true", "ratio"]:
            is_ratio = True
    except IndexError:
        pass


    # Extract evaluations

    if extract :
        get_eval_csv(row_number=game_to_extract, stockfish=True, lc0=False)
        print(" ================================================ ")

    """## TEST MODEL
    simple_model(evaluation_csv=f"Data/elite_chess_games_evaluations_{game_to_extract}_rows_stockfish.csv")

    # Extract trends
    move_low = 10
    move_high = 80
    step = 5
    if game_to_extract > 10:
        game_str = str(game_to_extract)
    else :
        game_str = "0" + str(game_to_extract)
    evaluation_file = f"Data/elite_chess_games_evaluations_{game_str}_rows_stockfish.csv"
    print(f"Extracting trends from {evaluation_file}")
    for i in range(move_low, move_high + 1, step):
        move_ratio = i / 100

        move_number = None
        if not is_ratio:
            move_number = i
            print(f"Extracting trends for {move_number} moves")
        else :
            print(f"Extracting trends for {move_ratio * 100}% of the game")
        trend_df = get_trends_from_csv(evaluation_file,move_ratio, move_number=move_number)

        move_number_str = str(int(move_ratio * 100))
        if not is_ratio:
            move_number_str = "0" + str(move_number) if  move_number<10 else str(move_number)
        trend_df.to_csv(f"{trend_directory}/trend_data_{move_number_str}_moves.csv", index=False)
        print(f"Data saved to {trend_directory}/trend_data_{move_number_str}_moves.csv")
        print(f" ================================================ ")

    # Test models
    trend_files = sorted(os.listdir(trend_directory))
    random_forest_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    support_vector_machine_model = svm.SVC(kernel="linear",  random_state=42)

    result_tab = []
    for file in trend_files:
        if ".csv" in file:
            df = pd.read_csv(f"{trend_directory}/{file}")
            X = df[
                ["avg_eval", "max_eval", "min_eval", "eval_diff", "eval_variance", "avg_rate_change", "max_rate_change",
                 "sign_changes", "white_advantage_time", "eval_range"]
            ]
            y = df["winner"]

            print(f"Processing {file} with Random Forest Model")
            print("...")
            result = test_model(X, y, model=random_forest_model)
            result = ["Random Forest", file.split('_')[-2], result["accuracy"], result["macro avg"]["precision"],
                      result["macro avg"]["recall"], result["macro avg"]["f1-score"],
                      result["weighted avg"]["precision"], result["weighted avg"]["recall"],
                      result["weighted avg"]["f1-score"]]

            result_tab.append(result)
            print(" ================================================ ")

            print(f"Processing {file} with Gradient Boosting Model")
            print("...")
            result = test_model(X, y, model=gradient_boosting_model)
            result = ["Gradient Boosting", file.split('_')[-2], result["accuracy"], result["macro avg"]["precision"], result["macro avg"]["recall"],
                      result["macro avg"]["f1-score"], result["weighted avg"]["precision"],
                      result["weighted avg"]["recall"], result["weighted avg"]["f1-score"]]
            result_tab.append(result)
            print(" ================================================ ")

            print(f"Processing {file} with Support Vector Machine Model")
            print("...")
            result = test_model(X, y, model=support_vector_machine_model)
            result = ["Support Vector Machine", file.split('_')[-2], result["accuracy"], result["macro avg"]["precision"], result["macro avg"]["recall"],
                      result["macro avg"]["f1-score"], result["weighted avg"]["precision"],
                      result["weighted avg"]["recall"], result["weighted avg"]["f1-score"]]
            result_tab.append(result)
            print(" ================================================ ")

            print("\n\n\n")


    result_df = pd.DataFrame(result_tab, columns=["Model","Move_number","Accuracy", "Precision", "Recall", "F1-Score", "Weighted Precision", "Weighted Recall", "Weighted F1-Score"])

    result_df.to_csv("Data/Model_Performances.csv", index=False)"""

    support_vector_machine_model = svm.SVC(kernel="linear",  random_state=42)

    df80 = pd.read_csv(f"{trend_directory}/trend_data_80_moves.csv")
    df20 = pd.read_csv(f"{trend_directory}/trend_data_20_moves.csv")

    X_train = df80[
        ["avg_eval", "max_eval", "min_eval", "eval_diff", "eval_variance", "avg_rate_change", "max_rate_change",
         "sign_changes", "white_advantage_time", "eval_range"]
    ][:int(len(df80)*0.70)]
    X_test = df20[
        ["avg_eval", "max_eval", "min_eval", "eval_diff", "eval_variance", "avg_rate_change", "max_rate_change",
         "sign_changes", "white_advantage_time", "eval_range"]
    ][:-int(len(df80)*0.70)]

    y_train = df80["winner"][:int(len(df80)*0.70)]
    y_test = df20["winner"][:-int(len(df80)*0.70)]
    print("inchallah ça marche")
    model_specific_test(X_train,X_test, y_train,y_test, model=support_vector_machine_model)
    print("All done!")

