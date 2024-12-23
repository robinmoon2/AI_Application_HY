import chess
import chess.engine
import chess.pgn
import sys

def play_games(number_of_games, time_limit, engine1_path, engine2_path, output_pgn_file, live = False):
    if sys.platform.startswith("darwin"):
        print("MacOS is not supported")
        return
    

    time_limit = chess.engine.Limit(time=time_limit)

    # Initialize engines

    if sys.platform.startswith("win32"):
        engine1_name = engine1_path.split("/")[0]
        engine2_name = engine2_path.split("/")[0]

        engine1 = chess.engine.SimpleEngine.popen_uci(engine1_path)
        engine2 = chess.engine.SimpleEngine.popen_uci(engine2_path)
    else:
        print("Only Stockfish is avaible on your system")
        engine1_name = "Stockfish"
        engine2_name = "Stockfish"

        engine1 = chess.engine.SimpleEngine.popen_uci('stockfish/stockfish-ubuntu-x86-64-avx2')
        engine2 = chess.engine.SimpleEngine.popen_uci('stockfish/stockfish-ubuntu-x86-64-avx2')

    # Create a PGN game collection
    pgn_collection = []

    for game_number in range(1, number_of_games + 1):
        # Alternate colors
        white_engine = engine1 if game_number % 2 != 0 else engine2
        black_engine = engine2 if game_number % 2 != 0 else engine1

        # Create a new chess game
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = f"{engine1_name} vs {engine2_name} - Game {game_number}"
        game.headers["White"] = engine1_name if white_engine == engine1 else engine2_name
        game.headers["Black"] = engine2_name if black_engine == engine2 else engine1_name
        # Play the game
        node = game
        if live:
            live_pgn_file = f"Data/live_game.pgn"

            live_pgn =  open(live_pgn_file, "w")
        while not board.is_game_over():
            # Choose engine based on the turn
            engine = white_engine if board.turn == chess.WHITE else black_engine

            # Get the best move
            result = engine.play(board, time_limit)
            move = result.move

            # Apply the move to the board
            board.push(move)

            # Add the move to the PGN node
            node = node.add_variation(move)
            # Write the current game state to the live PGN file
            if live:
                print(game, file=live_pgn)
                live_pgn.flush()


        # Save the game result
        game.headers["Result"] = board.result()
        print(f"Game {game_number} finished: {board.result()}")
        if board.result() == "1-0":
            print(f"{game.headers['White']} won")
        elif board.result() == "0-1":
            print(f"{game.headers['Black']} won")
        else:
            print("Draw")

        # Add game to collection
        pgn_collection.append(game)

    # Save all games to a single PGN file
    with open(output_pgn_file, "w") as pgn_file:
        for game in pgn_collection:
            print(game, file=pgn_file, end="\n\n")
    print(f"All {number_of_games} games saved to {output_pgn_file}")

    # Close engines
    engine1.quit()
    engine2.quit()

if __name__ == "__main__":
    args = sys.argv


    lc0_path = "LC0/lc0.exe"
    stockfish_path = "stockfish/stockfish-windows-x86-64-avx2.exe"
    if sys.platform.startswith("linux"):
        stockfish_path += "stockfish/stockfish-ubuntu-x86-64-avx2"
    output_pgn_file = "Data/ModelVsModelGames.pgn"

    number_of_games = 2
    time_limit = 0.2
    engine1_path = stockfish_path
    engine2_path = lc0_path

    live = False
    if len(args) > 1:
        try :
            number_of_games = int(args[1])
            time_limit = float(args[2])
            if number_of_games < 1 or time_limit < 0.1:
                raise Exception
        except:
            print("Invalid arguments. Number of games : 5, Time limit : 1.0")
            number_of_games = 5
            time_limit = 1.0
        if len(args) > 3:
            if args[3].lower() in ["stockfish", "sf"]:
                engine1_path = stockfish_path
            elif args[3].lower() in ["lc0", "lczero"]:
                engine1_path = lc0_path
            else :
                print("Invalid argument for engine 1. Using stockfish as default")
                engine1_path = stockfish_path
            if args[4].lower() in ["stockfish", "sf"]:
                engine2_path = stockfish_path
            elif args[4].lower() in ["lc0", "lczero"]:
                engine2_path = lc0_path
            else :
                print("Invalid argument for engine 2. Using LC0 as default")
                engine2_path = lc0_path
            if len(args) > 5:
                live = str(args[5]).lower() in ["live", "true"]
                if live:
                    print("Live mode enabled")
    play_games(number_of_games, time_limit, engine1_path, engine2_path, output_pgn_file,live)

