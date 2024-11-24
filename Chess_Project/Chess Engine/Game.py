import chess
import chess.engine
import chess.pgn
import sys


def play_games(number_of_games, time_limit, engine1, engine2, output_pgn_file, live = False):
    time_limit = chess.engine.Limit(time=time_limit)  # 0.5 seconds per move

    lc0_path = "LC0/lc0.exe"
    stockfish_path = "stockfish/stockfish-windows-x86-64-avx2.exe"

    # Initialize engines
    lc0_engine = chess.engine.SimpleEngine.popen_uci(engine1)
    stockfish_engine = chess.engine.SimpleEngine.popen_uci(engine2)

    # Create a PGN game collection
    pgn_collection = []

    for game_number in range(1, number_of_games + 1):
        # Alternate colors
        white_engine = lc0_engine if game_number % 2 != 0 else stockfish_engine
        black_engine = stockfish_engine if game_number % 2 != 0 else lc0_engine

        # Create a new chess game
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = f"Lc0 vs Stockfish - Game {game_number}"
        game.headers["White"] = "Lc0" if white_engine == lc0_engine else "Stockfish"
        game.headers["Black"] = "Stockfish" if black_engine == stockfish_engine else "Lc0"
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

        # Add game to collection
        pgn_collection.append(game)

    # Save all games to a single PGN file
    with open(output_pgn_file, "w") as pgn_file:
        for game in pgn_collection:
            print(game, file=pgn_file, end="\n\n")
    print(f"All {number_of_games} games saved to {output_pgn_file}")

    # Close engines
    lc0_engine.quit()
    stockfish_engine.quit()

if __name__ == "__main__":
    args = sys.argv

    lc0_path = "LC0/lc0.exe"
    stockfish_path = "stockfish/stockfish-windows-x86-64-avx2.exe"
    output_pgn_file = "Data/lc0_vs_stockfish.pgn"

    number_of_games = 5
    time_limit = 1.0
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

