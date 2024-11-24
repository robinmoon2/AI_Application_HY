import chess
import chess.engine

# Path to the Stockfish executable
stockfish_path = "stockfish/stockfish-windows-x86-64-avx2.exe"  # Update with your path

# Create a chess board (starting position or custom FEN string)
board = chess.Board()  # Standard starting position
# Or for a custom position, use FEN
# board = chess.Board("rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1")

# Initialize Stockfish engine
with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
    # Set a time limit for the engine to evaluate the best move
    time_limit = 1.0  # 1 second
    result = engine.play(board, chess.engine.Limit(time=time_limit))

    # Get the best move suggested by Stockfish
    print("Best move:", result.move)

    # Optionally, print the evaluation score
    info = engine.analyse(board, chess.engine.Limit(time=time_limit))
    print(info)
    print("Evaluation score:", info["score"])
