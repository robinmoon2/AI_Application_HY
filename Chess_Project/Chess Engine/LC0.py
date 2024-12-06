import chess
import chess.engine

# Path to the LCZero executable (downloaded from the LCZero website)
lc0_path = "LC0/lc0.exe"

# Initialize the chess engine
with chess.engine.SimpleEngine.popen_uci(lc0_path) as engine:
    # Define a chess position in FEN notation
    fen_position = "r1bqkbnr/pppppppp/n7/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Example starting position
    board = chess.Board(fen_position)

    # Get the best move
    result = engine.play(board, chess.engine.Limit(time=2.0))  # 2 seconds per move
    print("Best move:", result.move)

    info = engine.analyse(board, chess.engine.Limit(time=2.0))
    print(info)
