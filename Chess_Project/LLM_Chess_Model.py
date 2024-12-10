import chess
import chess.engine
import chess.pgn
import sys
import google.api_core.exceptions
import google.generativeai as genai
import dotenv
import os
import time

class GeminiChessModel:
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = self.initialize_gemini(model)
        self.chat = self.model.start_chat(
            history=[{"role": "user", "parts": "You are a chess engine.I give you an incomplete PGN file of a chess game."
                                                " For each PGN given, you will provide the next move in the format of UCI move with the starting square and ending square."
                                                " DO NOT provide move in the format of SAN. ALWAYS provide legal moves."
                                                " e.g. A : e2e4, e7e5, g1f3, g8f6 etc."}]
        )

    @staticmethod
    def initialize_gemini(gemini_model: str = "gemini-1.5-flash"):
        dotenv.load_dotenv()
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel(gemini_model)
        return gemini_model

    def get_move(self, game: chess.pgn.Game, legal_moves: list):
        prompt = (f"Here is an incomplete PGN file of a chess game: \n```PGN\n{game}\n```\n"
                  f"Please provide the next move in the format of UCI move with the starting square and ending square.\n"
                  f"DO NOT provide move in the format of SAN. ALWAYS provide legal moves.\n"
                  f"Here are the legal moves: \n{legal_moves}\nONLY CHOOSE an element in the list in order to win the game.")
        print(prompt)

        max_retries = 6
        for attempt in range(max_retries):
            try:
                response = self.chat.send_message(prompt)
                is_legal = False
                response_str = response.text.strip()
                if response.text in legal_moves:
                    is_legal = True
                while not is_legal:
                    response = self.chat.send_message("You have to provide a move in the list \n"+ prompt)
                    print(response.text)
                    response_str = response.text.strip()
                    if response_str in legal_moves:
                        is_legal = True
                return chess.Move.from_uci(response_str)

            except google.api_core.exceptions.ResourceExhausted as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Quota exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Quota exceeded.")
                    raise e

def initialize_stockfish(elo_rating : int=None):
    if sys.platform.startswith("linux"):
        stockfish_path = "Chess Engine/stockfish/stockfish-ubuntu-x86-64-avx2"
    elif sys.platform.startswith("win"):
        stockfish_path = "Chess Engine/stockfish/stockfish-windows-x86-64-avx2.exe"
    else:
        print("Unsupported OS")
        return None
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    if elo_rating:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo_rating})
    return engine

def play_game(stockfish_elo : int =  None, gemini_white : bool = True):
    engine = initialize_stockfish(stockfish_elo)
    gemini_model = GeminiChessModel(model="gemini-1.5-flash")
    if engine is None:
        return

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Stockfish vs Gemini"
    game.headers["White"] = "Gemini" if gemini_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if gemini_white else "Gemini"
    game.headers["WhiteELO"] = str(stockfish_elo) if not gemini_white and not stockfish_elo else "2900"
    game.headers["BlackELO"] = str(stockfish_elo) if gemini_white and stockfish_elo else "2900"

    node = game

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            gemini_turn = game.headers["White"] == "Gemini"
        else :
            gemini_turn = game.headers["Black"] == "Gemini"

        if  gemini_turn:
            legal_moves = get_legal_moves(board)
            move = gemini_model.get_move(game, legal_moves)
            print("Gemini move : ", move)
        else:
            result = engine.play(board, chess.engine.Limit(time=2.0))
            move = result.move
            print("engine move : ", move)
        board.push(move)
        node = node.add_variation(move)

    game.headers["Result"] = board.result()
    if board.result() == "1-0":
        print(f"{game.headers['White']} won")
    elif board.result() == "0-1":
        print(f"{game.headers['Black']} won")
    else:
        print("Draw")

    engine.quit()

    return game


def play_games(number_of_games : int = 10, stockfish_elo : int = None):
    if sys.platform.startswith("darwin"):
        print("MacOS is not supported")
        return

    pgn_collection = []

    for game_number in range(1, number_of_games + 1):
        game = play_game(stockfish_elo, game_number % 2 == 0)
        pgn_collection.append(game)

    output_pgn_file = "Data/ModelVsModelGames.pgn"
    with open(output_pgn_file, "w") as pgn_file:
        for game in pgn_collection:
            print(game, file=pgn_file, end="\n\n")
    print(f"All {number_of_games} games saved to {output_pgn_file}")

def get_legal_moves(board : chess.Board):
    return [move.uci() for move in board.legal_moves]

if __name__ == "__main__":
    args = sys.argv

    number_of_games = 2
    stockfish_elo = 1500
    if len(args) > 1:
        try:
            number_of_games = int(args[1])
            if number_of_games < 1:
                raise ValueError
        except ValueError:
            print("Invalid arguments. Number of games : 10")
            number_of_games = 10
        if len(args) > 2:
            try:
                stockfish_elo = int(args[2])
            except (ValueError):
                print("Invalid argument for Stockfish ELO rating. Using 2000 as default")
                stockfish_elo = 2000


    play_games(number_of_games, stockfish_elo)
