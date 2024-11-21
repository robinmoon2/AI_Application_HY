import re
import os
import chess.pgn
import pandas as pd
import time

NUMBER_OF_GAMES = 10000
MOVE_NO = 10

# Function to extract game ID from Lichess URL
def extract_game_id(lichess_url):
    match = re.search(r'https://lichess.org/([a-zA-Z0-9]+)', lichess_url)
    if match:
        return match.group(1)
    return None

# Initialize an empty list to store game data
data = []

# Path to the PGN files
pgn_directory = 'Lichess Elite Database/Lichess Elite Database/'

row = 0

# Get a list of PGN files sorted by dates
pgn_files = sorted(
    [os.path.join(pgn_directory, f) for f in os.listdir(pgn_directory) if f.endswith('.pgn')],
    key=lambda x: os.path.basename(x), reverse= True
)

# Process the PGN file
for pgn_file_path in pgn_files:
    with open(pgn_file_path) as pgn_file:
        while True and row < NUMBER_OF_GAMES:
            game = chess.pgn.read_game(pgn_file)
            if game is None:  # End of file
                break
            row += 1
            # Extract headers
            headers = game.headers
            game_id = extract_game_id(headers.get('Site', ''))
            moves = ' '.join(str(move) for move in game.mainline_moves())

            # Iterate
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                if board.fullmove_number == MOVE_NO:
                    break

            # Add to data list
            data.append({
                'id': game_id,
                'moves': moves,
                'fen': board.fen()
            })

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_csv_path = 'elite_chess_games_moves.csv'
df.to_csv(output_csv_path, index=False)

print(f'Data saved to {output_csv_path}')