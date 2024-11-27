import re
import os
import chess.pgn
import pandas as pd
import time

NUMBER_OF_GAMES = 10000000

# Function to determine the winner
def get_winner(result):
    if result == "1-0":
        return 1  # White wins
    elif result == "0-1":
        return -1  # Black wins
    elif result == "1/2-1/2":
        return 0  # Draw
    else:
        return None  # Unknown result

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

start_time = time.time()

# Process the PGN file
for pgn_file_path in pgn_files:
    file_start_time = time.time()
    with open(pgn_file_path) as pgn_file:
        while True and row < NUMBER_OF_GAMES:
            game = chess.pgn.read_game(pgn_file)
            if game is None:  # End of file
                break
            row += 1
            # Extract headers
            headers = game.headers
            black_id = headers.get('Black', None)
            white_id = headers.get('White', None)
            result = headers.get('Result', '')
            white_rating = headers.get('WhiteElo', None)
            black_rating = headers.get('BlackElo', None)
            opening_eco = headers.get('ECO', None)
            opening_name = headers.get('Opening', None)
            time_control = headers.get('TimeControl', None)
            turns = sum(1 for _ in game.mainline_moves())  # Number of moves
            lichess_url = headers.get('LichessURL', '')
            game_id = extract_game_id(lichess_url)

            victory_status = headers.get('Termination', '')

            # Add to data list
            data.append({
                'id': game_id,
                'turns': turns,
                'white_id': white_id,
                'black_id': black_id,
                'white_rating': int(white_rating) if white_rating else None,
                'black_rating': int(black_rating) if black_rating else None,
                'time_control': time_control,
                'opening_eco': opening_eco,
                'opening_name': opening_name,
                'victory_status': victory_status.lower() if victory_status else None,
                'winner': get_winner(result),
            })

            if row % (NUMBER_OF_GAMES//100) == 0:
                elapsed_time = time.time() - start_time
                print(f"Processed {row} games in {elapsed_time:.2f} seconds")

    file_elapsed_time = time.time() - file_start_time
    print(f"Processed file {pgn_file_path} in {file_elapsed_time:.2f} seconds")
# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_csv_path = '../CSV_Output/elite_chess_games_features.csv'
df.to_csv(output_csv_path, index=False)

print(f'Data saved to {output_csv_path}')