import os
import json
import pandas as pd
from tqdm import tqdm
from py7zr import SevenZipFile

DATA_DIR = "nba_tracking_7z"
TEMP_DIR = "temp_json"

def extract_and_load_json(archive_path):
    """Extract JSON files from a .7z archive and load them safely"""
    with SevenZipFile(archive_path, mode='r') as archive:
        archive.extractall(path=TEMP_DIR)
    
    files = [f for f in os.listdir(TEMP_DIR) if f.endswith(".json")]
    if not files:
        print(f"Warning: No JSON file in {archive_path}")
        return None
    
    json_path = os.path.join(TEMP_DIR, files[0])
    
    # Skip empty JSON files
    if os.path.getsize(json_path) == 0:
        print(f"Warning: {json_path} is empty. Skipping.")
        os.remove(json_path)
        return None
    
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: {json_path} is corrupt. Skipping.")
        os.remove(json_path)
        return None
    
    os.remove(json_path)  # free disk space
    return data

def summarize_game(game_data):
    """Dummy summary function - replace with your own logic"""
    return {
        "gameid": game_data.get("gameId"),
        "home_team": game_data.get("homeTeam"),
        "visitor_team": game_data.get("visitorTeam"),
        "home_score": game_data.get("homeScore"),
        "visitor_score": game_data.get("visitorScore")
    }

# --- Main loop ---
archives = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".7z")])
df = pd.DataFrame(columns=["gameid", "home_team", "visitor_team", "home_score", "visitor_score"])


MAX_GAMES = 10  # change to 10 if you want
processed_games = 0

archives = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".7z")])
df = pd.DataFrame(columns=["gameid", "home_team", "visitor_team", "home_score", "visitor_score"])

for file in tqdm(archives, desc="Processing games"):
    if processed_games >= MAX_GAMES:
        break
    
    path = os.path.join(DATA_DIR, file)
    data = extract_and_load_json(path)
    
    if data:  # only summarize if JSON is valid
        summary = summarize_game(data)
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        processed_games += 1

print(f"Processed {processed_games} valid games.")