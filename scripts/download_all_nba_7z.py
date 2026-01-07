"""
Download all 2015-2016 NBA SportVU tracking data (.7z files)
Source: https://github.com/linouk23/NBA-Player-Movements
"""

import os
import re
import json
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# GitHub API URL for 2015-16 game data folder
API_URL = "https://api.github.com/repos/linouk23/NBA-Player-Movements/contents/data/2016.NBA.Raw.SportVU.Game.Logs"

# Output directory
OUTPUT_DIR = "nba_tracking_7z"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📡 Fetching list of .7z files from GitHub...")
res = requests.get(API_URL)
if res.status_code != 200:
    raise RuntimeError(f"GitHub API request failed: {res.status_code}")

data = res.json()
files = [f for f in data if f["name"].endswith(".7z")]

print(f"✅ Found {len(files)} .7z files to download.\n")

def download_file(file_info):
    url = file_info["download_url"]
    name = file_info["name"]
    path = os.path.join(OUTPUT_DIR, name)

    if os.path.exists(path):
        return f"⏩ Skipped {name} (already downloaded)"
    
    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 200:
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return f"✅ Downloaded {name}"
        else:
            return f"⚠️ Failed {name} ({r.status_code})"
    except Exception as e:
        return f"❌ Error {name}: {e}"

# Download with thread pool for speed
with ThreadPoolExecutor(max_workers=8) as executor:
    for result in tqdm(executor.map(download_file, files), total=len(files)):
        print(result)

print("\n🏁 All downloads complete!")
print(f"Files saved in: {os.path.abspath(OUTPUT_DIR)}")
