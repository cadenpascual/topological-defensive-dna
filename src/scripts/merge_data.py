import json
import pandas as pd
from pathlib import Path

# ======================
# CONFIG
# ======================
GAME_JSON = Path("data/json/0021500622.json")   # already-extracted SportVU JSON
PBP_CSV = Path("data/2015-16_pbp.csv")
OUTPUT_JSON = Path("data/frames/0021500622.frames.json")

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# ======================
# LOAD DATA
# ======================
with open(GAME_JSON, "r", encoding="utf-8") as f:
    game = json.load(f)

pbp = pd.read_csv(PBP_CSV)

game_id = int(game["gameid"])

# ======================
# HELPERS
# ======================
def identify_possession(row):
    """
    Simplified NBA possession logic
    """
    msg = int(row["EVENTMSGTYPE"])

    if msg in {1, 2, 3, 4, 5}:  # shots, rebounds, turnovers
        return row["PLAYER1_TEAM_ID"]

    if "OFF.FOUL" in str(row["HOMEDESCRIPTION"]) or "OFF.FOUL" in str(row["VISITORDESCRIPTION"]):
        return row["PLAYER1_TEAM_ID"]

    if msg == 6:  # foul
        return row["PLAYER2_TEAM_ID"]

    return None

def safe_int(x):
    return int(x) if pd.notna(x) else None

def safe_float(x):
    return float(x) if x is not None else None

# ======================
# BUILD FRAME DATA
# ======================
frame_counter = 0
events_out = []

for event in game["events"]:
    # Skip events with no moments
    if not event.get("moments"):
        continue

    event_id = int(event["eventId"])

    pbp_row = pbp.loc[
        (pbp.GAME_ID == game_id) &
        (pbp.EVENTNUM == event_id)
    ]

    if len(pbp_row) != 1:
        continue

    row = pbp_row.iloc[0]
    possession_team_id = safe_int(identify_possession(row))

    # Quarter is constant within an event → take first moment
    quarter = int(event["moments"][0][0])

    event_obj = {
        "gameid": game_id,
        "event_id": event_id,
        "possession_team_id": possession_team_id,
        "quarter": quarter,
        "frames": []
    }

    for moment in event["moments"]:
        # Skip invalid moments
        if moment is None or len(moment) < 6:
            continue

        frame_counter += 1

        frame = {
            "frame_id": frame_counter,
            "game_clock": safe_float(moment[2]),
            "shot_clock": safe_float(moment[3]),
            "ball": {
                "x": safe_float(moment[5][0][2]),
                "y": safe_float(moment[5][0][3]),
                "z": safe_float(moment[5][0][4]),
            },
            "players": [
                {
                    "teamid": safe_int(p[0]),
                    "playerid": safe_int(p[1]),
                    "x": safe_float(p[2]),
                    "y": safe_float(p[3]),
                    "z": safe_float(p[4]),
                }
                for p in moment[5][1:]
            ]
        }

        event_obj["frames"].append(frame)

    # Only keep events that actually produced frames
    if event_obj["frames"]:
        events_out.append(event_obj)

# ======================
# SAVE TO JSON
# ======================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(events_out, f, indent=2)

print(f"✅ Saved {len(events_out)} events with {frame_counter} frames to {OUTPUT_JSON}")
