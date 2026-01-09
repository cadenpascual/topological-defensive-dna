import numpy as np
import xarray as xr     
import ot   
from ot.gromov import gromov_wasserstein, entropic_gromov_wasserstein, fused_gromov_wasserstein

#####  Data Processing ######
def assign_event_possession(event):
    """
    Assign possession_team_id for an event based on the first frame.
    """
    if not event.get("frames"):
        event["possession_team_id"] = None
        return

    first_frame = event["frames"][0]
    ball = first_frame.get("ball")
    players = first_frame.get("players", [])

    if ball is None or not players:
        event["possession_team_id"] = None
        return

    ball_pos = np.array([ball["x"], ball["y"]])

    # Find the player closest to the ball
    min_dist = float("inf")
    possession_team = None
    for p in players:
        player_pos = np.array([p["x"], p["y"]])
        dist = np.linalg.norm(ball_pos - player_pos)
        if dist < min_dist:
            min_dist = dist
            possession_team = p["teamid"]

    event["possession_team_id"] = possession_team

## Tensor Conversion Functions ##

# JSON -> Tensor (3D array: Moment, Players, (x,y) Coords)
# Tensor with all players and ball
def event_to_tensor(event, include_ball=False, max_frames=None):
    """
    Convert an event to a NumPy tensor of shape (T, N, 2),
    where:
        T = number of frames (or max_frames)
        N = number of players (+1 if include_ball)
        2 = x, y coordinates
    """
    frames = event.get("frames", [])

    if not frames:
        # No frames in this event
        return np.empty((0, 0, 0))

    if max_frames is not None:
        frames = frames[:max_frames]

    # Determine number of points (players + optional ball)
    num_players = len(frames[0].get("players", []))
    if num_players == 0:
        return np.empty((0, 0, 0))

    num_points = num_players + 1 if include_ball else num_players
    T = len(frames)

    traj = np.zeros((T, num_points, 2))

    for t, frame in enumerate(frames):
        # players
        for i, p in enumerate(frame.get("players", [])):
            traj[t, i, 0] = p.get("x", 0.0)
            traj[t, i, 1] = p.get("y", 0.0)

        # ball
        if include_ball:
            ball = frame.get("ball")
            if ball:
                traj[t, -1, 0] = ball.get("x", 0.0)
                traj[t, -1, 1] = ball.get("y", 0.0)

    return traj

# Tensor with only offensive players and ball
def event_to_tensor_offense(event, include_ball=False, max_frames=None):
    frames = event.get("frames", [])
    if max_frames is not None:
        frames = frames[:max_frames]

    # Determine offensive players
    possession_team_id = event.get("possession_team_id")
    offensive_players = [
        p for p in frames[0]["players"] if p["teamid"] == possession_team_id
    ]

    T = len(frames)
    N = len(offensive_players) + (1 if include_ball else 0)
    traj = np.zeros((T, N, 2))

    for t, frame in enumerate(frames):
        # offense
        for i, p in enumerate(frame["players"]):
            if p["teamid"] == possession_team_id:
                matching_indices = [j for j, op in enumerate(offensive_players) if op["playerid"] == p["playerid"]]
                if matching_indices:  # only update if player found
                    idx = matching_indices[0]
                    traj[t, idx, 0] = p["x"]
                    traj[t, idx, 1] = p["y"]

        # ball
        if include_ball:
            ball = frame.get("ball")
            if ball:
                traj[t, -1, 0] = ball["x"]
                traj[t, -1, 1] = ball["y"]

    return traj

def split_offense_defense(event, traj):
    """
    Split players in a tensor into offensive and defensive sets
    based on possession_team_id.
    
    traj: T x N x 2 tensor from event_to_tensor(include_ball=False)
    Returns:
        offense: T x n_offense x 2
        defense: T x n_defense x 2
    """
    frames = event["frames"]
    possession_team = event.get("possession_team_id")
    if possession_team is None or traj.shape[1] == 0:
        return np.empty((0, 0, 2)), np.empty((0, 0, 2))
    
    # Indices for offensive vs defensive players
    offense_idx = [i for i, p in enumerate(frames[0]["players"]) if p["teamid"] == possession_team]
    defense_idx = [i for i, p in enumerate(frames[0]["players"]) if p["teamid"] != possession_team]
    
    # Slice the tensor
    offense = traj[:, offense_idx, :]
    defense = traj[:, defense_idx, :]
    
    return offense, defense


# Dataset with offensive players + ball
def build_offensive_dataset(game, max_frames=150):
    plays = []

    for event in game:
        if not event.get("frames"):
            continue

        tensor = event_to_tensor_offense(event, max_frames=max_frames)
        if tensor.size > 0:
            plays.append(tensor)

    if not plays:
        return None

    P = len(plays)  # number of events
    T = max(p.shape[0] for p in plays)  # max time steps
    N = plays[0].shape[1]  # number of offensive players

    # Initialize array
    data = np.zeros((P, T, N, 2))

    # Fill array
    for i, p in enumerate(plays):
        data[i, :p.shape[0]] = p

    # Build xarray dataset
    dataset = xr.Dataset(
        data_vars={"positions": (["play", "time", "player", "coord"], data)},
        coords={
            "play": np.arange(P),
            "time": np.arange(T),
            "player": np.arange(N),
            "coord": ["x", "y"]
        }
    )

    return dataset



#####  Visualizations ######
import matplotlib.pyplot as plt

def plot_frame(frame, team_colors=None):
    """
    Plot a single NBA frame with players split by team and the ball.
    
    Parameters
    ----------
    frame : dict
        Single frame dictionary with keys: 'ball', 'players', 'frame_id', etc.
    team_colors : dict, optional
        Mapping from teamid to color. Example: {1610612739: 'blue', 1610612744: 'red'}
    """
    
    if team_colors is None:
        team_colors = {}
    
    plt.figure(figsize=(15, 7))
    
    # Draw court boundaries (simplified rectangle)
    plt.plot([0, 50], [0, 0], color='black')   # baseline
    plt.plot([0, 50], [94, 94], color='black') # opposite baseline
    plt.plot([0, 0], [0, 94], color='black')   # sideline
    plt.plot([50, 50], [0, 94], color='black') # opposite sideline
    
    # Draw the ball
    ball = frame["ball"]
    plt.scatter(ball["x"], ball["y"], c='orange', s=200, marker='o', label='Ball', edgecolors='black')
    
    # Plot players
    for player in frame["players"]:
        x, y = player["x"], player["y"]
        teamid = player["teamid"]
        color = team_colors.get(teamid, 'green')  # default green if teamid not in dict
        plt.scatter(x, y, c=color, s=150, label=f'Team {teamid}' if f'Team {teamid}' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(x+0.5, y+0.5, str(player["playerid"]), fontsize=9, color=color)
    
    plt.xlim(0, 50)
    plt.ylim(0, 94)
    plt.xlabel("Court X (ft)")
    plt.ylabel("Court Y (ft)")

    # convert game clock to MM:SS format
    minutes = int(frame['game_clock'] // 60)
    seconds = int(frame['game_clock'] % 60)

    # get shot clock if available
    shot_clock = (frame['shot_clock'])
    shot_str = f"{shot_clock:.1f}s" if shot_clock is not None else "-"

    plt.title(f"Frame {frame['frame_id']} - Game Clock: {minutes:02d}:{seconds:02d} | Shot Clock: {shot_str}")
    plt.legend()
    plt.show()



# Distance Profile Class
class DistanceProfile:
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def get_pairwise_dist(self, cloud, ord):
            # cloud shape: (N, D)
            # Using broadcasting: (N, 1, D) - (1, N, D) -> (N, N, D)
            diff = cloud[:, np.newaxis, :] - cloud[np.newaxis, :, :]
            # Calculate norm along the last axis (the coordinates)
            return np.linalg.norm(diff, ord=ord, axis=-1)

    def compute_LN_matrix(self,source,target,ord):
        """
        Compute the L-norm distance matrix
        """
        dist_source = self.get_pairwise_dist(source, ord)
        dist_target = self.get_pairwise_dist(target, ord)

        return dist_source, dist_target

    def compute_W_matrix(self, X, Y):
        """
        Computes W(i,j) for all i ∈ [n], j ∈ [m] as defined in the equation.

        X: array of shape (n, d)
        Y: array of shape (m, d)

        Returns: W matrix of shape (n, m)
        """
        n, _ = X.shape
        m, _ = Y.shape

        # Precompute all intra-set distances ||X_i - X_ℓ|| and ||Y_j - Y_ℓ||
        X_dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)  # shape (n, n)
        Y_dists = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)  # shape (m, m)

        # Calculate D(i, j) which is the Gromov-Wasserstein distance between the distributions of distances
        # Gromov-Wasserstein between the two empirical distributions
        C1 = ot.dist(X_dists)
        C2 = ot.dist(Y_dists)
        M = ot.dist(X, Y)
        p = ot.unif(n)
        q = ot.unif(m)
        Gwg, logw = fused_gromov_wasserstein(M, C1, C2, p, q, loss_fun="square_loss", alpha=1e-3, verbose=True, log=True)

        return Gwg