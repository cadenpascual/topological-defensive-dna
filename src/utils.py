import numpy as np
import xarray as xr     
import ot   
from ot.gromov import gromov_wasserstein, entropic_gromov_wasserstein, emd, fused_gromov_wasserstein

# JSON -> Tensor (3D array: Moment, Players, (x,y) Coords)
def event_to_tensor(event, max_frames=None):
    moments = event["moments"]
    if max_frames:
        moments = moments[:max_frames]

    T = len(moments)
    traj = np.zeros((T, 10, 2))

    for t, m in enumerate(moments):
        players = m[5][1:]  # skip ball
        for i, p in enumerate(players):
            traj[t, i, 0] = p[2]  # x
            traj[t, i, 1] = p[3]  # y

    return traj


def build_dataset(game, max_frames=150):
    plays = []
    for event in game["events"]:
        if "moments" not in event or len(event["moments"]) < 10:
            continue
        plays.append(event_to_tensor(event, max_frames))

    P = len(plays)
    T = max(p.shape[0] for p in plays)

    data = np.zeros((P, T, 10, 2))

    for i, p in enumerate(plays):
        data[i, :p.shape[0]] = p

    return xr.Dataset(
        data_vars={
            "positions": (["play", "time", "player", "coord"], data)
        },
        coords={
            "play": np.arange(P),
            "time": np.arange(T),
            "player": np.arange(10),
            "coord": ["x", "y"]
        }
    )

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

    def compute_W_matrix(X, Y):
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

        W = np.zeros((n, m))

        # Calculate D(i, j) which is the Gromov-Wasserstein distance between the distributions of distances

        for i in range(n):
            Xi_distances = X_dists[i]  # vector of length n
            for j in range(m):
                Yj_distances = Y_dists[j]  # vector of length m

                # Gromov-Wasserstein between the two empirical distributions
                C1 = ot.dist(Xi_distances)
                C2 = ot.dist(Yj_distances)
                M = ot.dist(X[i:i+1], Y[j:j+1])  # cost matrix between single points
                p = ot.unif(n)
                q = ot.unif(m)
                Gwg, logw = fused_gromov_wasserstein(M, C1, C2, p, q, loss_fun="square_loss", alpha=1e-3, verbose=True, log=True)
                W[i, j] = Gwg

        map_matrix = emd(np.ones(n) / n, np.ones(m) / m, W)

        return W, map_matrix

