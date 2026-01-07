import numpy as np

# JSON -> Tensor (4D array: Time x Players x (x,y))
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


