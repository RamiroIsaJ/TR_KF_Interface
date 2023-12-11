import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
from Kalman_filter import KalmanF


class Tracks(object):

    def __init__(self, detection, track_id, delta):
        super(Tracks, self).__init__()
        self.track_id = track_id
        self.kalman = KalmanF(dt=delta, method='velocity')
        self.kalman.predict()
        self.kalman.correct(np.array(detection).reshape(2, 1))
        self.prediction = detection.reshape(1, 2)
        self.trace = deque(maxlen=20)
        self.skip_frames = 0

    def predict(self, detection):
        self.prediction = np.array(self.kalman.predict()).reshape(1, 2)
        self.kalman.correct(detection.reshape(2, 1))


class Tracker(object):

    def __init__(self, dist_thresh, frames_skip, max_trace):
        super(Tracker, self).__init__()
        self.dist_thresh = dist_thresh
        self.max_frames_skip = frames_skip
        self.max_trace = max_trace
        self.tracks = []
        self.track_id = 0

    def update(self, detections, delta):

        if len(self.tracks) == 0:
            for i in range(detections.shape[0]):
                track = Tracks(detections[i], self.track_id, delta)
                self.track_id += 1
                self.tracks.append(track)

        cost = []
        for i in range(len(self.tracks)):
            diff = np.linalg.norm(self.tracks[i].prediction - detections.reshape(-1, 2), axis=1)
            cost.append(diff)

        cost = np.array(cost) * 0.1
        row, col = linear_sum_assignment(cost)
        assignment = [-1] * len(self.tracks)
        for i in range(len(row)):
            assignment[row[i]] = col[i]

        no_assigned_tracks = []

        for i in range(len(assignment)):
            if assignment[i] != -1:
                if cost[i][assignment[i]] > self.dist_thresh:
                    assignment[i] = -1
                    no_assigned_tracks.append(i)
                else:
                    self.tracks[i].skip_frames += 1

        del_tracks = []
        for i in range(len(self.tracks)):
            if self.tracks[i].skip_frames > self.max_frames_skip:
                del_tracks.append(i)

        if len(del_tracks) > 0:
            for i in range(len(del_tracks)):
                del self.tracks[i]
                del assignment[i]

        for i in range(len(detections)):
            if i not in assignment:
                track = Tracks(detections[i], self.track_id)
                self.track_id += 1
                self.tracks.append(track)

        for i in range(len(assignment)):
            if assignment[i] != -1:
                self.tracks[i].skip_frames = 0
                self.tracks[i].predict(detections[assignment[i]])
            self.tracks[i].trace.append(self.tracks[i].prediction)
