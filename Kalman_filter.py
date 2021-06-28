import numpy as np


class KalmanF(object):

    def __init__(self, state_var=1, val_var=1, dt=1, method='velocity'):

        super(KalmanF, self).__init__()
        self.state_var = state_var
        self.val_var = val_var
        self.method = method
        self.dt = dt
        self.U = 0
        self.init_model()

    def init_model(self):
        if self.method != 'velocity':
            self.U = 1

        self.A = np.mat([[1, self.dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.dt], [0, 0, 0, 1]])

        self.B = np.mat([[self.dt**2/2], [self.dt], [self.dt**2/2], [self.dt]])

        self.H = np.mat([[1, 0, 0, 0], [0, 0, 1, 0]])

        self.P = np.mat(self.state_var * np.identity(self.A.shape[0]))
        self.R = np.mat(self.val_var * np.identity(self.H.shape[0]))

        self.Q = np.mat([[self.dt**4/4, self.dt**3/2, 0, 0],
                         [self.dt**3/2, self.dt**2, 0, 0],
                         [0, 0, self.dt**4/4, self.dt**3/2],
                         [0, 0, self.dt**3/2, self.dt**2]])

        self.state = np.mat([[0], [1], [0], [1]])
        self.error = self.P

    def predict(self):
        self.predictState = self.A * self.state + self.B * self.U
        self.predictError = self.A*self.error*self.A.T + self.Q
        temp = np.asarray(self.predictState)
        return temp[0], temp[2]

    def correct(self, val_current):
        self.kalmanGain = self.predictError*self.H.T*np.linalg.pinv(self.H*self.predictError*self.H.T+self.R)

        self.state = self.predictState + self.kalmanGain*(val_current-(self.H*self.predictState))
        self.error = (np.identity(self.P.shape[0])-self.kalmanGain*self.H)*self.predictError






