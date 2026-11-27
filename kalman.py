import numpy as np

class Kalman3D():
    def __init__(self):
 
        ### Initialize Kalman filter ###
        # The initial state (6x1).
        self.state = np.array([[0], # Position along the x-axis
                    [0], # Velocity along the x-axis
                    [0], # Position along the y-axis
                    [0], # Velocity along the y-axis
                    [0], # Position along the z-axis
                    [0], # Velocity along the z-axis
                    ])

        # The initial uncertainty (6x6).
        self.P = np.diag([1000,1000,1000,1000,1000,1000]) #Starts at 1000 just like in the exercise

        # The external motion (6x1).
        self.u = np.zeros((6,1)) #Set to 0 here

        # The transition matrix (6x6). 
        self.F = np.array([[1, 1, 0, 0, 0 ,0],
                    [0, 1, 0, 0, 0 ,0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1 ,1],
                    [0, 0, 0, 0, 0, 1]])

        # The observation matrix (3x6).
        # We only get the position as measurement.
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0]])

        # The measurement uncertainty.
        self.R = np.diag([3.6e-4, 9.1e-3, 3.1e-3])

        # The process noise (6x6), one [position, velocity] block per axis.
        q = 1e-5
        self.Q = q * np.kron(np.eye(3), [[1/4, 1/2],
                                         [1/2, 1  ]])

    def update(self, measurement, variance=None):
        ### Insert update function
        if variance is not None:
            self.R = np.diag(variance)
        measurement = np.reshape(measurement, (3, 1))
        Y = measurement - self.H@self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.state = self.state + K @ Y
        self.P = (np.identity(len(self.state)) - K @ self.H) @ self.P
        return self.state, self.P

    def predict(self):
        ### insert predict function
        self.state = self.F @ self.state + self.u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state, self.P
        

class Kalman2D():
    def __init__(self, sigma_px=3.0, q=0.1):
        # The initial state (4x1), overwritten by the first measurement (see update).
        self.state = np.zeros((4, 1))

        # The initial uncertainty (4x4).
        self.P = np.diag([1000.0, 1000.0, 1000.0, 1000.0])

        # The external motion (4x1).
        self.u = np.zeros((4, 1))

        # The transition matrix (4x4): position += velocity each frame.
        self.F = np.array([[1, 1, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 1],
                           [0, 0, 0, 1]])

        # The observation matrix (2x4): we only measure the box centre (u, v).
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])

        # The measurement uncertainty: box-centre jitter of sigma_px pixels (same 3 px as
        # sigma_u/sigma_v in getDepthImage.measurement_variance).
        self.R = np.diag([sigma_px ** 2, sigma_px ** 2])

        # The process noise (4x4), in px^2 - NOT the 1e-5 of Kalman3D, which is in m^2.
        # Larger q follows turns faster but jitters more; tune on sequences 3 and 4.
        self.Q = q * np.kron(np.eye(2), [[1/4, 1/2],
                                         [1/2, 1  ]])

        self.initialized = False

    def update(self, measurement):
        z = np.reshape(np.asarray(measurement, dtype=float), (2, 1))
        if not self.initialized:
            # Start at the first detection instead of at pixel (0, 0)
            self.state[[0, 2]] = z
            self.P = np.diag([self.R[0, 0], 1000.0, self.R[1, 1], 1000.0])
            self.initialized = True
            return self.state, self.P
        Y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ Y
        self.P = (np.identity(len(self.state)) - K @ self.H) @ self.P
        return self.state, self.P

    def predict(self):
        self.state = self.F @ self.state + self.u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state, self.P




