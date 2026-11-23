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

    def update(self, measurement):
        ### Insert update function
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
        



