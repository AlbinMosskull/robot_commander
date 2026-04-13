import numpy as np


class KalmanFilter:
    def __init__(
        self,
        F: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
    ):
        self.F = F
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self, u: np.ndarray) -> None:
        # Propagate state estimate using motion model
        self.x = self.F @ self.x + self.B @ u

        # Propagate uncertainty — grows due to process noise Q
        self.P = self.F @ self.P @ self.F.transpose() + self.Q

    def update(self, z: np.ndarray) -> np.ndarray:
        # Compute residual between measurement and prediction
        y = z - self.H @ self.x

        # Compute residual covariance
        S = self.H @ self.P @ self.H.transpose() + self.R

        # Compute Kalman gain
        K = np.linalg.solve(S.T, self.H @ self.P.T).T

        # Update state estimate
        self.x = self.x + K @ y

        # Update covariance — shrinks after incorporating measurement
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

        return y

