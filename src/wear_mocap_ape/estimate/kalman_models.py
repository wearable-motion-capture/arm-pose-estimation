from bayesian_torch.layers.flipout_layers.linear_flipout import LinearFlipout

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class ProcessModelWindow(nn.Module):
    """
    Takes a window (sequence) of previous states to predict the next state.
    LinearFlipout layers produce solution ensembles.
    Therefore, the input and output dimensions of the forward pass are:
    Input: [batch_size, ensemble_size, window_size, dim_x(features)]
    Output: [batch_size, ensemble_size, dim_x(features)]
    """

    def __init__(self, num_ensemble: int, dim_x: int, win_size: int):
        """
        Instantiates a ProcessModelWindow object. See the class description for details.
        :param num_ensemble: number of ensembles
        :param dim_x: number of state vars
        :param win_size: window size. Number of previous states to predict the next.
        """
        super(ProcessModelWindow, self).__init__()

        # store these params to speed up the forward pass
        self._num_ensemble = num_ensemble
        self._dim_x = dim_x
        self._total_win_x = dim_x * win_size

        # kept the old layer names to be compatible with legacy checkpoints
        self.bayes1 = LinearFlipout(in_features=self._total_win_x, out_features=256)
        self.bayes3 = LinearFlipout(in_features=256, out_features=512)
        self.bayes_m2 = torch.nn.Linear(512, dim_x)

    def forward(self, x):
        """
        :param x: window of previous states [batch_size, ensemble_size, window_size, dim_x]
        :return: predicted state_t [batch_size, ensemble_size, dim_x]
        """
        bs = x.size(0)  # keep the batch size for the return
        x = torch.reshape(x, (bs * self._num_ensemble, self._total_win_x))
        x, _ = self.bayes1(x)
        x = nn.functional.leaky_relu(x)
        x, _ = self.bayes3(x)
        x = nn.functional.leaky_relu(x)
        x = self.bayes_m2(x)
        output = torch.reshape(x, (bs, self._num_ensemble, self._dim_x))
        return output


class ObservationNoise(nn.Module):
    def __init__(self, dim_z: int):
        """
        The observation noise model is used to learn the observation noise covariance matrix
        R from the learned observation. The kalman filter requires an explicit matrix for R.
        Therefore, we construct the diag of R to model the noise here.
        input -> [batch_size, 1, dim_z]
        output -> [batch_size, dim_z, dim_z]
        """
        super(ObservationNoise, self).__init__()

        self._dim_z = dim_z

        # prepare the constants for retrieving the covariance
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._init = torch.ones(dim_z, device=device) * 0.038729833  # (r_diag ** 2 - const)
        self._constant = torch.ones(dim_z, device=device) * 1e-3

        # kept the old layer names to be compatible with legacy checkpoints
        self.fc1 = nn.Linear(dim_z, 32)
        self.fc2 = nn.Linear(32, dim_z)

    def forward(self, x):
        bsk = x.size(0) * x.size(1)  # bs * k for the return
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.square(x + self._constant) + self._init
        x = torch.reshape(x, (bsk, self._dim_z))  # [batch_size * k, dimz]
        return torch.diag_embed(x)


class SensorModelWindow(nn.Module):
    """
    The sensor model takes the current raw observations (in this case smartwatch and phone sensor data)
    and maps it to the lower-dimensional observation space.
    The WINDOW Sensor Model takes in a window (sequence) of N previous raw observations.

    input -> [batch_size, win, 1, raw_input]
    output ->  [batch_size, ensemble_members, dim_z] (all members), [batch_size, 1, dim_z] (mean)
    """

    def __init__(self, num_ensemble: int, dim_z: int, win_size: int, raw_obs_size: int):
        super(SensorModelWindow, self).__init__()
        self._dim_z = dim_z
        self._win_size = win_size

        self._num_ensemble = num_ensemble
        self._raw_obs_size = raw_obs_size
        self._total_win_raw_obs = raw_obs_size * win_size
        self._total_win_z = dim_z * win_size

        # kept the old layer names to be compatible with legacy checkpoints
        self.fc2 = nn.Linear(raw_obs_size * win_size, 256)
        self.fc3 = LinearFlipout(256, 256)
        self.fc5 = LinearFlipout(256, 64)
        self.fc6 = LinearFlipout(64, dim_z)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Returns ensembles of observations Z and the mean observation Z average of all ensemble members
        :return : [batch_size, ensemble_members, dim_z] (all members), [batch_size, 1, dim_z] (mean)
        """
        batch_size = x.shape[0]  # bs for the return

        # repeat prediction for every ensemble member
        x = x.repeat(self._num_ensemble, 1, 1, 1)
        x = torch.reshape(x, (batch_size * self._num_ensemble, self._total_win_raw_obs))

        x = self.fc2(x)
        x = nn.functional.leaky_relu(x)
        x, _ = self.fc3(x)
        x = nn.functional.leaky_relu(x)
        x, _ = self.fc5(x)
        x = nn.functional.leaky_relu(x)
        x, _ = self.fc6(x)

        # all obs ensemble members
        x = torch.reshape(x, (batch_size, self._num_ensemble, self._dim_z))
        # mean obs. averaged ensembles
        mean_x = torch.mean(x, axis=1)[:, None, :]
        return x, mean_x


class KalmanSmartwatchModel(nn.Module):
    """
    This is where all modules come together.
    The KalmanSmartwatchModel uses the ProcessModelWindow, SensorModelWindow, and ObservationNoise.
    The process model propagates the state forward in time. The sensor and observation noise models are used to
    correct the state.
    """

    def __init__(self, num_ensemble, win_size, dim_x: int = 14, dim_z: int = 14, raw_obs_size: int = 22):
        super(KalmanSmartwatchModel, self).__init__()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_ensemble = num_ensemble
        self._dim_x = dim_x
        self.dim_z = dim_z
        self.win_size = win_size

        # instantiate modules
        self.process_model = ProcessModelWindow(
            self._num_ensemble, self._dim_x, self.win_size
        )
        self.sensor_model = SensorModelWindow(
            self._num_ensemble, self.dim_z, win_size, raw_obs_size
        )
        self.observation_noise = ObservationNoise(
            self.dim_z
        )

    def format_state(self, state: torch.Tensor):
        k = state.size(0)
        state = state.repeat(self._num_ensemble, 1, 1)
        state = torch.reshape(state, (self._num_ensemble * k, self._dim_x))
        cov = torch.eye(self._dim_x) * 0.1

        mns = MultivariateNormal(torch.zeros(self._dim_x), cov)
        init_dist = mns.sample((self._num_ensemble,)).to(self._device)

        return state + init_dist

    def forward(self, raw_obs, state_prev):
        """
        See the paper for the workings of state prediction and correction.
        """

        # 1. Prediction step
        state_pred = self.process_model(state_prev)
        # zero mean the state ensemble
        state_m = torch.mean(state_pred, axis=1)  # state_m -> [bs, dim_x]
        A = state_pred - state_m
        AT = A.transpose(1, 2)  # swap ensemble and dim_x

        # 2. Update step
        # since observation model is identity function
        H_A = A
        H_XT = state_pred.transpose(1, 2)
        H_AT = AT

        # get learned observation
        ensemble_z, z = self.sensor_model(raw_obs)

        # measurement update
        y = ensemble_z.transpose(1, 2)
        R = self.observation_noise(z)

        innovation = (1 / (self._num_ensemble - 1)) * torch.matmul(H_AT, H_A) + R
        inv_innovation = torch.linalg.inv(innovation)
        K = (1 / (self._num_ensemble - 1)) * torch.matmul(
            torch.matmul(AT, H_A), inv_innovation
        )

        gain = torch.matmul(K, y - H_XT).transpose(1, 2)
        # finally, correct the state prediction
        state_corrected = state_pred + gain

        # gather output
        m_state_corrected = torch.mean(state_corrected, axis=1)[:, None, :]
        m_state_pred = state_m[:, None, :]
        output = (
            state_corrected,
            m_state_corrected,
            m_state_pred,
            z,
            ensemble_z,
        )
        return output
