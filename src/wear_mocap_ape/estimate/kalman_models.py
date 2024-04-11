from bayesian_torch.layers.flipout_layers.linear_flipout import LinearFlipout
from torch.distributions.multivariate_normal import MultivariateNormal
from einops import rearrange, repeat
import torch
import numpy as np
import torch.nn as nn


class Utils:
    def __init__(self, num_ensemble, dim_x, dim_z):
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z

    def multivariate_normal_sampler(self, mean, cov, k):
        sampler = MultivariateNormal(mean, cov)
        return sampler.sample((k,))

    def format_state(self, state):
        state = repeat(state, "k dim -> n k dim", n=self.num_ensemble)
        state = rearrange(state, "n k dim -> (n k) dim")
        cov = torch.eye(self.dim_x) * 0.1
        init_dist = self.multivariate_normal_sampler(
            torch.zeros(self.dim_x), cov, self.num_ensemble
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        init_dist = init_dist.to(device)
        state = state + init_dist
        state = state.to(dtype=torch.float32)
        return state


class ProcessModelSeqMLP(nn.Module):
    def __init__(self, num_ensemble, dim_x, win_size, dim_model, num_heads):
        super(ProcessModelSeqMLP, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.win_size = win_size

        self.bayes1 = LinearFlipout(in_features=self.dim_x * win_size, out_features=256)
        self.bayes3 = LinearFlipout(in_features=256, out_features=512)
        self.bayes_m2 = torch.nn.Linear(512, self.dim_x)

    def forward(self, input):
        input = rearrange(input, "n en k dim -> (n en) (k dim)")
        # branch of the state
        x, _ = self.bayes1(input)
        x = nn.functional.leaky_relu(x)
        x, _ = self.bayes3(x)
        x = nn.functional.leaky_relu(x)
        x = self.bayes_m2(x)
        output = rearrange(x, "(bs en) dim -> bs en dim", en=self.num_ensemble)
        return output


class NewObservationNoise(nn.Module):
    def __init__(self, dim_z, r_diag):
        """
        observation noise model is used to learn the observation noise covariance matrix
        R from the learned observation, kalman filter require a explicit matrix for R
        therefore we construct the diag of R to model the noise here
        input -> [batch_size, 1, encoding/dim_z]
        output -> [batch_size, dim_z, dim_z]
        """
        super(NewObservationNoise, self).__init__()
        self.dim_z = dim_z
        self.r_diag = r_diag

        self.fc1 = nn.Linear(self.dim_z, 32)
        self.fc2 = nn.Linear(32, self.dim_z)

    def forward(self, inputs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        constant = np.ones(self.dim_z) * 1e-3
        init = np.sqrt(np.square(self.r_diag) - constant)
        diag = self.fc1(inputs)
        diag = nn.functional.relu(diag)
        diag = self.fc2(diag)
        diag = torch.square(diag + torch.Tensor(constant).to(device)) + torch.Tensor(
            init
        ).to(device)
        diag = rearrange(diag, "bs k dim -> (bs k) dim")
        R = torch.diag_embed(diag)
        return R


class SensorModelSeq(nn.Module):
    """
    the sensor model takes the current raw sensor (usually high-dimensional images)
    and map the raw sensor to low-dimension
    Many advanced model architecture can be explored here, i.e., Vision transformer, FlowNet,
    RAFT, and ResNet families, etc.

    input -> [batch_size, 1, win, raw_input]
    output ->  [batch_size, num_ensemble, dim_z]
    """

    def __init__(self, num_ensemble, dim_z, win_size, input_size_1):
        super(SensorModelSeq, self).__init__()
        self.dim_z = dim_z
        self.num_ensemble = num_ensemble

        self.fc2 = nn.Linear(input_size_1 * win_size, 256)
        self.fc3 = LinearFlipout(256, 256)
        self.fc5 = LinearFlipout(256, 64)
        self.fc6 = LinearFlipout(64, self.dim_z)

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "bs k en dim -> bs (k en dim)")
        x = repeat(x, "bs dim -> bs k dim", k=self.num_ensemble)
        x = rearrange(x, "bs k dim -> (bs k) dim")

        x = self.fc2(x)
        x = nn.functional.leaky_relu(x)
        x, _ = self.fc3(x)
        x = nn.functional.leaky_relu(x)
        x, _ = self.fc5(x)
        x = nn.functional.leaky_relu(x)
        encoding = x
        obs, _ = self.fc6(x)
        obs = rearrange(
            obs, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )
        obs_z = torch.mean(obs, axis=1)
        obs_z = rearrange(obs_z, "bs (k dim) -> bs k dim", k=1)
        encoding = rearrange(
            encoding, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )
        encoding = torch.mean(encoding, axis=1)
        encoding = rearrange(encoding, "(bs k) dim -> bs k dim", bs=batch_size, k=1)
        return obs, obs_z, encoding


class KalmanSmartwatchModel(nn.Module):
    def __init__(self, num_ensemble, win_size, dim_x, dim_z, input_size_1):
        super(KalmanSmartwatchModel, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.win_size = win_size
        self.r_diag = np.ones((self.dim_z)).astype(np.float32) * 0.05
        self.r_diag = self.r_diag.astype(np.float32)

        # instantiate model
        self.process_model = ProcessModelSeqMLP(
            self.num_ensemble, self.dim_x, self.win_size, 256, 8
        )
        self.sensor_model = SensorModelSeq(
            self.num_ensemble, self.dim_z, win_size, input_size_1
        )
        self.observation_noise = NewObservationNoise(self.dim_z, self.r_diag)

    def forward(self, inputs, states):
        # decompose inputs and states
        batch_size = inputs[0].shape[0]
        raw_obs = inputs
        state_old = states

        ##### prediction step #####
        state_pred = self.process_model(state_old)
        m_A = torch.mean(state_pred, axis=1)  # m_A -> [bs, dim_x]

        # zero mean
        mean_A = repeat(m_A, "bs dim -> bs k dim", k=self.num_ensemble)
        A = state_pred - mean_A
        A = rearrange(A, "bs k dim -> bs dim k")

        ##### update step #####

        # since observation model is identity function
        H_X = state_pred
        mean = torch.mean(H_X, axis=1)
        m = repeat(mean, "bs dim -> bs k dim", k=self.num_ensemble)
        H_A = H_X - m
        # transpose operation
        H_XT = rearrange(H_X, "bs k dim -> bs dim k")
        H_AT = rearrange(H_A, "bs k dim -> bs dim k")

        # get learned observation
        ensemble_z, z, encoding = self.sensor_model(raw_obs)

        # measurement update
        y = rearrange(ensemble_z, "bs k dim -> bs dim k")
        R = self.observation_noise(z)

        innovation = (1 / (self.num_ensemble - 1)) * torch.matmul(H_AT, H_A) + R
        inv_innovation = torch.linalg.inv(innovation)
        K = (1 / (self.num_ensemble - 1)) * torch.matmul(
            torch.matmul(A, H_A), inv_innovation
        )

        gain = rearrange(torch.matmul(K, y - H_XT), "bs dim k -> bs k dim")
        state_new = state_pred + gain

        # gather output
        m_state_new = torch.mean(state_new, axis=1)
        m_state_new = rearrange(m_state_new, "bs (k dim) -> bs k dim", k=1)
        m_state_pred = rearrange(m_A, "bs (k dim) -> bs k dim", k=1)
        output = (
            state_new.to(dtype=torch.float32),
            m_state_new.to(dtype=torch.float32),
            m_state_pred.to(dtype=torch.float32),
            z.to(dtype=torch.float32),
            ensemble_z.to(dtype=torch.float32),
        )
        return output
