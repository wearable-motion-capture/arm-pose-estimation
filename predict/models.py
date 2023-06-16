import hashlib
import logging
import os
import random

import torch
import torch.nn.functional as F

import config


class PPGOneHotFF(torch.nn.Module):
    """
    PPG specific Feed Forward model
    """

    def __init__(self,
                 concise: bool,
                 sequence_len: int = 50,
                 output_size: int = 2):
        super().__init__()
        input_size = 2 if concise else 13
        self.hl1 = torch.nn.Linear(input_size * sequence_len, 1024)
        self.hl2 = torch.nn.Linear(1024, 512)
        self.hl3 = torch.nn.Linear(512, 32)
        self.hl4 = torch.nn.Linear(32, output_size)
        self.dropout = 0.1

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.hl1(x))
        x = F.leaky_relu(self.hl2(x))
        x = F.dropout(x, self.dropout)
        x = F.leaky_relu(self.hl3(x))
        return self.hl4(x)


class PPGOneHot1dCNN(torch.nn.Module):
    """
    PPG specific 1D CNN model
    """

    def __init__(self,
                 concise: bool,
                 sequence_len: int = 50,
                 output_size: int = 2):
        super().__init__()
        input_size = 2 if concise else 13
        self.conv1 = torch.nn.Conv1d(in_channels=input_size,
                                     out_channels=input_size,
                                     kernel_size=3,
                                     stride=1)
        self.hl1 = torch.nn.Linear(int((sequence_len - 2) * input_size), 512)
        self.hl2 = torch.nn.Linear(512, 32)
        self.hl3 = torch.nn.Linear(32, output_size)
        self.dropout = 0.1

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.hl1(x))
        x = F.leaky_relu(self.hl2(x))
        return self.hl3(x)


class OneHotCNN(torch.nn.Module):
    def __init__(self,
                 input_size,
                 sequence_len,
                 output_size,
                 channel_mult=1):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=2 * channel_mult,
                                     kernel_size=3,
                                     stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=2 * channel_mult,
                                     out_channels=channel_mult,
                                     kernel_size=3,
                                     stride=1)
        self.hl1 = torch.nn.Linear(
            int((sequence_len - 4) * int((input_size - 4)) * channel_mult),
            128)
        # self.hl2 = torch.nn.Linear(1024, 128)
        self.hl3 = torch.nn.Linear(128, output_size)
        self.dropout = 0.1

    def forward(self, x):
        """
        expects data as a 3D tensor [batch_size, sequence, input]
        :param x:
        :return:
        """
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        # x = F.dropout(x, self.dropout)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.hl1(x))
        return self.hl3(x)


class DropoutCNN(torch.nn.Module):
    def __init__(self,
                 input_size,
                 sequence_len,
                 channel_mult,
                 output_size,
                 dropout=0.2):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=2 * channel_mult,
                                     kernel_size=3,
                                     stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=2 * channel_mult,
                                     out_channels=4 * channel_mult,
                                     kernel_size=3,
                                     stride=1)
        self.hl1 = torch.nn.Linear(
            int(
                (sequence_len - 4)  # * 0.5
                * int(
                    (input_size - 4)  # * 0.5
                ) * 4 * channel_mult),
            1024
        )
        self.hl2 = torch.nn.Linear(1024, 128)
        self.hl3 = torch.nn.Linear(128, output_size)
        self.dropout = dropout

    def forward(self, x):
        """
        expects data as a 3D tensor [batch_size, sequence, input]
        :param x:
        :return:
        """
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        # x = F.max_pool2d(x, 2)
        x = F.dropout(x, self.dropout)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.hl1(x))
        x = F.dropout(x, self.dropout)
        x = F.leaky_relu(self.hl2(x))
        x = self.hl3(x)
        return x


class DropoutFF(torch.nn.Module):
    """
    this model is our own implementation for Monte Carlo predictions. It has a dropout at the output layer that can be
    activated outside of training to create a distribution of predictions.
    """

    def __init__(self,
                 output_size,
                 hidden_layer_size,
                 hidden_layer_count,
                 input_size,
                 dropout=0.2):
        super(DropoutFF, self).__init__()

        self.output_size = output_size
        self._input_layer = torch.nn.Linear(input_size, hidden_layer_size)
        self._activation_function = F.leaky_relu

        # add hidden layers according to count variable
        self._hidden_layers = torch.nn.ModuleList()
        for hl in range(hidden_layer_count):
            self._hidden_layers.append(torch.nn.Linear(hidden_layer_size, hidden_layer_size))

        # end with dropout and then two final layers
        self._do = torch.nn.Dropout(dropout)
        self._output_layer = torch.nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        """
        simple forward pass
        :param x: input
        :return: output layer
        """
        x = self._activation_function(self._input_layer(x))
        # propagate stacked hidden layers
        for h in self._hidden_layers:
            x = self._activation_function(h(x))
        # apply dropout
        x = self._do(x)
        # final layer
        x = self._output_layer(x)
        return x

    def monte_carlo_predictions(self,
                                n_samples: int,
                                x: torch.tensor):
        """
        Expects data as a 2D tensor [batch_size, input_data]
        :param data:
        :param n_samples:
        :return:
        """
        # mc predictions require the dropout layers.
        # Make sure training mode is activated to apply dropout
        self._do.train()
        # repeat data such that it becomes a 3D Tensor (n_samples, batch_size, input_size)
        rep_data = x.repeat((n_samples, 1, 1))
        return self(rep_data)


class DropoutLSTM(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_layer_size,
                 hidden_layer_count,
                 output_size,
                 dropout=0.2):
        super().__init__()

        self.lstm = torch.nn.LSTM(input_size,
                                  hidden_size=hidden_layer_size,
                                  num_layers=hidden_layer_count,
                                  batch_first=True,
                                  dropout=dropout)
        self.output_layer = torch.nn.Linear(hidden_layer_size, output_size)
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_count = hidden_layer_count

    def forward(self, x, hs=None):
        """
        expects data as a 3D tensor [batch_size, sequence, input]
        :param x:
        :param hs: Defaults to zeros if (h_0, c_0) is not provided
        :return:
        """
        # all: all hidden states of every step in the given sequence (if a sequence was put in)
        # (h_n,c_n) : the last hidden state and internal state (the final prediction of the sequence)
        all, hs_n = self.lstm(x, hs)
        return self.output_layer(all)

    def monte_carlo_predictions(self,
                                n_samples: int,
                                x: torch.tensor,
                                hs: torch.tensor = None):
        """
        expects data as a 3D tensor [batch_size, sequence, input]
        :param x:
        :param n_samples:
        :param hs: Defaults to zeros if (h_0, c_0) is not provided
        :return:
        """
        if x.shape[0] > 1:
            raise UserWarning("MC predictions only for batch size 1")
        # activate training mode to use dropout layers
        self.lstm.train()
        # (num_samples * batch_size (1), seq_len, input_size)
        rep_data = x.repeat((n_samples, 1, 1))
        return self(rep_data, hs)


class SplitLSTM(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_layer_size,
                 hidden_layer_count,
                 output_size,
                 dropout=0.2):
        super().__init__()

        self.lstm = torch.nn.LSTM(input_size,
                                  hidden_size=hidden_layer_size,
                                  num_layers=hidden_layer_count,
                                  batch_first=True,
                                  dropout=dropout)
        self.output_layer = torch.nn.Linear(hidden_layer_size, output_size)
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_count = hidden_layer_count
        self.mask_thresh = 0.5

    def forward(self, x, hs=None):
        """
        expects data as a 3D tensor [batch_size, sequence, input]
        :param x:
        :return:
        """
        dr, rot, lacc, x_arm = torch.tensor_split(x, [2, 8, 11], dim=-1)

        if self.training:
            # mask rot values at random to emphasize training on other sensor measurements
            if random.random() > self.mask_thresh:
                rot = rot * 0.

        x = torch.concat([dr, rot, lacc, x_arm], dim=-1)

        # all: all hidden states of every step in the given sequence (if a sequence was put in)
        # (h_n,c_n) : the last hidden state and internal state (the final prediction of the sequence)
        all, hs_n = self.lstm(x, hs)
        # hn is of shape (num_layers,batch_size,hidden_size)
        # values of the final layer are our prediction
        # l_hn = h_n[-1, :, :]
        # apply activation function
        fl_hs = F.leaky_relu(self.output_layer(all))
        return fl_hs, hs_n

    def monte_carlo_predictions(self,
                                n_samples: int,
                                x: torch.tensor,
                                hs: torch.tensor = None):
        """
        expects data as a 3D tensor [batch_size, sequence, input]
        :param x:
        :param n_samples:
        :return:
        """
        if x.shape[1] > 1:
            raise UserWarning("MC predictions only for sequence len 1")
        if x.shape[0] > 1:
            raise UserWarning("MC predictions only for batch size 1")
        # mc predictions require the dropout layers to be active.
        # Therefore, training should be activated for layers with dropout
        self.lstm.train()
        # (num_samples * batch_size (1), seq_len(1), input_size)
        rep_data = x.repeat((n_samples, 1, 1))
        y, hs = self(rep_data, hs)
        # (num_samples * batch_size, seq_len, output_size)
        resh_pred = y.reshape((n_samples, 1, self.output_size))
        return resh_pred, hs


def load_model_from_params(params, save_path=os.path.join(config.paths["deploy"], "nn")):
    """
    use this function to load a model from file. They are stored under a hash that has to be recreated.
    :param params: a parameters dictionary
    :param save_path: custom save path to load the data from (defaults to the deploy folder)
    :return: the loaded torch model
    """

    nn_model = params["model"](
        input_size=len(params["x_inputs"].value),
        hidden_layer_size=params["hidden_layer_size"],
        hidden_layer_count=params["hidden_layer_count"],
        output_size=len(params["y_targets"].value),
        dropout=params["dropout"]
    )

    # load hash from params if available, otherwise generate one
    name = get_nn_name(params)

    if "checkpoint.pt" in os.listdir(save_path):
        full_path = os.path.join(save_path, "checkpoint.pt")
    elif name in os.listdir(save_path):
        full_path = os.path.join(save_path, name, "checkpoint.pt")
    else:
        raise UserWarning("No saved model found under checkpoint.pt or {} in {}".format(name, save_path))

    model_state, _ = torch.load(full_path, map_location="cpu")  # model_state, optimizer_state
    nn_model.load_state_dict(model_state)
    # if we load the model, we are typically not planning on training it further
    nn_model.eval()
    logging.info("loaded model in eval mode from {}".format(full_path))
    return nn_model


def get_nn_name(params):
    """simply hashes the entire parameters dictionary"""
    name = str(params)
    # some names become too long this way. Therefore, we hash them
    sha1 = hashlib.sha1()
    sha1.update(name.encode("utf-8"))
    return str(sha1.hexdigest())
