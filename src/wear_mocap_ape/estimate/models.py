import hashlib
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

import wear_mocap_ape.config as config


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
                ) * 4 * channel_mult
            ),
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


class OneHotLSTM(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_layer_size,
                 hidden_layer_count,
                 output_size):
        super().__init__()

        self.lstm = torch.nn.LSTM(input_size,
                                  hidden_size=hidden_layer_size,
                                  num_layers=hidden_layer_count,
                                  batch_first=True)
        self.output_layer = torch.nn.Linear(hidden_layer_size, output_size)
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_count = hidden_layer_count

    def forward(self, x, hs=None):
        """
        :param x: expects data as a 3D tensor [batch_size, sequence, input]
        :param hs: Defaults to zeros if (h_0, c_0) is not provided
        :return:
        """
        yp, _ = self.lstm(x, hs)
        return self.output_layer(yp)


class OneHot2DFF(torch.nn.Module):

    def __init__(self,
                 output_size,
                 hidden_layer_size,
                 hidden_layer_count,
                 input_size,
                 seq_len):
        super(OneHot2DFF, self).__init__()

        self.output_size = output_size
        self._input_layer = torch.nn.Linear(input_size * seq_len, hidden_layer_size)
        self._activation_function = F.leaky_relu
        # add hidden layers according to count variable
        self._hidden_layers = torch.nn.ModuleList()
        for hl in range(hidden_layer_count):
            self._hidden_layers.append(torch.nn.Linear(hidden_layer_size, hidden_layer_size))
        # end with dropout and then two final layers
        self._output_layer = torch.nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        """
        simple forward pass
        :param x: input
        :return: output layer
        """
        xf = torch.flatten(x, start_dim=1)
        x = self._activation_function(self._input_layer(xf))
        # propagate stacked hidden layers
        for h in self._hidden_layers:
            x = self._activation_function(h(x))
        # final layer with a sequence length of 1
        x = self._output_layer(x)[:, None, :]
        return x


class OneHotFF(torch.nn.Module):
    def __init__(self,
                 output_size,
                 hidden_layer_size,
                 hidden_layer_count,
                 input_size):
        super(OneHotFF, self).__init__()

        self.output_size = output_size
        self._input_layer = torch.nn.Linear(input_size, hidden_layer_size)
        self._activation_function = F.sigmoid

        # add hidden layers according to count variable
        self._hidden_layers = torch.nn.ModuleList()
        for hl in range(hidden_layer_count):
            self._hidden_layers.append(torch.nn.Linear(hidden_layer_size, hidden_layer_size))

        # end with dropout and then two final layers
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
        # final layer
        x = self._output_layer(x)
        # x = F.log_softmax(x, dim=2)
        return x


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
        :param x: expects data as a 3D tensor [batch_size, sequence, input]
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
        :param x: expects data as a 3D tensor [batch_size, sequence, input]
        :param n_samples: mc samples are the number of repetitions
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


class ImuPoseLSTM(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_layer_size,
                 hidden_layer_count,
                 output_size,
                 dropout=0.2):
        """
        We kept the unused parameters to be compatible with our regular training code
        """

        super().__init__()
        self.input_layer = torch.nn.Linear(input_size, 256)
        self.lstm = torch.nn.LSTM(256,
                                  hidden_size=256,
                                  num_layers=2,
                                  batch_first=True,
                                  bidirectional=False,
                                  dropout=dropout)
        self._activation_function = F.relu
        self.output_layer = torch.nn.Linear(256, output_size)
        self.output_size = output_size
        self.input_size = input_size

    def forward(self, x, hs=None):
        """
        :param x: expects data as a 3D tensor [batch_size, sequence, input]
        :param hs: Defaults to zeros if (h_0, c_0) is not provided
        :return:
        """
        x = self._activation_function(self.input_layer(x))
        a, hs_n = self.lstm(x, hs)
        return self.output_layer(a)

    def monte_carlo_predictions(self, n_samples: int, x: torch.tensor):
        """
        This model does not have a dropout layer, therefore MC predictions are the same as regular forward passes.
        We kept this function such that it is compatible with our evaluation code
        """
        return self(x, None)


class DropoutFF2D(torch.nn.Module):

    def __init__(self,
                 output_size,
                 hidden_layer_size,
                 hidden_layer_count,
                 input_size,
                 seq_len,
                 dropout=0.2):
        super(DropoutFF2D, self).__init__()

        self.output_size = output_size
        self._input_layer = torch.nn.Linear(input_size * seq_len, hidden_layer_size)
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

        xf = torch.flatten(x, start_dim=1)

        x = self._activation_function(self._input_layer(xf))
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
        :param n_samples:
        :param x:
        :return:
        """
        x = torch.flatten(x, start_dim=1)
        # mc predictions require the dropout layers.
        # Make sure training mode is activated to apply dropout
        self._do.train()
        # (num_samples * batch_size (1), seq_len, input_size)
        rep_data = x.repeat((n_samples, 1, 1))
        return self(rep_data)


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
        :param n_samples:
        :param x:
        :return:
        """
        # mc predictions require the dropout layers.
        # Make sure training mode is activated to apply dropout
        self._do.train()
        # (num_samples * batch_size (1), seq_len, input_size)
        rep_data = x.repeat((n_samples, 1, 1))
        return self(rep_data)


def load_deployed_model_from_hash(hash_str: str):
    """
    :param hash_str: the model hash is in its directory name and created from the params JSON
    :return: (loaded_model, params_dict)
    """
    # compose paths
    save_path = Path(config.PATHS["deploy"]) / "nn" / hash_str
    json_path = save_path / "results.json"
    chkpt_path = save_path / "checkpoint.pt"

    # check if files exist
    if not json_path.exists():
        raise UserWarning(f"no json found {json_path}")
    if not chkpt_path.exists():
        raise UserWarning(f"no checkpoint found {chkpt_path}")

    # load and parse json
    with open(json_path, 'r') as f:
        params = json.load(f)

    if params["model"] == "DropoutLSTM":
        params["model"] = DropoutLSTM
    elif params["model"] == "DropoutFF":
        params["model"] = DropoutFF
    elif params["model"] == "ImuPoseLSTM":
        params["model"] = ImuPoseLSTM
    else:
        raise UserWarning(f"{params['model']} not handled")

    nn_model = params["model"](
        input_size=len(params["x_inputs_v"]),
        hidden_layer_size=params["hidden_layer_size"],
        hidden_layer_count=params["hidden_layer_count"],
        output_size=len(params["y_targets_v"]),
        dropout=params["dropout"]
    )

    model_state, _ = torch.load(chkpt_path, map_location="cpu")  # model_state, optimizer_state
    nn_model.load_state_dict(model_state)
    # if we load the model, we are typically not planning on training it further
    nn_model.eval()
    logging.info("loaded model in eval mode from {}".format(save_path))
    return nn_model, params


def get_nn_name(params):
    """simply hashes the entire parameters dictionary"""
    name = str(params)
    # some names become too long this way. Therefore, we hash them
    sha1 = hashlib.sha1()
    sha1.update(name.encode("utf-8"))
    return str(sha1.hexdigest())
