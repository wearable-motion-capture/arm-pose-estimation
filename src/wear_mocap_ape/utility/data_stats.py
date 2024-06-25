import datetime
import logging
from pathlib import Path
import pickle
import numpy as np

from wear_mocap_ape.utility.names import NNS_INPUTS, NNS_TARGETS
from wear_mocap_ape import config


def get_norm_stats(x_inputs: NNS_INPUTS, y_targets: NNS_TARGETS, data_list: list = None) -> dict:
    """
    Check if normalization stats for given params dict and m_data_list exist.
    If none exists, creates a dictionary with stats to normalize data columns:
    {
        time : datetime of dict creation (dt.now())
        x_inputs : names of input columns
        y_targets : names of target columns
        data_list_len : length of m_data_list
        xx_m : input columns means
        xx_s : input columns stds
        yy_m : target columns means
        yy_s : target columns stds
    }
    :param x_inputs:
    :param y_targets:
    :param data_list: list of files to analyze
    :return: stats dictionary
    """
    f_name = "{}_{}.pkl".format(x_inputs.name, y_targets.name)
    f_dir = Path(config.PATHS["deploy"]) / "data_stats"
    f_path = f_dir / f_name

    if f_path.exists():
        with open(f_path, 'rb') as handle:
            logging.info("loaded data stats from {}".format(f_path))
            dat = pickle.load(handle)

            if data_list is None:
                return dat
            else:
                len_m_data = len(data_list)
                if dat["data_list_len"] == len_m_data:
                    return dat
                else:
                    logging.info("number of files changed")

    # create new stats file one
    logging.info("creating stats file {}".format(f_path))
    if data_list is None:
        raise UserWarning("attempting to create stats file without data list")

    # estimate mean and std of complete data columns
    x_inputs_v = x_inputs.value
    y_targets_v = y_targets.value
    xxs, yys = [], []
    for m_data in data_list:
        m_data.dropna(inplace=True)
        xx = m_data.loc[:, x_inputs_v].to_numpy()
        yy = m_data.loc[:, y_targets_v].to_numpy()
        xxs.append(xx)
        yys.append(yy)
    xx = np.vstack(xxs)
    yy = np.vstack(yys)

    # replace stds of 0 with 1 for no change
    xx_s = xx.std(axis=0)
    xx_s[xx_s == 0] = 1.0
    yy_s = yy.std(axis=0)
    yy_s[yy_s == 0] = 1.0

    # save in a dictionary
    stats = {
        "time": datetime.datetime.now(),
        "x_inputs": x_inputs.value,
        "y_targets": y_targets.value,
        "data_list_len": len(data_list),
        "xx_m": xx.mean(axis=0),
        "xx_s": xx_s,
        "yy_m": yy.mean(axis=0),
        "yy_s": yy_s
    }

    # create directory if cache dir is empty
    if not f_dir.exists():
        f_dir.mkdir()

    # Store data on file (serialize)
    with open(f_path, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("saved data stats to {}".format(f_path))
    return stats


def get_norm_and_one_hot_stats(x_inputs: NNS_INPUTS,
                               y_targets: NNS_TARGETS,
                               data_list: list = None,
                               force_new: bool = False) -> dict:
    """
    Check if normalization stats for given params dict and m_data_list exist.
    If none exists, creates a dictionary with stats to normalize data columns:
    {
        time : datetime of dict creation (dt.now())
        x_inputs : names of input columns
        y_targets : names of target columns
        data_list_len : length of m_data_list
        xx_m : input columns means
        xx_s : input columns stds
        yy_len : number of label columns
        yy_names : names of label columns
        yy_map : a dictionary for labels and one-hot positions
    }
    :param x_inputs:
    :param y_targets:
    :param data_list: list of files to analyze
    :param force_new: force to create a new file
    :return: stats dictionary
    """

    f_name = "{}_1hot_{}.pkl".format(x_inputs.name, y_targets.name)
    f_dir = Path(config.PATHS["deploy"]) / "data_stats"
    f_path = f_dir / f_name

    if f_path.exists():
        with open(f_path, 'rb') as handle:
            logging.info("loaded data stats from {}".format(f_path))
            dat = pickle.load(handle)

            if data_list is None:
                return dat
            elif force_new:
                logging.info("force new stats file")
            else:
                # check if amount of data has increased
                if dat["data_list_len"] == len(data_list):
                    return dat
                else:
                    logging.info("number of files changed")

    # create new stats file one
    logging.info("creating stats file {}".format(f_path))
    if data_list is None or not data_list:
        raise UserWarning("Make sure to provide a data list if you try to load a stats file that does not exist")

    # estimate mean and std of complete data columns
    x_inputs_v = x_inputs.value
    y_targets_v = y_targets.value
    xxs = []
    yys = {}
    for m_data in data_list:
        m_data.dropna(inplace=True)
        xx = m_data.loc[:, x_inputs_v].to_numpy().astype(np.float32)
        xxs.append(xx)
        yy = m_data.loc[:, y_targets_v]
        yy.drop_duplicates(inplace=True)
        for v in yy.values:
            if v not in yys:
                yys[v] = len(yys)

    # replace stds of 0 with 1 for no change
    xx = np.vstack(xxs)
    xx_s = xx.std(axis=0)
    xx_s[xx_s == 0] = 1.0

    # save in a dictionary
    stats = {
        "time": datetime.datetime.now(),
        "x_inputs": x_inputs.value,
        "y_targets": y_targets.value,
        "data_list_len": len(data_list),
        "xx_m": xx.mean(axis=0),
        "xx_s": xx_s,
        "yy_m": 0.0,
        "yy_s": 1.0,
        "yy_len": len(yys),
        "yy_names": list(yys.keys()),
        "yy_map": yys
    }

    # create directory if cache dir is empty
    if not f_dir.exists():
        f_dir.mkdir()

    # Store data on file (serialize)
    with open(f_path, 'wb') as handle:
        pickle.dump(stats, handle)
    logging.info("saved data stats to {}".format(f_path))
    return stats
